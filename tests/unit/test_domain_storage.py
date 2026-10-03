from __future__ import annotations

from decimal import Decimal

import pytest

from sell_rag.domain import Product, Role


def test_product_normalizes_price_and_rejects_negative_stock() -> None:
    product = Product(sku="A", name="水", price="2", stock=1)
    assert product.price == Decimal("2.00")
    with pytest.raises(ValueError):
        Product(sku="B", name="错误商品", stock=-1)


def test_document_version_duplicate_and_soft_delete(services, tmp_path) -> None:
    source = tmp_path / "doc.md"
    source.write_text("# 标题\n\n正文", encoding="utf-8")
    first = services.ingestion.ingest(source, source_id="stable-doc")
    second = services.ingestion.ingest(source, source_id="stable-doc")
    assert first["duplicate"] is False
    assert second["duplicate"] is True
    document_id = first["document"]["document_id"]
    assert services.database.active_chunks(Role.GUEST)
    services.database.soft_delete_document(document_id)
    assert services.database.active_chunks(Role.GUEST) == []


def test_new_version_can_roll_back(services, tmp_path) -> None:
    source = tmp_path / "versioned.md"
    source.write_text("第一版内容", encoding="utf-8")
    first = services.ingestion.ingest(source, source_id="versioned")
    source.write_text("第二版内容", encoding="utf-8")
    second = services.ingestion.ingest(source, source_id="versioned")
    document_id = first["document"]["document_id"]
    assert second["document"]["version"] == 2
    assert any("第二版" in item.text for item in services.database.active_chunks(Role.GUEST))
    services.database.activate_version(document_id, 1)
    assert any("第一版" in item.text for item in services.database.active_chunks(Role.GUEST))


def test_product_snapshots_follow_version_rollback_and_deletion(services, tmp_path) -> None:
    source = tmp_path / "catalog.csv"
    source.write_text("sku,name,price,stock\nA,水,2,10\nB,茶,4,5", encoding="utf-8")
    first = services.ingestion.ingest(source)
    source.write_text("sku,name,price,stock\nA,水,9,1", encoding="utf-8")
    services.ingestion.ingest(source)
    assert [(p.sku, p.price) for p in services.database.query_products()] == [("A", Decimal("9"))]
    document_id = first["document"]["document_id"]
    services.database.activate_version(document_id, 1)
    assert [(p.sku, p.stock) for p in services.database.query_products()] == [("A", 10), ("B", 5)]
    services.database.soft_delete_document(document_id)
    assert services.database.query_products() == []
    services.database.activate_version(document_id, 1)
    assert len(services.database.query_products()) == 2


def test_product_acl_is_applied_before_result_limit(services) -> None:
    services.database.upsert_product(Product(sku="private", name="内部", price=1, acl=[Role.ADMIN]))
    services.database.upsert_product(Product(sku="public", name="公开", price=2))
    assert [p.sku for p in services.database.query_products(limit=1)] == ["public"]


def test_acl_change_creates_version_and_failed_parse_can_retry(services, tmp_path, monkeypatch) -> None:
    source = tmp_path / "policy.md"
    source.write_text("可见资料", encoding="utf-8")
    original = services.ingestion.parser.parse
    monkeypatch.setattr(services.ingestion.parser, "parse", lambda _: (_ for _ in ()).throw(RuntimeError("temporary")))
    with pytest.raises(RuntimeError):
        services.ingestion.ingest(source)
    monkeypatch.setattr(services.ingestion.parser, "parse", original)
    retried = services.ingestion.ingest(source)
    assert retried["document"]["status"] == "ready"
    assert retried["document"]["version"] == 1
    private = services.ingestion.ingest(source, acl=[Role.ADMIN])
    assert private["document"]["version"] == 2
    assert services.database.active_chunks(Role.GUEST) == []


def test_duplicate_sku_from_another_source_does_not_replace_existing_product(services, tmp_path) -> None:
    source = tmp_path / "catalog.csv"
    source.write_text("sku,name,price,stock\nA,水,2,10", encoding="utf-8")
    services.ingestion.ingest(source, source_id="original")
    with pytest.raises(ValueError, match="SKU A"):
        services.ingestion.ingest(source, source_id="another", acl=[Role.ADMIN])
    assert services.database.query_products()[0].price == Decimal("2")


def test_index_failure_does_not_publish_staged_products_or_document(services, tmp_path, monkeypatch) -> None:
    source = tmp_path / "atomic.csv"
    source.write_text("sku,name,price,stock\nA,水,2,10", encoding="utf-8")
    first = services.ingestion.ingest(source)
    services.index.build()
    previous = services.database.active_index()["index_version"]
    source.write_text("sku,name,price,stock\nA,水,9,1", encoding="utf-8")
    staged = services.ingestion.ingest(source, activate=False)
    document_id = first["document"]["document_id"]
    assert services.database.query_products()[0].price == Decimal("2")
    encode = services.index.embedding.encode_documents
    monkeypatch.setattr(services.index.embedding, "encode_documents", lambda _: (_ for _ in ()).throw(RuntimeError("model offline")))
    with pytest.raises(RuntimeError, match="model offline"):
        services.index.build({document_id: staged["document"]["version"]})
    assert services.database.query_products()[0].price == Decimal("2")
    assert services.database.active_index()["index_version"] == previous
    monkeypatch.setattr(services.index.embedding, "encode_documents", encode)
    services.index.build({document_id: staged["document"]["version"]})
    assert services.database.query_products()[0].price == Decimal("9")
    assert services.database.active_index()["index_version"] != previous


def test_stale_index_publication_is_rejected(services) -> None:
    from sell_rag.storage.database import IndexConflictError

    first = services.index.build()
    second = services.index.build()
    with pytest.raises(IndexConflictError):
        services.database.save_index_version("stale", {}, True,
                                             expected_index=first["version"], verify_current=True)
    assert services.database.active_index()["index_version"] == second["version"]


def test_old_incomplete_product_snapshot_requires_reingestion(services, tmp_path) -> None:
    source = tmp_path / "legacy.csv"
    source.write_text("sku,name,price,stock\nA,水,2,10", encoding="utf-8")
    services.ingestion.PARSER_VERSION = "sell-rag-v1"
    first = services.ingestion.ingest(source)
    document_id = first["document"]["document_id"]
    with services.database.connect() as db:
        db.execute("DELETE FROM product_versions")
    with pytest.raises(ValueError, match="缺少完整商品快照"):
        services.database.activate_version(document_id, 1)
    services.ingestion.PARSER_VERSION = "sell-rag-v2"
    upgraded = services.ingestion.ingest(source)
    assert upgraded["document"]["version"] == 2
    assert services.database.query_products()[0].price == Decimal("2")
