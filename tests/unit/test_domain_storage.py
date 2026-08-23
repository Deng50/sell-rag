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
