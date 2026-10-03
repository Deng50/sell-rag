from __future__ import annotations

import pytest

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from sell_rag.api import create_app


def test_health_auth_upload_and_query(services) -> None:
    client = TestClient(create_app(services=services))
    assert client.get("/health").status_code == 200
    assert client.get("/v1/documents").status_code == 403
    headers = {"X-Role": "operator", "Authorization": "Bearer operator-test"}
    upload = client.post(
        "/v1/documents", headers=headers,
        files={"file": ("knowledge.md", b"# History\n\nThe founding year was 1895.")},
    )
    assert upload.status_code == 202
    job = client.get(f"/v1/jobs/{upload.json()['job_id']}", headers=headers)
    assert job.status_code == 200
    answer = client.post("/v1/query", json={"query": "What was the founding year?"})
    assert answer.status_code == 200
    assert answer.json()["grounded"] is True
    speech = client.post("/v1/speech/synthesize", json={"text": "测试"})
    assert speech.status_code == 200


def test_upload_preserves_filename_and_stable_generated_sku(services) -> None:
    client = TestClient(create_app(services=services))
    headers = {"X-Role": "operator", "Authorization": "Bearer operator-test"}
    for price in (2, 3):
        response = client.post("/v1/documents", headers=headers,
                               files={"file": ("products.csv", f"name,price,stock\n水,{price},5".encode())})
        job = services.database.get_job(response.json()["job_id"])
        assert job["status"] == "complete", job
        products = services.database.query_products()
        assert len(products) == 1
        if price == 2:
            sku = products[0].sku
        else:
            assert products[0].sku == sku
    assert {d["filename"] for d in services.database.list_documents()} == {"products.csv"}
    assert not list((services.settings.runtime_dir / "incoming").iterdir())


def test_stream_preserves_newlines_and_completion_status(services) -> None:
    import json

    services.ingestion.ingest("tests/fixtures/products.csv")
    client = TestClient(create_app(services=services))
    query = {"query": "清泉饮用水和校园纪念杯比较价格"}
    expected = client.post("/v1/query", json=query).json()
    response = client.post("/v1/query/stream", json=query)
    events = [block.splitlines() for block in response.text.strip().split("\n\n")]
    tokens = [json.loads(lines[1][6:])["text"] for lines in events if lines[0] == "event: token"]
    assert "".join(tokens) == expected["answer"]
    assert json.loads(events[-1][1][6:])["grounded"] is True


def test_invalid_api_inputs_are_client_errors(services) -> None:
    client = TestClient(create_app(services=services))
    admin = {"X-Role": "admin", "Authorization": "Bearer admin-test"}
    assert client.post("/v1/documents/absent/versions/1/activate", headers=admin).status_code == 404
    assert client.post("/v1/query", json={"query": "   "}).status_code == 422
    assert client.get("/v1/products?min_price=nan").status_code == 422
    assert client.get("/v1/products?min_price=5&max_price=1").status_code == 400
    assert client.post("/v1/speech/transcribe", files={"audio": ("empty.pcm", b"")}).status_code == 400
    assert client.post("/v1/speech/transcribe?sample_rate=123", files={"audio": ("a.pcm", b"hi")}).status_code == 422


def test_speech_normalization_runs_at_api_boundary(services) -> None:
    from sell_rag.domain import Transcript

    services.settings.asr_confirm_threshold = 0.9
    services.speech.transcribe = lambda *_: Transcript(text=" hello world。", confidence=0.8)
    client = TestClient(create_app(services=services))
    result = client.post("/v1/speech/transcribe", files={"audio": ("a.pcm", b"hi")}).json()
    assert result["text"] == "hello world"
    assert result["needs_confirmation"] is True


def test_failed_upload_index_leaves_previous_catalog_active(services, monkeypatch) -> None:
    services.ingestion.ingest("tests/fixtures/products.csv", source_id="products.csv")
    services.index.build()
    monkeypatch.setattr(services.index, "build", lambda *args: (_ for _ in ()).throw(RuntimeError("offline")))
    headers = {"X-Role": "operator", "Authorization": "Bearer operator-test"}
    client = TestClient(create_app(services=services))
    response = client.post("/v1/documents", headers=headers,
                           files={"file": ("products.csv", b"sku,name,price,stock\nNEW,New,999,1")})
    assert services.database.get_job(response.json()["job_id"])["status"] == "failed"
    assert {p.sku for p in services.database.query_products()} == {"DRINK-001", "SNACK-001", "GIFT-001"}
