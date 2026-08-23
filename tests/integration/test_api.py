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
