from __future__ import annotations

from pathlib import Path

import pytest

from sell_rag.app import build_services
from sell_rag.settings import Settings


@pytest.fixture()
def services(tmp_path: Path):
    settings = Settings(
        fake_providers=True,
        runtime_dir=tmp_path,
        database=tmp_path / "sell_rag.sqlite3",
        documents_dir=tmp_path / "documents",
        indexes_dir=tmp_path / "indexes",
        admin_token="admin-test", operator_token="operator-test",
    )
    for directory in (settings.documents_dir, settings.indexes_dir):
        directory.mkdir(parents=True, exist_ok=True)
    return build_services(settings)

