from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Settings:
    host: str = "127.0.0.1"
    port: int = 8765
    fake_providers: bool = True
    runtime_dir: Path = Path("runtime")
    database: Path = Path("runtime/sell_rag.sqlite3")
    documents_dir: Path = Path("runtime/documents")
    indexes_dir: Path = Path("runtime/indexes")
    max_file_mb: int = 50
    child_tokens: int = 300
    parent_tokens: int = 900
    overlap_ratio: float = 0.12
    embedding_model: str = "BAAI/bge-small-zh-v1.5"
    reranker_model: str = "BAAI/bge-reranker-v2-m3"
    dense_top_k: int = 24
    lexical_top_k: int = 24
    final_top_k: int = 8
    rrf_k: int = 60
    max_context_tokens: int = 3200
    asr_confirm_threshold: float = 0.65
    presence_window: int = 8
    presence_hits: int = 5
    presence_cooldown_seconds: int = 30
    admin_token: str | None = None
    operator_token: str | None = None
    zhipuai_api_key: str | None = None
    baidu_api_key: str | None = None
    baidu_secret_key: str | None = None
    raw: dict[str, Any] = field(default_factory=dict, repr=False)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Settings":
        config_path = Path(path or os.getenv("SELL_RAG_CONFIG", "configs/default.yaml"))
        data: dict[str, Any] = {}
        if config_path.exists():
            data = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        app, storage = data.get("app", {}), data.get("storage", {})
        ingestion, retrieval = data.get("ingestion", {}), data.get("retrieval", {})
        multimodal = data.get("multimodal", {})
        fake_env = os.getenv("SELL_RAG_FAKE_PROVIDERS")
        settings = cls(
            host=app.get("host", "127.0.0.1"),
            port=int(app.get("port", 8765)),
            fake_providers=(fake_env.lower() == "true") if fake_env else bool(app.get("fake_providers", True)),
            runtime_dir=Path(storage.get("runtime_dir", "runtime")),
            database=Path(storage.get("database", "runtime/sell_rag.sqlite3")),
            documents_dir=Path(storage.get("documents_dir", "runtime/documents")),
            indexes_dir=Path(storage.get("indexes_dir", "runtime/indexes")),
            max_file_mb=int(ingestion.get("max_file_mb", 50)),
            child_tokens=int(ingestion.get("child_tokens", 300)),
            parent_tokens=int(ingestion.get("parent_tokens", 900)),
            overlap_ratio=float(ingestion.get("overlap_ratio", 0.12)),
            embedding_model=retrieval.get("embedding_model", "BAAI/bge-small-zh-v1.5"),
            reranker_model=retrieval.get("reranker_model", "BAAI/bge-reranker-v2-m3"),
            dense_top_k=int(retrieval.get("dense_top_k", 24)),
            lexical_top_k=int(retrieval.get("lexical_top_k", 24)),
            final_top_k=int(retrieval.get("final_top_k", 8)),
            rrf_k=int(retrieval.get("rrf_k", 60)),
            max_context_tokens=int(retrieval.get("max_context_tokens", 3200)),
            asr_confirm_threshold=float(multimodal.get("asr_confirm_threshold", 0.65)),
            presence_window=int(multimodal.get("presence_window", 8)),
            presence_hits=int(multimodal.get("presence_hits", 5)),
            presence_cooldown_seconds=int(multimodal.get("presence_cooldown_seconds", 30)),
            admin_token=os.getenv("SELL_RAG_ADMIN_TOKEN"),
            operator_token=os.getenv("SELL_RAG_OPERATOR_TOKEN"),
            zhipuai_api_key=os.getenv("ZHIPUAI_API_KEY"),
            baidu_api_key=os.getenv("BAIDU_API_KEY"),
            baidu_secret_key=os.getenv("BAIDU_SECRET_KEY"),
            raw=data,
        )
        for directory in [settings.runtime_dir, settings.documents_dir, settings.indexes_dir]:
            directory.mkdir(parents=True, exist_ok=True)
        settings.database.parent.mkdir(parents=True, exist_ok=True)
        return settings
