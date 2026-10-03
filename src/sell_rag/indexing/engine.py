from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import time
import threading
from collections import Counter
from pathlib import Path
from typing import Protocol

import numpy as np

from sell_rag.domain import Chunk, Role, SearchHit
from sell_rag.settings import Settings
from sell_rag.storage import Database


QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："


def tokenize(text: str) -> list[str]:
    try:
        import jieba
        return [item.strip().lower() for item in jieba.cut(text) if item.strip()]
    except ImportError:
        return re.findall(r"[\u4e00-\u9fff]|[A-Za-z0-9_.+-]+", text.lower())


class EmbeddingProvider(Protocol):
    name: str

    def encode_documents(self, texts: list[str]) -> np.ndarray: ...
    def encode_query(self, text: str) -> np.ndarray: ...


class Reranker(Protocol):
    name: str

    def score(self, query: str, texts: list[str]) -> list[float]: ...


class HashEmbedding:
    """Deterministic offline fallback. It is intentionally not presented as a semantic model."""

    name = "hash-char-ngram-v1"

    def __init__(self, dimensions: int = 384):
        self.dimensions = dimensions

    def _encode(self, text: str) -> np.ndarray:
        vector = np.zeros(self.dimensions, dtype=np.float32)
        clean = re.sub(r"\s+", "", text.lower())
        terms = tokenize(text) + [clean[i : i + 2] for i in range(max(0, len(clean) - 1))]
        for term in terms:
            digest = hashlib.blake2b(term.encode("utf-8"), digest_size=8).digest()
            position = int.from_bytes(digest, "little") % self.dimensions
            vector[position] += 1.0
        norm = np.linalg.norm(vector)
        return vector / norm if norm else vector

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.vstack([self._encode(text) for text in texts]) if texts else np.empty((0, self.dimensions))

    def encode_query(self, text: str) -> np.ndarray:
        return self._encode(text)


class BGEEmbedding:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer

        self.name = model_name
        self.model = SentenceTransformer(model_name, device=device)

    def encode_documents(self, texts: list[str]) -> np.ndarray:
        return np.asarray(self.model.encode(texts, normalize_embeddings=True), dtype=np.float32)

    def encode_query(self, text: str) -> np.ndarray:
        return np.asarray(self.model.encode(QUERY_INSTRUCTION + text, normalize_embeddings=True), dtype=np.float32)


class OverlapReranker:
    name = "token-overlap-v1"

    def score(self, query: str, texts: list[str]) -> list[float]:
        query_terms = set(tokenize(query))
        return [len(query_terms & set(tokenize(text))) / max(1, len(query_terms)) for text in texts]


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import CrossEncoder

        self.name = model_name
        self.model = CrossEncoder(model_name, device=device)

    def score(self, query: str, texts: list[str]) -> list[float]:
        return [float(item) for item in self.model.predict([(query, text) for text in texts])]


class RoleIndex:
    def __init__(self, chunks: list[Chunk], vectors: np.ndarray):
        if vectors.ndim != 2 or len(vectors) != len(chunks) or not np.isfinite(vectors).all():
            raise ValueError("索引向量和文档块不一致")
        self.chunks = chunks
        self.vectors = vectors.astype(np.float32)
        self.tokens = [tokenize(chunk.text) for chunk in chunks]
        self.doc_frequency = Counter(term for terms in self.tokens for term in set(terms))
        self.avg_length = sum(map(len, self.tokens)) / max(1, len(self.tokens))
        self.faiss_index = None
        try:
            import faiss
            if len(vectors):
                self.faiss_index = faiss.IndexFlatIP(vectors.shape[1])
                self.faiss_index.add(self.vectors)
        except ImportError:
            pass

    def dense(self, query: np.ndarray, top_k: int) -> list[int]:
        if not self.chunks:
            return []
        count = min(top_k, len(self.chunks))
        if self.faiss_index is not None:
            _, indices = self.faiss_index.search(query.reshape(1, -1).astype(np.float32), count)
            return [int(index) for index in indices[0] if index >= 0]
        scores = self.vectors @ query
        return [int(index) for index in np.argsort(-scores)[:count]]

    def lexical(self, query: str, top_k: int) -> list[int]:
        query_terms = tokenize(query)
        total_docs, k1, b = max(1, len(self.tokens)), 1.5, 0.75
        scores = []
        for index, terms in enumerate(self.tokens):
            frequencies = Counter(terms)
            score = 0.0
            for term in query_terms:
                frequency = frequencies[term]
                document_frequency = self.doc_frequency[term]
                idf = math.log(1 + (total_docs - document_frequency + 0.5) / (document_frequency + 0.5))
                denominator = frequency + k1 * (1 - b + b * len(terms) / max(1, self.avg_length))
                score += idf * frequency * (k1 + 1) / max(1e-9, denominator)
            scores.append(score)
        return [int(index) for index in np.argsort(-np.asarray(scores))[: min(top_k, len(scores))] if scores[int(index)] > 0]


class IndexManager:
    def __init__(self, settings: Settings, database: Database,
                 embedding: EmbeddingProvider | None = None, reranker: Reranker | None = None):
        self.settings = settings
        self.database = database
        self.degraded: list[str] = []
        self.embedding = embedding or self._embedding()
        self.reranker = reranker or self._reranker()
        self.indexes: dict[Role, RoleIndex] = {}
        self.version: str | None = None
        self._lock = threading.RLock()

    def _embedding(self) -> EmbeddingProvider:
        if not self.settings.fake_providers:
            try:
                device = "cuda" if os.getenv("SELL_RAG_DEVICE") == "cuda" else "cpu"
                return BGEEmbedding(self.settings.embedding_model, device)
            except (ImportError, OSError, RuntimeError) as exc:
                self.degraded.append(f"BGE不可用，使用哈希Embedding: {exc}")
        else:
            self.degraded.append("Fake模式使用哈希Embedding")
        return HashEmbedding()

    def _reranker(self) -> Reranker:
        if not self.settings.fake_providers:
            try:
                device = "cuda" if os.getenv("SELL_RAG_DEVICE") == "cuda" else "cpu"
                return CrossEncoderReranker(self.settings.reranker_model, device)
            except (ImportError, OSError, RuntimeError) as exc:
                self.degraded.append(f"Cross-Encoder不可用，使用词项重排: {exc}")
        else:
            self.degraded.append("Fake模式使用词项重排")
        return OverlapReranker()

    def build(self) -> dict[str, object]:
        with self._lock:
            return self._build()

    def _build(self) -> dict[str, object]:
        version = time.strftime("%Y%m%d_%H%M%S") + "_" + hashlib.sha256(os.urandom(8)).hexdigest()[:8]
        staging = self.settings.indexes_dir / f".{version}.staging"
        target = self.settings.indexes_dir / version
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True)
        indexes: dict[Role, RoleIndex] = {}
        manifest: dict[str, object] = {
            "version": version, "embedding_model": self.embedding.name,
            "reranker_model": self.reranker.name, "roles": {}, "documents": [],
            "degraded": self.degraded,
        }
        for role in Role:
            chunks = [item for item in self.database.active_chunks(role) if not item.prompt_injection]
            vectors = self._cached_vectors(chunks)
            indexes[role] = RoleIndex(chunks, vectors)
            (staging / f"{role.value}.json").write_text(
                json.dumps([item.model_dump(mode="json") for item in chunks], ensure_ascii=False), encoding="utf-8"
            )
            np.save(staging / f"{role.value}.npy", vectors)
            manifest["roles"][role.value] = len(chunks)
            if role == Role.ADMIN:
                manifest["documents"] = sorted({
                    f"{item.document_id}:v{item.document_version}" for item in chunks
                })
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        staging.replace(target)
        active_tmp = self.settings.indexes_dir / f".{version}.active.tmp"
        active_tmp.write_text(version, encoding="utf-8")
        active_tmp.replace(self.settings.indexes_dir / "ACTIVE")
        self.database.save_index_version(version, manifest, True)
        self.indexes, self.version = indexes, version
        return manifest

    def _cached_vectors(self, chunks: list[Chunk]) -> np.ndarray:
        if not chunks:
            dimensions = getattr(self.embedding, "dimensions", 384)
            return np.empty((0, dimensions), dtype=np.float32)
        vectors: list[np.ndarray | None] = []
        missing_indices, missing_texts = [], []
        for index, chunk in enumerate(chunks):
            cached = self.database.get_cached_embedding(chunk.content_hash, self.embedding.name)
            vectors.append(np.asarray(cached, dtype=np.float32) if cached is not None else None)
            if cached is None:
                missing_indices.append(index)
                missing_texts.append(chunk.text)
        if missing_texts:
            encoded = self.embedding.encode_documents(missing_texts)
            for output_index, vector in zip(missing_indices, encoded):
                vectors[output_index] = vector
                self.database.save_cached_embedding(
                    chunks[output_index].content_hash, self.embedding.name, vector.tolist()
                )
        return np.vstack(vectors).astype(np.float32)

    def ensure_loaded(self) -> None:
        with self._lock:
            self._ensure_loaded()

    def _ensure_loaded(self) -> None:
        # SQLite is the publication point shared with CLI and other API processes.
        active = self.database.active_index()
        if not active:
            self.build()
            return
        version = active["index_version"]
        if self.indexes and self.version == version:
            return
        if not re.fullmatch(r"\d{8}_\d{6}_[a-f0-9]{8}", version):
            self.build()
            return
        directory = self.settings.indexes_dir / version
        indexes: dict[Role, RoleIndex] = {}
        try:
            manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
            if manifest.get("embedding_model") != self.embedding.name:
                raise ValueError("索引的 Embedding 模型已改变")
            for role in Role:
                chunks = [Chunk.model_validate(item) for item in json.loads((directory / f"{role.value}.json").read_text(encoding="utf-8"))]
                vectors = np.load(directory / f"{role.value}.npy", allow_pickle=False)
                indexes[role] = RoleIndex(chunks, vectors)
        except (OSError, ValueError, json.JSONDecodeError):
            self.build()
            return
        self.indexes, self.version = indexes, version

    def search(self, query: str, role: Role, top_k: int | None = None) -> list[SearchHit]:
        with self._lock:
            self._ensure_loaded()
            index = self.indexes[role]
        # Revocation/deletion takes effect even while a replacement index is being built.
        permitted = {chunk.chunk_id for chunk in self.database.active_chunks(role)
                     if not chunk.prompt_injection}
        dense = index.dense(self.embedding.encode_query(query), self.settings.dense_top_k)
        lexical = index.lexical(query, self.settings.lexical_top_k)
        scores: dict[int, float] = {}
        ranks: dict[int, tuple[int | None, int | None]] = {}
        for position, item_index in enumerate(dense, 1):
            scores[item_index] = scores.get(item_index, 0) + 1 / (self.settings.rrf_k + position)
            ranks[item_index] = (position, None)
        for position, item_index in enumerate(lexical, 1):
            scores[item_index] = scores.get(item_index, 0) + 1 / (self.settings.rrf_k + position)
            dense_rank, _ = ranks.get(item_index, (None, None))
            ranks[item_index] = (dense_rank, position)
        candidates = [i for i in sorted(scores, key=scores.get, reverse=True)
                      if index.chunks[i].chunk_id in permitted][:30]
        rerank_scores = self.reranker.score(query, [index.chunks[i].text for i in candidates]) if candidates else []
        ordered = sorted(zip(candidates, rerank_scores), key=lambda item: (item[1], scores[item[0]]), reverse=True)
        result = []
        limit = self.settings.final_top_k if top_k is None else max(0, top_k)
        for item_index, rerank_score in ordered:
            # The deterministic fallback has no semantic evidence when overlap is zero.
            if isinstance(self.reranker, OverlapReranker) and rerank_score <= 0:
                continue
            if len(result) >= limit:
                break
            dense_rank, lexical_rank = ranks[item_index]
            result.append(SearchHit(
                chunk=index.chunks[item_index], score=scores[item_index], dense_rank=dense_rank,
                lexical_rank=lexical_rank, rerank_score=rerank_score,
            ))
        return result
