from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal
from pathlib import Path
from typing import Any, Iterable, Iterator

from sell_rag.domain import Chunk, DocumentElement, DocumentVersion, Product, Role


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


class Database:
    """SQLite source of truth. Every write is transactional and thread-safe per call."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA journal_mode=WAL")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def initialize(self) -> None:
        with self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS documents (
                    document_id TEXT PRIMARY KEY, source_id TEXT NOT NULL UNIQUE,
                    filename TEXT NOT NULL, owner TEXT NOT NULL, deleted_at TEXT
                );
                CREATE TABLE IF NOT EXISTS document_versions (
                    document_id TEXT NOT NULL, version INTEGER NOT NULL, source_type TEXT NOT NULL,
                    sha256 TEXT NOT NULL, acl_json TEXT NOT NULL, parser_version TEXT NOT NULL,
                    status TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0, created_at TEXT NOT NULL,
                    error TEXT, stored_path TEXT, PRIMARY KEY(document_id, version),
                    FOREIGN KEY(document_id) REFERENCES documents(document_id)
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_version_hash
                    ON document_versions(document_id, sha256);
                CREATE TABLE IF NOT EXISTS elements (
                    element_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    document_version INTEGER NOT NULL, element_order INTEGER NOT NULL,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(document_id, document_version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY, document_id TEXT NOT NULL,
                    document_version INTEGER NOT NULL, parent_id TEXT, level TEXT NOT NULL,
                    chunk_index INTEGER NOT NULL, text TEXT NOT NULL, content_hash TEXT NOT NULL,
                    acl_json TEXT NOT NULL, prompt_injection INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    FOREIGN KEY(document_id, document_version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );
                CREATE INDEX IF NOT EXISTS idx_chunks_active
                    ON chunks(document_id, document_version, level);
                CREATE TABLE IF NOT EXISTS products (
                    sku TEXT PRIMARY KEY, name TEXT NOT NULL, category TEXT NOT NULL,
                    price_cents INTEGER NOT NULL, stock INTEGER NOT NULL, active INTEGER NOT NULL,
                    source_document_id TEXT, payload_json TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS product_versions (
                    document_id TEXT NOT NULL, document_version INTEGER NOT NULL,
                    sku TEXT NOT NULL, payload_json TEXT NOT NULL,
                    PRIMARY KEY(document_id, document_version, sku),
                    FOREIGN KEY(document_id, document_version)
                        REFERENCES document_versions(document_id, version) ON DELETE CASCADE
                );
                CREATE TABLE IF NOT EXISTS jobs (
                    job_id TEXT PRIMARY KEY, kind TEXT NOT NULL, status TEXT NOT NULL,
                    progress REAL NOT NULL DEFAULT 0, detail TEXT,
                    created_at TEXT NOT NULL, updated_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS index_versions (
                    index_version TEXT PRIMARY KEY, status TEXT NOT NULL,
                    manifest_json TEXT NOT NULL, active INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS embedding_cache (
                    content_hash TEXT NOT NULL, embedding_version TEXT NOT NULL,
                    vector_json TEXT NOT NULL, PRIMARY KEY(content_hash, embedding_version)
                );
                CREATE TABLE IF NOT EXISTS feedback (
                    feedback_id TEXT PRIMARY KEY, request_id TEXT, rating INTEGER,
                    comment TEXT, created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS evaluation_runs (
                    run_id TEXT PRIMARY KEY, status TEXT NOT NULL,
                    metrics_json TEXT NOT NULL, created_at TEXT NOT NULL
                );
                """
            )
            # Preserve the currently available snapshot when upgrading an existing database.
            db.execute(
                "INSERT OR IGNORE INTO product_versions SELECT p.source_document_id, "
                "v.version,p.sku,p.payload_json FROM products p JOIN document_versions v "
                "ON p.source_document_id=v.document_id "
                "AND json_extract(p.payload_json,'$.source_document_version')=v.version"
            )
            db.execute("DROP INDEX IF EXISTS idx_version_hash")
            db.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_version_content_policy "
                "ON document_versions(document_id,sha256,acl_json,parser_version)"
            )

    def begin_document_version(
        self, *, source_id: str, filename: str, source_type: str, sha256: str,
        owner: str, acl: list[Role], parser_version: str, stored_path: str,
    ) -> tuple[DocumentVersion, bool]:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            acl_json = _json(sorted({x.value for x in acl}))
            row = db.execute("SELECT * FROM documents WHERE source_id=?", (source_id,)).fetchone()
            document_id = row["document_id"] if row else uuid.uuid5(uuid.NAMESPACE_URL, source_id).hex
            if not row:
                db.execute(
                    "INSERT INTO documents(document_id,source_id,filename,owner) VALUES(?,?,?,?)",
                    (document_id, source_id, filename, owner),
                )
            duplicate = db.execute(
                "SELECT version,status,active,created_at,error FROM document_versions "
                "WHERE document_id=? AND sha256=? AND acl_json=? AND parser_version=?",
                (document_id, sha256, acl_json, parser_version),
            ).fetchone()
            if duplicate:
                retry = duplicate["status"] == "failed"
                if retry:
                    db.execute(
                        "UPDATE document_versions SET status='processing',error=NULL "
                        "WHERE document_id=? AND version=?", (document_id, duplicate["version"]),
                    )
                return DocumentVersion(
                    document_id=document_id, source_id=source_id, version=duplicate["version"],
                    filename=filename, source_type=source_type, sha256=sha256, owner=owner,
                    acl=acl, parser_version=parser_version,
                    status="processing" if retry else duplicate["status"],
                    active=bool(duplicate["active"]), created_at=duplicate["created_at"],
                    error=duplicate["error"],
                ), not retry
            version = db.execute(
                "SELECT COALESCE(MAX(version),0)+1 FROM document_versions WHERE document_id=?",
                (document_id,),
            ).fetchone()[0]
            created_at = _now()
            db.execute(
                "INSERT INTO document_versions VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                (document_id, version, source_type, sha256, acl_json,
                 parser_version, "processing", 0, created_at, None, stored_path),
            )
            return DocumentVersion(
                document_id=document_id, source_id=source_id, version=version,
                filename=filename, source_type=source_type, sha256=sha256, owner=owner,
                acl=acl, parser_version=parser_version, status="processing", created_at=created_at,
            ), False

    def complete_document_version(
        self, version: DocumentVersion, elements: Iterable[DocumentElement],
        chunks: Iterable[Chunk], products: Iterable[Product],
    ) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM elements WHERE document_id=? AND document_version=?",
                       (version.document_id, version.version))
            db.execute("DELETE FROM chunks WHERE document_id=? AND document_version=?",
                       (version.document_id, version.version))
            for element in elements:
                db.execute("INSERT INTO elements VALUES(?,?,?,?,?)", (
                    element.element_id, element.document_id, element.document_version,
                    element.order, element.model_dump_json(),
                ))
            for chunk in chunks:
                db.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
                    chunk.chunk_id, chunk.document_id, chunk.document_version, chunk.parent_id,
                    chunk.level, chunk.index, chunk.text, chunk.content_hash,
                    _json([x.value for x in chunk.acl]), int(chunk.prompt_injection),
                    chunk.model_dump_json(),
                ))
            db.execute("DELETE FROM product_versions WHERE document_id=? AND document_version=?",
                       (version.document_id, version.version))
            for product in products:
                db.execute("INSERT INTO product_versions VALUES(?,?,?,?)", (
                    version.document_id, version.version, product.sku, product.model_dump_json(),
                ))
            self._restore_products(db, version.document_id, version.version)
            db.execute("UPDATE document_versions SET active=0 WHERE document_id=?", (version.document_id,))
            db.execute(
                "UPDATE document_versions SET status='ready',active=1,error=NULL "
                "WHERE document_id=? AND version=?", (version.document_id, version.version),
            )
            db.execute("UPDATE documents SET deleted_at=NULL WHERE document_id=?", (version.document_id,))

    @staticmethod
    def _restore_products(db: sqlite3.Connection, document_id: str, version: int) -> None:
        db.execute("DELETE FROM products WHERE source_document_id=?", (document_id,))
        for row in db.execute(
            "SELECT payload_json FROM product_versions WHERE document_id=? AND document_version=?",
            (document_id, version),
        ).fetchall():
            product = Product.model_validate_json(row[0])
            if db.execute("SELECT 1 FROM products WHERE sku=?", (product.sku,)).fetchone():
                raise ValueError(f"SKU {product.sku} 已由其他文档或手工商品占用")
            db.execute("INSERT INTO products VALUES(?,?,?,?,?,?,?,?,?)", (
                product.sku, product.name, product.category, int(product.price * 100),
                product.stock, int(product.active), document_id,
                product.model_dump_json(), product.updated_at.isoformat(),
            ))

    def fail_document_version(self, document_id: str, version: int, error: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE document_versions SET status='failed',error=? WHERE document_id=? AND version=?",
                (error[:2000], document_id, version),
            )

    def list_documents(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT d.*,v.version,v.source_type,v.sha256,v.acl_json,v.status,v.active,"
                "v.created_at,v.error FROM documents d JOIN document_versions v "
                "ON d.document_id=v.document_id ORDER BY d.filename,v.version DESC"
            ).fetchall()
        return [{**dict(row), "acl": json.loads(row["acl_json"])} for row in rows]

    def activate_version(self, document_id: str, version: int) -> None:
        with self.connect() as db:
            target = db.execute(
                "SELECT status FROM document_versions WHERE document_id=? AND version=?",
                (document_id, version),
            ).fetchone()
            if not target:
                raise KeyError((document_id, version))
            if target["status"] != "ready":
                raise ValueError("Only a ready document version can be activated")
            self._restore_products(db, document_id, version)
            db.execute("UPDATE document_versions SET active=0 WHERE document_id=?", (document_id,))
            db.execute("UPDATE document_versions SET active=1 WHERE document_id=? AND version=?",
                       (document_id, version))
            db.execute("UPDATE documents SET deleted_at=NULL WHERE document_id=?", (document_id,))

    def soft_delete_document(self, document_id: str) -> None:
        with self.connect() as db:
            if not db.execute("SELECT 1 FROM documents WHERE document_id=?", (document_id,)).fetchone():
                raise KeyError(document_id)
            db.execute("UPDATE documents SET deleted_at=? WHERE document_id=?", (_now(), document_id))
            db.execute("UPDATE document_versions SET active=0 WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM products WHERE source_document_id=?", (document_id,))

    def active_chunks(self, role: Role, *, children_only: bool = True) -> list[Chunk]:
        query = (
            "SELECT c.payload_json FROM chunks c JOIN document_versions v "
            "ON c.document_id=v.document_id AND c.document_version=v.version "
            "JOIN documents d ON c.document_id=d.document_id "
            "WHERE v.active=1 AND v.status='ready' AND d.deleted_at IS NULL"
        )
        if children_only:
            query += " AND c.level='child'"
        with self.connect() as db:
            rows = db.execute(query).fetchall()
        result = []
        for row in rows:
            chunk = Chunk.model_validate_json(row[0])
            if role in chunk.acl:
                result.append(chunk)
        return result

    def get_chunk(self, chunk_id: str, role: Role) -> Chunk | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT c.payload_json FROM chunks c JOIN document_versions v "
                "ON c.document_id=v.document_id AND c.document_version=v.version "
                "JOIN documents d ON c.document_id=d.document_id "
                "WHERE c.chunk_id=? AND v.active=1 AND v.status='ready' AND d.deleted_at IS NULL",
                (chunk_id,),
            ).fetchone()
        if not row:
            return None
        chunk = Chunk.model_validate_json(row[0])
        return chunk if role in chunk.acl else None

    def prompt_injection_findings(self) -> list[dict[str, Any]]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT c.chunk_id,c.document_id,c.document_version,c.text,d.filename "
                "FROM chunks c JOIN documents d ON c.document_id=d.document_id "
                "WHERE c.prompt_injection=1 ORDER BY c.document_id,c.chunk_index"
            ).fetchall()
        return [{**dict(row), "text": row["text"][:500]} for row in rows]

    def query_products(
        self, *, text: str | None = None, category: str | None = None,
        min_price: Decimal | None = None, max_price: Decimal | None = None,
        in_stock: bool = False, limit: int = 20, role: Role = Role.GUEST,
        sku: str | None = None,
    ) -> list[Product]:
        clauses = ["active=1", "EXISTS (SELECT 1 FROM json_each(payload_json,'$.acl') WHERE value=?)"]
        values: list[Any] = [role.value]
        if sku is not None:
            clauses.append("sku=?")
            values.append(sku)
        if text:
            clauses.append("(name LIKE ? OR sku LIKE ? OR payload_json LIKE ?)")
            values.extend([f"%{text}%"] * 3)
        if category:
            clauses.append("category=?")
            values.append(category)
        if min_price is not None:
            clauses.append("price_cents>=?")
            values.append(int(min_price * 100))
        if max_price is not None:
            clauses.append("price_cents<=?")
            values.append(int(max_price * 100))
        if in_stock:
            clauses.append("stock>0")
        values.append(limit)
        with self.connect() as db:
            rows = db.execute(
                f"SELECT payload_json FROM products WHERE {' AND '.join(clauses)} "
                "ORDER BY stock>0 DESC,price_cents ASC LIMIT ?", values,
            ).fetchall()
        products = [Product.model_validate_json(row[0]) for row in rows]
        return [product for product in products if role in product.acl]

    def upsert_product(self, product: Product) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO products VALUES(?,?,?,?,?,?,?,?,?) ON CONFLICT(sku) DO UPDATE SET "
                "name=excluded.name,category=excluded.category,price_cents=excluded.price_cents,"
                "stock=excluded.stock,active=excluded.active,payload_json=excluded.payload_json,"
                "source_document_id=excluded.source_document_id,"
                "updated_at=excluded.updated_at",
                (product.sku, product.name, product.category, int(product.price * 100), product.stock,
                 int(product.active), product.source_document_id, product.model_dump_json(),
                 product.updated_at.isoformat()),
            )

    def create_job(self, kind: str, detail: str = "") -> str:
        job_id, now = uuid.uuid4().hex, _now()
        with self.connect() as db:
            db.execute("INSERT INTO jobs VALUES(?,?,?,?,?,?,?)",
                       (job_id, kind, "pending", 0, detail, now, now))
        return job_id

    def update_job(self, job_id: str, status: str, progress: float, detail: str = "") -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET status=?,progress=?,detail=?,updated_at=? WHERE job_id=?",
                       (status, progress, detail, _now(), job_id))

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE job_id=?", (job_id,)).fetchone()
        return dict(row) if row else None

    def save_index_version(self, version: str, manifest: dict[str, Any], active: bool) -> None:
        with self.connect() as db:
            if active:
                db.execute("UPDATE index_versions SET active=0")
            db.execute("INSERT OR REPLACE INTO index_versions VALUES(?,?,?,?,?)",
                       (version, "ready", _json(manifest), int(active), _now()))

    def active_index(self) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM index_versions WHERE active=1").fetchone()
        return dict(row) if row else None

    def get_cached_embedding(self, content_hash: str, embedding_version: str) -> list[float] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT vector_json FROM embedding_cache WHERE content_hash=? AND embedding_version=?",
                (content_hash, embedding_version),
            ).fetchone()
        return json.loads(row[0]) if row else None

    def save_cached_embedding(self, content_hash: str, embedding_version: str,
                              vector: list[float]) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO embedding_cache VALUES(?,?,?)",
                (content_hash, embedding_version, _json(vector)),
            )

    def add_feedback(self, request_id: str | None, rating: int, comment: str) -> str:
        feedback_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO feedback VALUES(?,?,?,?,?)",
                       (feedback_id, request_id, rating, comment, _now()))
        return feedback_id

    def save_evaluation(self, status: str, metrics: dict[str, Any]) -> str:
        run_id = uuid.uuid4().hex
        with self.connect() as db:
            db.execute("INSERT INTO evaluation_runs VALUES(?,?,?,?)",
                       (run_id, status, _json(metrics), _now()))
        return run_id
