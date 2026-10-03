from __future__ import annotations

from datetime import datetime, timezone
from decimal import Decimal
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Role(str, Enum):
    GUEST = "guest"
    OPERATOR = "operator"
    ADMIN = "admin"


class QueryRoute(str, Enum):
    PRODUCT_EXACT = "product_exact"
    PRODUCT_RECOMMENDATION = "product_recommendation"
    KNOWLEDGE = "knowledge"
    CHITCHAT = "chitchat"
    CLARIFY = "clarify"


class DocumentVersion(BaseModel):
    document_id: str
    source_id: str
    version: int
    filename: str
    source_type: str
    sha256: str
    owner: str = "admin"
    acl: list[Role] = Field(default_factory=lambda: [Role.GUEST, Role.OPERATOR, Role.ADMIN])
    parser_version: str = "sell-rag-v1"
    status: Literal["pending", "processing", "ready", "failed", "deleted"] = "pending"
    active: bool = False
    created_at: datetime = Field(default_factory=utc_now)
    error: str | None = None


class DocumentElement(BaseModel):
    element_id: str
    document_id: str
    document_version: int
    order: int
    content_type: Literal["title", "text", "table", "image", "product"]
    text: str = ""
    title: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_no: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    bbox: list[float] | None = None
    table_json: list[list[Any]] | None = None
    image_ref: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Chunk(BaseModel):
    chunk_id: str
    document_id: str
    document_version: int
    parent_id: str | None = None
    level: Literal["parent", "child"] = "child"
    index: int
    text: str
    title: str = ""
    section_path: list[str] = Field(default_factory=list)
    page_no: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None
    content_type: str = "text"
    content_hash: str
    acl: list[Role] = Field(default_factory=lambda: [Role.GUEST, Role.OPERATOR, Role.ADMIN])
    prompt_injection: bool = False
    metadata: dict[str, Any] = Field(default_factory=dict)


class Product(BaseModel):
    sku: str = Field(min_length=1)
    name: str = Field(min_length=1)
    category: str = "未分类"
    price: Decimal = Decimal("0.00")
    stock: int = 0
    target_group: list[str] = Field(default_factory=list)
    description: str = ""
    active: bool = True
    source_document_id: str | None = None
    source_document_version: int | None = None
    acl: list[Role] = Field(default_factory=lambda: [Role.GUEST, Role.OPERATOR, Role.ADMIN])
    updated_at: datetime = Field(default_factory=utc_now)
    attributes: dict[str, Any] = Field(default_factory=dict)

    @field_validator("sku", "name", mode="before")
    @classmethod
    def strip_identifier(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator("price")
    @classmethod
    def non_negative_price(cls, value: Decimal) -> Decimal:
        if not value.is_finite() or not Decimal("0") <= value <= Decimal("92233720368547758.07"):
            raise ValueError("price must be finite, non-negative and fit in SQLite cents")
        return value.quantize(Decimal("0.01"))

    @field_validator("stock")
    @classmethod
    def non_negative_stock(cls, value: int) -> int:
        if value < 0:
            raise ValueError("stock must be non-negative")
        return value


class QueryRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    role: Role = Role.GUEST
    session_id: str | None = None

    @field_validator("query", mode="before")
    @classmethod
    def strip_query(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value


class SearchHit(BaseModel):
    chunk: Chunk
    score: float
    dense_rank: int | None = None
    lexical_rank: int | None = None
    rerank_score: float | None = None


class Citation(BaseModel):
    citation_id: str
    document_id: str
    document_version: int
    title: str
    excerpt: str
    page_no: int | None = None
    sheet_name: str | None = None
    cell_range: str | None = None


class AnswerResponse(BaseModel):
    answer: str
    route: QueryRoute
    citations: list[Citation] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    grounded: bool = False
    refused: bool = False
    refusal_reason: str | None = None
    degraded: list[str] = Field(default_factory=list)
    request_id: str | None = None


class Transcript(BaseModel):
    text: str
    confidence: float | None = None
    needs_confirmation: bool = False
    alternatives: list[str] = Field(default_factory=list)
