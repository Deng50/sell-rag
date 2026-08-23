from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal

from sell_rag.domain import Citation, Product, QueryRoute, Role, SearchHit
from sell_rag.indexing import IndexManager
from sell_rag.settings import Settings
from sell_rag.storage import Database


@dataclass
class QueryAnalysis:
    route: QueryRoute
    product_text: str | None = None
    product_names: list[str] = field(default_factory=list)
    category: str | None = None
    min_price: Decimal | None = None
    max_price: Decimal | None = None
    in_stock: bool = False


class QueryAnalyzer:
    GREETINGS = {"你好", "您好", "嗨", "hello", "谢谢", "再见"}
    RECOMMEND = ("推荐", "适合", "想买", "吃什么", "喝什么", "来点", "最便宜")
    EXACT = ("价格", "多少钱", "库存", "还有吗", "编号", "sku", "比较", "差价", "哪个便宜")

    def __init__(self, database: Database):
        self.database = database

    def analyze(self, query: str, role: Role = Role.GUEST) -> QueryAnalysis:
        clean = query.strip()
        if clean.lower() in self.GREETINGS or len(clean) <= 2:
            return QueryAnalysis(QueryRoute.CHITCHAT)
        if clean in {"推荐", "商品", "价格"}:
            return QueryAnalysis(QueryRoute.CLARIFY)
        products = self.database.query_products(limit=500, role=role)
        matched_products = [item for item in products if item.name in clean or item.sku.lower() in clean.lower()]
        price_range = re.search(r"(\d+(?:\.\d+)?)\s*(?:-|到|至)\s*(\d+(?:\.\d+)?)\s*元", clean)
        max_price = re.search(r"(\d+(?:\.\d+)?)\s*元\s*(?:以下|以内|不超过)", clean)
        min_price = re.search(r"(?:超过|高于|至少)\s*(\d+(?:\.\d+)?)\s*元", clean)
        category = next((item.category for item in products if item.category and item.category in clean), None)
        filters = dict(
            product_text=matched_products[0].name if len(matched_products) == 1 else None,
            product_names=[item.name for item in matched_products],
            category=category,
            min_price=Decimal(price_range.group(1)) if price_range else (Decimal(min_price.group(1)) if min_price else None),
            max_price=Decimal(price_range.group(2)) if price_range else (Decimal(max_price.group(1)) if max_price else None),
            in_stock=any(word in clean for word in ("有货", "库存", "还有")),
        )
        if matched_products or any(word in clean.lower() for word in self.EXACT):
            return QueryAnalysis(QueryRoute.PRODUCT_EXACT, **filters)
        if any(word in clean for word in self.RECOMMEND) or category or price_range or max_price or min_price:
            return QueryAnalysis(QueryRoute.PRODUCT_RECOMMENDATION, **filters)
        return QueryAnalysis(QueryRoute.KNOWLEDGE, **filters)


class RetrievalService:
    def __init__(self, settings: Settings, database: Database, index_manager: IndexManager):
        self.settings = settings
        self.database = database
        self.index_manager = index_manager

    def knowledge(self, query: str, role: Role) -> tuple[list[SearchHit], list[Citation], str]:
        hits = self.index_manager.search(query, role)
        evidence_parts, citations, seen = [], [], set()
        budget = self.settings.max_context_tokens * 2
        used = 0
        for hit in hits:
            source = self.database.get_chunk(hit.chunk.parent_id, role) if hit.chunk.parent_id else hit.chunk
            source = source or hit.chunk
            if source.chunk_id in seen:
                continue
            seen.add(source.chunk_id)
            excerpt = source.text.strip()
            if used + len(excerpt) > budget:
                excerpt = excerpt[: max(0, budget - used)]
            if not excerpt:
                continue
            citation_id = f"S{len(citations) + 1}"
            citations.append(Citation(
                citation_id=citation_id, document_id=source.document_id,
                document_version=source.document_version,
                title=source.title or (source.section_path[-1] if source.section_path else "未命名文档"),
                excerpt=excerpt[:300], page_no=source.page_no, sheet_name=source.sheet_name,
                cell_range=source.cell_range,
            ))
            evidence_parts.append(f"[{citation_id}] {excerpt}")
            used += len(excerpt)
            if used >= budget:
                break
        return hits, citations, "\n\n".join(evidence_parts)

    def products(self, analysis: QueryAnalysis, role: Role) -> list[Product]:
        if analysis.product_names:
            selected = []
            for name in analysis.product_names:
                selected.extend(self.database.query_products(text=name, in_stock=False, limit=1, role=role))
            return list({item.sku: item for item in selected}.values())
        return self.database.query_products(
            text=analysis.product_text, category=analysis.category,
            min_price=analysis.min_price, max_price=analysis.max_price,
            in_stock=analysis.in_stock or analysis.route == QueryRoute.PRODUCT_RECOMMENDATION,
            limit=8, role=role,
        )

    @staticmethod
    def product_evidence(products: list[Product]) -> tuple[list[Citation], str]:
        citations, parts = [], []
        for index, product in enumerate(products, 1):
            citation_id = f"S{index}"
            text = (f"SKU：{product.sku}；商品：{product.name}；类别：{product.category}；"
                    f"价格：{product.price}元；库存：{product.stock}；特点：{product.description}")
            citations.append(Citation(
                citation_id=citation_id, document_id=product.source_document_id or "product-db",
                document_version=product.source_document_version or 1,
                title=f"商品库/{product.name}", excerpt=text,
            ))
            parts.append(f"[{citation_id}] {text}")
        return citations, "\n".join(parts)
