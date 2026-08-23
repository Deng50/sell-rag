from __future__ import annotations

import re
import uuid

from sell_rag.domain import AnswerResponse, QueryRequest, QueryRoute
from sell_rag.generation.providers import Generator
from sell_rag.indexing import IndexManager
from sell_rag.retrieval import QueryAnalyzer, RetrievalService
from sell_rag.settings import Settings
from sell_rag.storage import Database


class AnswerService:
    def __init__(self, settings: Settings, database: Database,
                 index_manager: IndexManager, generator: Generator):
        self.settings = settings
        self.database = database
        self.index_manager = index_manager
        self.generator = generator
        self.analyzer = QueryAnalyzer(database)
        self.retrieval = RetrievalService(settings, database, index_manager)

    def answer(self, request: QueryRequest) -> AnswerResponse:
        request_id = uuid.uuid4().hex
        analysis = self.analyzer.analyze(request.query, request.role)
        if analysis.route == QueryRoute.CLARIFY:
            return AnswerResponse(
                answer="请补充商品名称、类别、预算或希望了解的资料主题。",
                route=analysis.route, refused=True, refusal_reason="query_requires_clarification",
                degraded=self.index_manager.degraded, request_id=request_id,
            )
        if analysis.route == QueryRoute.CHITCHAT:
            text = self.generator.generate(request.query, "", [], analysis.route)
            return AnswerResponse(answer=text, route=analysis.route, grounded=False,
                                  degraded=self.index_manager.degraded, request_id=request_id)
        if analysis.route in {QueryRoute.PRODUCT_EXACT, QueryRoute.PRODUCT_RECOMMENDATION}:
            if (analysis.route == QueryRoute.PRODUCT_EXACT and not analysis.product_names
                    and not analysis.category and analysis.min_price is None and analysis.max_price is None):
                return self._refuse(analysis.route, request_id, "未识别到知识库中的具体商品，请补充商品名称或编号。")
            products = self.retrieval.products(analysis, request.role)
            if not products:
                return self._refuse(analysis.route, request_id, "未找到满足名称、价格或库存条件的商品。")
            citations, evidence = self.retrieval.product_evidence(products)
            if analysis.route == QueryRoute.PRODUCT_EXACT:
                lines = [f"{item.name}（SKU {item.sku}）：{item.price} 元，库存 {item.stock}。 [S{i}]"
                         for i, item in enumerate(products, 1)]
                if len(products) >= 2 and any(word in request.query for word in ("比较", "差价", "便宜")):
                    cheapest = min(products, key=lambda item: item.price)
                    difference = max(item.price for item in products) - cheapest.price
                    lines.append(f"其中 {cheapest.name} 价格最低，最大差价为 {difference:.2f} 元。 "
                                 f"[S{products.index(cheapest) + 1}]")
                answer = "\n".join(lines)
            else:
                answer = self.generator.generate(request.query, evidence,
                                                 [item.citation_id for item in citations], analysis.route)
            return self._validated(answer, analysis.route, citations, products, request_id)
        _, citations, evidence = self.retrieval.knowledge(request.query, request.role)
        if not citations:
            return self._refuse(analysis.route, request_id, "知识库中没有足够证据回答该问题。")
        answer = self.generator.generate(request.query, evidence,
                                         [item.citation_id for item in citations], analysis.route)
        return self._validated(answer, analysis.route, citations, [], request_id)

    def _validated(self, answer: str, route: QueryRoute, citations, products, request_id: str) -> AnswerResponse:
        referenced = set(re.findall(r"\[(S\d+)\]", answer))
        allowed = {item.citation_id for item in citations}
        if not referenced or not referenced.issubset(allowed):
            return self._refuse(route, request_id, "模型回答未通过引用校验。")
        selected = [item for item in citations if item.citation_id in referenced]
        return AnswerResponse(
            answer=answer, route=route, citations=selected, products=products,
            grounded=True, degraded=self.index_manager.degraded, request_id=request_id,
        )

    def _refuse(self, route: QueryRoute, request_id: str, reason: str) -> AnswerResponse:
        return AnswerResponse(
            answer=reason, route=route, refused=True, refusal_reason=reason,
            degraded=self.index_manager.degraded, request_id=request_id,
        )
