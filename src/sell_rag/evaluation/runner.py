from __future__ import annotations

import json
import math
import re
import statistics
import time
from pathlib import Path
from typing import Any

from sell_rag.domain import QueryRequest, Role
from sell_rag.generation.service import AnswerService
from sell_rag.indexing import IndexManager
from sell_rag.storage import Database


def character_error_rate(reference: str, prediction: str) -> float:
    previous = list(range(len(prediction) + 1))
    for i, left in enumerate(reference, 1):
        current = [i]
        for j, right in enumerate(prediction, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1] / max(1, len(reference))


def table_cell_accuracy(reference: list[list[Any]], prediction: list[list[Any]]) -> float:
    expected = {(r, c): str(cell).strip() for r, row in enumerate(reference) for c, cell in enumerate(row)}
    actual = {(r, c): str(cell).strip() for r, row in enumerate(prediction) for c, cell in enumerate(row)}
    positions = expected.keys() | actual.keys()
    matches = sum(key in expected and key in actual and expected[key] == actual[key] for key in positions)
    return matches / max(1, len(positions))


class EvaluationRunner:
    def __init__(self, answer_service: AnswerService, index_manager: IndexManager, database: Database):
        self.answer_service = answer_service
        self.index_manager = index_manager
        self.database = database

    def run(self, dataset: str | Path) -> dict[str, Any]:
        cases = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        latencies, reciprocal_ranks, recalls3, recalls10, ndcg10 = [], [], [], [], []
        citation_ok, refusal_ok, facts_ok = [], [], []
        for case in cases:
            if not isinstance(case, dict) or not isinstance(case.get("query"), str):
                raise ValueError("每个评测用例必须是包含 query 字符串的 JSON 对象")
            query = case["query"]
            started = time.perf_counter()
            hits = self.index_manager.search(query, Role.GUEST, top_k=10)
            answer = self.answer_service.answer(QueryRequest(query=query))
            latencies.append((time.perf_counter() - started) * 1000)
            relevant = set(case.get("relevant_document_ids", []))
            ranked = list(dict.fromkeys(item.chunk.document_id for item in hits))
            if relevant:
                ranks = [i + 1 for i, value in enumerate(ranked) if value in relevant]
                reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
                recalls3.append(len(relevant & set(ranked[:3])) / len(relevant))
                recalls10.append(len(relevant & set(ranked[:10])) / len(relevant))
                dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
                ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
                ndcg10.append(dcg / ideal)
            if not answer.refused and answer.route.value not in {"chitchat", "clarify"}:
                referenced = set(re.findall(r"\[(S\d+)\]", answer.answer))
                cited = {citation.citation_id for citation in answer.citations}
                citation_ok.append(bool(cited) and cited == referenced)
            refusal_ok.append(answer.refused == bool(case.get("should_refuse", False)))
            if case.get("expected_facts"):
                facts_ok.append(all(str(fact) in answer.answer for fact in case["expected_facts"]))
        metrics = {
            "case_count": len(cases), "retrieval_case_count": len(recalls3),
            "recall_at_3": statistics.fmean(recalls3) if recalls3 else None,
            "recall_at_10": statistics.fmean(recalls10) if recalls10 else None,
            "mrr_at_10": statistics.fmean(reciprocal_ranks) if reciprocal_ranks else None,
            "ndcg_at_10": statistics.fmean(ndcg10) if ndcg10 else None,
            "citation_case_count": len(citation_ok),
            "citation_accuracy": statistics.fmean(citation_ok) if citation_ok else None,
            "refusal_accuracy": statistics.fmean(refusal_ok) if cases else 0,
            "expected_fact_accuracy": statistics.fmean(facts_ok) if facts_ok else None,
            "latency_p50_ms": statistics.median(latencies) if latencies else 0,
            "latency_p95_ms": sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0,
            "note": "内置基准仅用于回归，不代表真实业务准确率。",
        }
        metrics["run_id"] = self.database.save_evaluation("complete", metrics)
        return metrics
