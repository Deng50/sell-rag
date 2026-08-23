from __future__ import annotations

import json
import math
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
    expected = [str(cell).strip() for row in reference for cell in row]
    actual = [str(cell).strip() for row in prediction for cell in row]
    width = max(len(expected), len(actual), 1)
    matches = sum(i < len(expected) and i < len(actual) and expected[i] == actual[i] for i in range(width))
    return matches / width


class EvaluationRunner:
    def __init__(self, answer_service: AnswerService, index_manager: IndexManager, database: Database):
        self.answer_service = answer_service
        self.index_manager = index_manager
        self.database = database

    def run(self, dataset: str | Path) -> dict[str, Any]:
        cases = [json.loads(line) for line in Path(dataset).read_text(encoding="utf-8").splitlines() if line.strip()]
        latencies, reciprocal_ranks, recalls3, recalls10, ndcg10 = [], [], [], [], []
        citation_ok, refusal_ok, facts_ok = [], [], []
        for case in cases:
            query = case["query"]
            started = time.perf_counter()
            hits = self.index_manager.search(query, Role.GUEST, top_k=10)
            answer = self.answer_service.answer(QueryRequest(query=query))
            latencies.append((time.perf_counter() - started) * 1000)
            relevant = set(case.get("relevant_document_ids", []))
            ranked = [item.chunk.document_id for item in hits]
            ranks = [i + 1 for i, value in enumerate(ranked) if value in relevant]
            reciprocal_ranks.append(1 / min(ranks) if ranks else 0)
            recalls3.append(bool(relevant & set(ranked[:3])) if relevant else True)
            recalls10.append(bool(relevant & set(ranked[:10])) if relevant else True)
            dcg = sum(1 / math.log2(rank + 1) for rank in ranks if rank <= 10)
            ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(relevant), 10) + 1))
            ndcg10.append(dcg / ideal if ideal else 1.0)
            citation_ok.append(all(citation.citation_id in answer.answer for citation in answer.citations))
            refusal_ok.append(answer.refused == bool(case.get("should_refuse", False)))
            facts_ok.append(all(str(fact) in answer.answer for fact in case.get("expected_facts", [])))
        metrics = {
            "case_count": len(cases), "recall_at_3": statistics.fmean(recalls3) if cases else 0,
            "recall_at_10": statistics.fmean(recalls10) if cases else 0,
            "mrr_at_10": statistics.fmean(reciprocal_ranks) if cases else 0,
            "ndcg_at_10": statistics.fmean(ndcg10) if cases else 0,
            "citation_accuracy": statistics.fmean(citation_ok) if cases else 0,
            "refusal_accuracy": statistics.fmean(refusal_ok) if cases else 0,
            "expected_fact_accuracy": statistics.fmean(facts_ok) if cases else 0,
            "latency_p50_ms": statistics.median(latencies) if latencies else 0,
            "latency_p95_ms": sorted(latencies)[max(0, math.ceil(len(latencies) * 0.95) - 1)] if latencies else 0,
            "note": "内置基准仅用于回归，不代表真实业务准确率。",
        }
        metrics["run_id"] = self.database.save_evaluation("complete", metrics)
        return metrics
