from __future__ import annotations

from sell_rag.domain import Transcript
from sell_rag.evaluation import character_error_rate, table_cell_accuracy
from sell_rag.multimodal import PresenceGate, SpeechNormalizer


def test_presence_gate_debounce_and_cooldown() -> None:
    gate = PresenceGate(window=4, required_hits=3, cooldown_seconds=10)
    assert [gate.observe(value, now=i) for i, value in enumerate([True, False, True])] == [False] * 3
    assert gate.observe(True, now=3) is True
    assert all(not gate.observe(True, now=4 + i) for i in range(4))
    assert gate.observe(True, now=20) is True


def test_low_confidence_and_hotword_normalization() -> None:
    result = SpeechNormalizer(["每日坚果"]).normalize(
        Transcript(text="每日坚裹。", confidence=0.5)
    )
    assert result.needs_confirmation is True


def test_evaluation_helpers() -> None:
    assert character_error_rate("天津大学", "天津大雪") == 0.25
    assert table_cell_accuracy([["a", "b"]], [["a", "x"]]) == 0.5


def test_regression_evaluation_runs(services) -> None:
    services.ingestion.ingest("tests/fixtures/products.csv")
    services.ingestion.ingest("tests/fixtures/knowledge.md")
    services.index.build()
    metrics = services.evaluation.run("tests/fixtures/evaluation.jsonl")
    assert metrics["case_count"] == 3
    assert {"recall_at_3", "mrr_at_10", "ndcg_at_10", "citation_accuracy"} <= metrics.keys()
