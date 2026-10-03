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


def test_fake_synthesis_is_a_valid_wave_file() -> None:
    import io
    import wave
    from sell_rag.multimodal import FakeSpeech

    with wave.open(io.BytesIO(FakeSpeech().synthesize("测试"))) as audio:
        assert audio.getframerate() == 16000
        assert audio.getnframes() > 0


def test_barge_in_does_not_play_audio_from_cancelled_synthesis() -> None:
    import threading
    from sell_rag.multimodal import StreamingTTSPlayer

    started, release = threading.Event(), threading.Event()
    played = []

    class Provider:
        def synthesize(self, text):
            if text == "旧回答。":
                started.set()
                assert release.wait(2)
            return text.encode()

    player = StreamingTTSPlayer(Provider(), played.append)
    player.speak("旧回答。")
    assert started.wait(1)
    old_threads = player._threads
    player.speak("新回答。")
    release.set()
    assert player.wait(2)
    for thread in old_threads:
        thread.join(2)
        assert not thread.is_alive()
    assert played == ["新回答。".encode()]


def test_synthesis_failure_terminates_consumer() -> None:
    from sell_rag.multimodal import StreamingTTSPlayer

    class Provider:
        def synthesize(self, _):
            raise RuntimeError("offline")

    player = StreamingTTSPlayer(Provider(), lambda _: None)
    player.speak("测试。")
    assert player.wait(2)
    assert isinstance(player.last_error, RuntimeError)
