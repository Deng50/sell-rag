from __future__ import annotations

import pytest

pytest.importorskip("pytestqt")
pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QTabWidget

from sell_rag.ui.main_window import MainWindow


@pytest.mark.ui
def test_main_window_has_chat_and_admin_views(qtbot, monkeypatch) -> None:
    monkeypatch.setattr(MainWindow, "_request", lambda *args, **kwargs: None)
    window = MainWindow()
    qtbot.addWidget(window)
    tabs = window.findChild(QTabWidget)
    assert tabs.count() == 2
    assert [tabs.tabText(index) for index in range(tabs.count())] == ["智能问答", "知识库管理"]


@pytest.mark.ui
def test_answers_escape_html_and_keep_overlapping_workers_alive(qtbot, monkeypatch) -> None:
    import threading
    from sell_rag.ui.main_window import AudioWorker

    release = threading.Event()
    monkeypatch.setattr(MainWindow, "_request", lambda *args, **kwargs: None)
    def pending(self):
        release.wait(2)
    monkeypatch.setattr(AudioWorker, "run", pending)
    window = MainWindow()
    qtbot.addWidget(window)
    payload = {"answer": "<img src='bad'>\n第二行", "citations": []}
    try:
        window._show_answer(payload)
        first = window.audio_worker
        window._show_answer(payload)
        assert first in window.workers
        assert window.audio_worker is not first
        assert "<img src='bad'>" in window.transcript.toPlainText()
    finally:
        release.set()
        qtbot.waitUntil(lambda: not window.workers)


@pytest.mark.ui
def test_query_worker_consumes_sse_completion(qtbot, monkeypatch) -> None:
    from contextlib import contextmanager
    from sell_rag.ui.main_window import QueryWorker

    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            return iter(['event: token', 'data: {"text":"hello\\n"}', '',
                         'event: done', 'data: {"answer":"hello\\n", "citations":[]}'])

    @contextmanager
    def stream(*args, **kwargs):
        yield Response()

    monkeypatch.setattr("sell_rag.ui.main_window.httpx.stream", stream)
    worker = QueryWorker("POST", "http://local/v1/query/stream")
    received, completed = [], []
    worker.token.connect(received.append)
    worker.completed.connect(completed.append)
    worker.run()
    assert received == ["hello\n"]
    assert completed[0]["answer"] == "hello\n"
