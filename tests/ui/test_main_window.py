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
