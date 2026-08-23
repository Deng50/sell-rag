from __future__ import annotations

import json
import os
import sys
import winsound
from pathlib import Path

import httpx
from PyQt5.QtCore import QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QLabel, QLineEdit, QListWidget,
    QCheckBox, QMainWindow, QMessageBox, QPushButton, QStyle, QTabWidget, QTextBrowser,
    QVBoxLayout, QWidget,
)

from sell_rag.multimodal import PresenceGate, PresenceMonitor, VADRecorder


class RequestWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, method: str, url: str, **kwargs):
        super().__init__()
        self.method, self.url, self.kwargs = method, url, kwargs

    def run(self) -> None:
        try:
            response = httpx.request(self.method, self.url, timeout=120, **self.kwargs)
            response.raise_for_status()
            self.completed.emit(response.json())
        except Exception as exc:
            self.failed.emit(str(exc))


class VoiceWorker(QThread):
    completed = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, base_url: str, headers: dict[str, str]):
        super().__init__()
        self.base_url, self.headers = base_url, headers

    def run(self) -> None:
        try:
            audio = VADRecorder().record()
            if not audio:
                raise RuntimeError("没有检测到有效语音")
            response = httpx.post(
                self.base_url + "/v1/speech/transcribe", headers=self.headers,
                files={"audio": ("speech.pcm", audio, "application/octet-stream")}, timeout=60,
            )
            response.raise_for_status()
            self.completed.emit(response.json())
        except Exception as exc:
            self.failed.emit(str(exc))


class AudioWorker(QThread):
    failed = pyqtSignal(str)

    def __init__(self, base_url: str, headers: dict[str, str], text: str):
        super().__init__()
        self.base_url, self.headers, self.text = base_url, headers, text

    def run(self) -> None:
        try:
            response = httpx.post(self.base_url + "/v1/speech/synthesize",
                                  headers=self.headers, json={"text": self.text}, timeout=60)
            response.raise_for_status()
            if response.content.startswith(b"RIFF"):
                winsound.PlaySound(response.content, winsound.SND_MEMORY)
        except Exception as exc:
            self.failed.emit(str(exc))


class CameraWorker(QThread):
    presence = pyqtSignal()
    failed = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.monitor = None

    def run(self) -> None:
        try:
            gate = PresenceGate(window=8, required_hits=5, cooldown_seconds=30)
            self.monitor = PresenceMonitor(gate, self.presence.emit,
                                           os.getenv("SELL_RAG_DETECTOR_MODEL", "yolov5s.pt"))
            self.monitor.run()
        except Exception as exc:
            self.failed.emit(str(exc))

    def stop(self) -> None:
        if self.monitor:
            self.monitor.stop()


class MainWindow(QMainWindow):
    def __init__(self, base_url: str = "http://127.0.0.1:8765"):
        super().__init__()
        self.base_url = base_url.rstrip("/")
        self.workers: list[RequestWorker] = []
        self.voice_worker = None
        self.audio_worker = None
        self.camera_worker = None
        self.setWindowTitle("Sell-RAG 智能售卖终端")
        self.resize(980, 680)
        tabs = QTabWidget()
        tabs.addTab(self._chat_tab(), "智能问答")
        tabs.addTab(self._admin_tab(), "知识库管理")
        self.setCentralWidget(tabs)
        self.statusBar().showMessage("正在连接本地服务...")
        self._request("GET", "/health", self._health)

    def _chat_tab(self) -> QWidget:
        widget, layout = QWidget(), QVBoxLayout()
        title = QLabel("小棠 · 智能售卖助手")
        title.setFont(QFont("Microsoft YaHei", 17, QFont.Bold))
        header = QHBoxLayout()
        header.addWidget(title)
        header.addStretch(1)
        self.camera_toggle = QCheckBox("人员感知")
        self.camera_toggle.setToolTip("仅检测人员是否出现；不保存画面，不做人脸或人口属性识别")
        self.camera_toggle.toggled.connect(self._toggle_camera)
        header.addWidget(self.camera_toggle)
        layout.addLayout(header)
        self.transcript = QTextBrowser()
        self.transcript.setOpenExternalLinks(False)
        layout.addWidget(self.transcript, 1)
        row = QHBoxLayout()
        self.query = QLineEdit()
        self.query.setPlaceholderText("询问商品价格、库存、推荐或知识库内容")
        self.query.returnPressed.connect(self._ask)
        row.addWidget(self.query, 1)
        voice = QPushButton()
        voice.setIcon(self.style().standardIcon(QStyle.SP_MediaVolume))
        voice.setToolTip("语音输入")
        voice.clicked.connect(self._record_voice)
        row.addWidget(voice)
        send = QPushButton("发送")
        send.setIcon(self.style().standardIcon(QStyle.SP_ArrowForward))
        send.clicked.connect(self._ask)
        row.addWidget(send)
        layout.addLayout(row)
        self.citations = QListWidget()
        self.citations.setMaximumHeight(130)
        layout.addWidget(QLabel("引用证据"))
        layout.addWidget(self.citations)
        widget.setLayout(layout)
        return widget

    def _admin_tab(self) -> QWidget:
        widget, layout = QWidget(), QVBoxLayout()
        row = QHBoxLayout()
        upload = QPushButton("上传文档")
        upload.setIcon(self.style().standardIcon(QStyle.SP_DialogOpenButton))
        upload.clicked.connect(self._upload)
        refresh = QPushButton("刷新")
        refresh.setIcon(self.style().standardIcon(QStyle.SP_BrowserReload))
        refresh.clicked.connect(self._refresh_documents)
        evaluate = QPushButton("运行基准评测")
        evaluate.clicked.connect(self._evaluate)
        row.addWidget(upload)
        row.addWidget(refresh)
        row.addWidget(evaluate)
        row.addStretch(1)
        layout.addLayout(row)
        self.documents = QListWidget()
        layout.addWidget(self.documents, 1)
        self.admin_output = QTextBrowser()
        self.admin_output.setMaximumHeight(180)
        layout.addWidget(self.admin_output)
        widget.setLayout(layout)
        return widget

    def _headers(self, role: str = "guest") -> dict[str, str]:
        headers = {"X-Role": role}
        token = os.getenv("SELL_RAG_ADMIN_TOKEN" if role == "admin" else "SELL_RAG_OPERATOR_TOKEN")
        if token:
            headers["Authorization"] = f"Bearer {token}"
        return headers

    def _request(self, method: str, path: str, callback, **kwargs) -> None:
        worker = RequestWorker(method, self.base_url + path, **kwargs)
        self.workers.append(worker)
        worker.completed.connect(callback)
        worker.failed.connect(lambda message: QMessageBox.warning(self, "请求失败", message))
        worker.finished.connect(lambda: self.workers.remove(worker) if worker in self.workers else None)
        worker.start()

    def _ask(self) -> None:
        text = self.query.text().strip()
        if not text:
            return
        self.query.clear()
        self.transcript.append(f"<b>你：</b>{text}")
        self._request("POST", "/v1/query", self._show_answer,
                      headers=self._headers(), json={"query": text})

    def _show_answer(self, payload: dict) -> None:
        self.transcript.append(f"<b>助手：</b>{payload['answer']}")
        self.citations.clear()
        for citation in payload.get("citations", []):
            where = citation.get("page_no") or citation.get("sheet_name") or ""
            self.citations.addItem(f"[{citation['citation_id']}] {citation['title']} {where}: {citation['excerpt']}")
        self.audio_worker = AudioWorker(self.base_url, self._headers(), payload["answer"])
        self.audio_worker.start()

    def _upload(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择知识库文件", "", "Documents (*.pdf *.docx *.xlsx *.csv *.json *.txt *.md *.png *.jpg *.jpeg)")
        if not path:
            return
        files = {"file": (Path(path).name, Path(path).read_bytes())}
        self._request("POST", "/v1/documents", self._job_started,
                      headers=self._headers("operator"), files=files)

    def _job_started(self, payload: dict) -> None:
        self.admin_output.append(f"任务已创建：{payload['job_id']}")

    def _refresh_documents(self) -> None:
        self._request("GET", "/v1/documents", self._show_documents,
                      headers=self._headers("operator"))

    def _show_documents(self, payload: list[dict]) -> None:
        self.documents.clear()
        for item in payload:
            active = "当前" if item["active"] else "历史"
            self.documents.addItem(f"{item['filename']} · v{item['version']} · {item['status']} · {active} · ACL {item['acl']}")

    def _evaluate(self) -> None:
        self._request("POST", "/v1/evaluations", lambda payload: self.admin_output.append(json.dumps(payload, ensure_ascii=False, indent=2)),
                      headers=self._headers("operator"), json={})

    def _health(self, payload: dict) -> None:
        message = f"服务正常 · {payload['mode']} · {payload['embedding']}"
        if payload.get("degraded"):
            message += " · 已降级"
        self.statusBar().showMessage(message)

    def _record_voice(self) -> None:
        winsound.PlaySound(None, winsound.SND_PURGE)
        if self.voice_worker and self.voice_worker.isRunning():
            return
        self.statusBar().showMessage("正在聆听...")
        self.voice_worker = VoiceWorker(self.base_url, self._headers())
        self.voice_worker.completed.connect(self._voice_result)
        self.voice_worker.failed.connect(lambda message: QMessageBox.warning(self, "语音输入失败", message))
        self.voice_worker.start()

    def _voice_result(self, payload: dict) -> None:
        text = payload.get("text", "")
        if not text:
            return
        if payload.get("needs_confirmation"):
            choice = QMessageBox.question(self, "确认识别结果", f"识别为：{text}\n是否继续？")
            if choice != QMessageBox.Yes:
                return
        self.query.setText(text)
        self._ask()

    def _toggle_camera(self, enabled: bool) -> None:
        if enabled:
            self.camera_worker = CameraWorker()
            self.camera_worker.presence.connect(
                lambda: self.transcript.append("<b>助手：</b>你好，需要我推荐商品吗？")
            )
            self.camera_worker.failed.connect(self._camera_failed)
            self.camera_worker.start()
        elif self.camera_worker:
            self.camera_worker.stop()

    def _camera_failed(self, message: str) -> None:
        self.camera_toggle.setChecked(False)
        QMessageBox.warning(self, "人员感知不可用", message)

    def closeEvent(self, event) -> None:
        if self.camera_worker:
            self.camera_worker.stop()
            self.camera_worker.wait(2000)
        winsound.PlaySound(None, winsound.SND_PURGE)
        super().closeEvent(event)


def run_ui(base_url: str = "http://127.0.0.1:8765") -> int:
    application = QApplication.instance() or QApplication(sys.argv)
    application.setStyle("Fusion")
    window = MainWindow(base_url)
    window.show()
    return application.exec_()
