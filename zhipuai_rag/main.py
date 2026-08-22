import json
import sys
from audio import audio_module, audio_output
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QTextEdit, QPushButton, QListWidget, \
    QListWidgetItem, QHBoxLayout, QCheckBox
from rag import Chatdoc
import threading
from pathlib import Path


class ChatApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.chat_doc = Chatdoc()
        self.chat_doc.doc = "./files/test.docx"

    def initUI(self):
        self.setWindowTitle("AI Chat Interface")
        self.setGeometry(30, 10, 400, 300)

        layout = QVBoxLayout()

        self.label = QLabel("Output:")
        layout.addWidget(self.label)

        self.output_text = QTextEdit()
        self.output_text.setReadOnly(True)
        layout.addWidget(self.output_text)

        self.input_text = QTextEdit()
        self.input_text.setPlaceholderText("Type your question here...")
        layout.addWidget(self.input_text)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.on_submit)
        layout.addWidget(self.submit_button)

        self.voice_button = QPushButton("Record Voice")
        self.voice_button.clicked.connect(self.use_voice_input)
        layout.addWidget(self.voice_button)

        checkbox_layout = QHBoxLayout()
        self.checkboxes = []

        options = self.load_options_from_json("./dataset/chroma_db/directory.json")  # 加载选项

        for option in options:
            checkbox = QCheckBox(option)
            self.checkboxes.append(checkbox)
            checkbox_layout.addWidget(checkbox)

        layout.addLayout(checkbox_layout)

        self.setLayout(layout)

    def load_options_from_json(self, file_path):
        """从JSON文件加载选项"""
        path = Path(file_path)
        if path.exists():
            with path.open('r', encoding='utf-8') as file:
                return json.load(file)
        else:
            print(f"警告: {file_path} 文件不存在。")
            return []  # 如果文件不存在，则返回空列表

    def on_submit(self):
        # 获取所有被选中的选项
        selected_options = [cb.text() for cb in self.checkboxes if cb.isChecked()]

        # 打印或处理所选选项
        print("Selected options:", selected_options)
        prompt = self.input_text.toPlainText()
        self.get_response(prompt, selected_options)

    def get_response(self, prompt, name):
        if prompt:
            ai_message = self.chat_doc.chatWithDoc(prompt, name)

            self.output_text.append("User: " + prompt)
            self.output_text.append("AI: " + ai_message + "\n")

            threading.Thread(target=audio_output, args=(ai_message,)).start()
            self.input_text.clear()

    def use_voice_input(self):
        recognized_text = audio_module()
        if recognized_text:
            self.get_response(recognized_text)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    chat_app = ChatApp()
    chat_app.show()
    sys.exit(app.exec_())
