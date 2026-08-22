import os
import tkinter as tk
from tkinter import filedialog
from tkinter import messagebox
from rag import Chatdoc


class add_files():
    def __init__(self):
        self.chat_doc = Chatdoc()

    def add_file(self, path, name):
        self.chat_doc.doc = path
        self.chat_doc.name = name
        document = self.chat_doc.embeddingAndVectorDB()
        return document


def add_file(path, name):
    chat_doc = Chatdoc()
    chat_doc.doc = path
    chat_doc.name = name
    document = chat_doc.embeddingAndVectorDB()
    return document


def choose_file_or_folder():
    """让用户选择文件或文件夹路径"""
    path = filedialog.askopenfilename(title="选择文件")
    if not path:  # 如果用户没有选择文件，尝试选择文件夹
        path = filedialog.askdirectory(title="选择文件夹")
    if path:
        path_entry.config(state=tk.NORMAL)
        path_entry.delete(0, tk.END)
        path_entry.insert(0, path)
        path_entry.config(state=tk.DISABLED)


def choose_file():
    """让用户选择文件路径"""
    path = filedialog.askopenfilename(title="选择文件")
    if path:
        path_entry.config(state=tk.NORMAL)
        path_entry.delete(0, tk.END)
        path_entry.insert(0, path)
        path_entry.config(state=tk.DISABLED)


def choose_folder():
    """让用户选择文件夹路径"""
    path = filedialog.askdirectory(title="选择文件夹")
    if path:
        path_entry.config(state=tk.NORMAL)
        path_entry.delete(0, tk.END)
        path_entry.insert(0, path)
        path_entry.config(state=tk.DISABLED)


def submit_action():
    """处理提交按钮点击事件"""
    file_list = []
    file_or_folder_path = path_entry.get()
    user_input_text = input_text.get()

    if not file_or_folder_path:
        messagebox.showwarning("警告", "请选择一个文件或文件夹路径！")
        return

    if not user_input_text:
        messagebox.showwarning("警告", "请输入内容！")
        return

    if os.path.isdir(file_or_folder_path):
        # 如果是文件夹，则扫描其中的所有文件
        print(f"扫描文件夹: {file_or_folder_path}")
        for root, dirs, files in os.walk(file_or_folder_path):
            for file in files:
                print(os.path.join(root, file))
                file_list.append(os.path.join(root, file))
    elif os.path.isfile(file_or_folder_path):
        file_list.append(file_or_folder_path)
        # 如果是单个文件，直接输出路径
        print(f"单个文件: {file_or_folder_path}")
    else:
        messagebox.showerror("错误", "路径无效，请重新选择！")
        return

    # 处理用户输入并调用其他逻辑
    print(f"路径: {file_or_folder_path}")
    print(f"用户输入: {user_input_text}")
    # add_file(path=file_or_folder_path, name=user_input_text)
    add_file(path=file_list, name=user_input_text)
    messagebox.showinfo("提交成功", "知识库已创建！")


# 创建主窗口
root = tk.Tk()
root.title("文件选择和输入示例")
root.geometry("800x180")

# 文件/文件夹路径选择
frame1 = tk.Frame(root)
frame1.pack(pady=10, padx=10, fill=tk.X)

path_label = tk.Label(frame1, text="文件/文件夹路径:")
path_label.pack(side=tk.LEFT)

path_entry = tk.Entry(frame1, width=30, state=tk.DISABLED)
path_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

choose_file_button = tk.Button(frame1, text="选择文件", command=choose_file)
choose_file_button.pack(side=tk.LEFT, padx=5)

choose_folder_button = tk.Button(frame1, text="选择文件夹", command=choose_folder)
choose_folder_button.pack(side=tk.LEFT, padx=5)

# 文本输入框
frame2 = tk.Frame(root)
frame2.pack(pady=10, padx=10, fill=tk.X)

input_label = tk.Label(frame2, text="输入知识库名称:")
input_label.pack(side=tk.LEFT)

input_text = tk.Entry(frame2, width=30)
input_text.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)

# 提交按钮
frame3 = tk.Frame(root)
frame3.pack(pady=20)

submit_button = tk.Button(frame3, text="生成知识库", command=submit_action)
submit_button.pack()

# 运行主循环
root.mainloop()
