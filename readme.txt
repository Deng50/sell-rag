Sell-RAG 智能无人售卖交互系统
===============================

一、项目简介
------------
Sell-RAG 是一个面向无人售卖场景的智能交互原型。项目将摄像头行人检测、图像理解、
检索增强生成（RAG）、语音识别/合成和桌面对话界面组合在一起，可根据商品资料和用户
提问进行自然语言回答与商品推荐。

二、主要功能
------------
1. 使用 YOLOv5 和摄像头检测人员并截取图像。
2. 调用智谱 AI 视觉模型分析图像中的人物特征。
3. 使用本地 Embedding 模型和 Chroma 构建、查询商品知识库。
4. 调用大语言模型生成商品问答和个性化推荐。
5. 通过百度语音接口实现中文语音识别与语音合成。
6. 提供 PyQt5 文本/语音对话界面。
7. C8 目录包含另一套数据准备、索引构建、检索优化和生成集成示例。

三、目录说明
------------
zhipuai_rag/main.py                  PyQt5 对话界面入口
zhipuai_rag/sell.py                  摄像头检测、视觉分析和语音对话主流程
zhipuai_rag/detect.py                独立的 YOLOv5 人员检测示例
zhipuai_rag/audio.py                 录音、语音识别和语音合成
zhipuai_rag/rag.py                   文档加载、向量检索和大模型问答
zhipuai_rag/tokenizer.py             Xiaobu Embedding 封装
zhipuai_rag/add_files.py             知识库文件添加工具
zhipuai_rag/rag_data_preprocessor.py RAG 数据预处理
zhipuai_rag/rag_vector_retriever.py  RAG 向量检索器
zhipuai_rag/C8/                      模块化 RAG 示例

四、环境要求
------------
- 建议使用 Windows 和 Python 3.9 或兼容版本。
- 摄像头和麦克风用于完整交互流程。
- 安装 PyAudio 时可能需要系统音频开发组件或预编译 wheel。
- requirements.txt 中的 torch 使用 CUDA 11.8 版本；没有 NVIDIA GPU 时请按 PyTorch
  官方说明安装合适的 CPU 版本。

五、安装依赖
------------
进入核心代码目录并创建虚拟环境：

  cd zhipuai_rag
  python -m venv .venv
  .venv\Scripts\activate
  pip install -r requirements.txt

六、配置密钥
------------
本仓库不保存任何真实 API 密钥。运行前请在 PowerShell 中设置环境变量：

  $env:ZHIPUAI_API_KEY="你的智谱AI密钥"
  $env:BAIDU_API_KEY="你的百度语音API Key"
  $env:BAIDU_SECRET_KEY="你的百度语音Secret Key"
  $env:MOONSHOT_API_KEY="你的Moonshot密钥"

其中 ZHIPUAI_API_KEY 用于主程序；BAIDU_API_KEY 和 BAIDU_SECRET_KEY 用于语音功能；
MOONSHOT_API_KEY 仅用于 C8 示例。也可以参考仓库根目录的 .env.example，但程序默认读取
操作系统环境变量，不会自动加载 .env 文件。

七、准备外部资源
--------------
为控制仓库体积，本仓库不包含模型文件、向量数据库和第三方源码。运行前需要自行准备：

1. 将 YOLOv5 权重放到 zhipuai_rag/weights/，默认文件为 yolov5s.pt。
2. 将 xiaobu-embedding-v2 模型放到
   zhipuai_rag/RAG/xiaobu-embedding-v2/。
3. 按需准备商品文档，再运行 add_files.py 或相关预处理脚本建立向量数据库。
4. 向量数据库默认生成在 zhipuai_rag/dataset/chroma_db/，该目录不会提交到 Git。

八、运行方式
------------
所有相对路径均以 zhipuai_rag 为工作目录，因此请先进入该目录。

启动完整的摄像头和语音交互流程：

  python sell.py

启动桌面对话界面：

  python main.py

仅运行人员检测示例：

  python detect.py

九、安全说明
------------
- 不要把真实 API 密钥写入源码、配置样例或提交记录。
- .gitignore 已排除 .env、模型、向量库、缓存和运行生成文件。
- 图像人物特征分析可能涉及隐私，请在取得授权并遵守适用法律的前提下使用。
- 本项目是研究与原型代码，部署到生产环境前应补充异常处理、权限控制、日志脱敏和测试。

十、许可证
----------
当前项目尚未提供许可证。未经作者明确授权，不应假定代码可用于商业分发。
