# Sell-RAG：多模态售卖终端知识助手

面向本地智能售卖终端的检索增强问答系统。导入商品表格、校史资料和服务说明后，可进行带引用的知识问答、精确价格/库存查询和预算推荐；同时提供 HTTP API、命令行和 PyQt 桌面客户端。

**可以先用默认 Fake 模式离线体验。** 核心流程无需云端密钥或大型模型；真实语义模型、语音、OCR 和摄像头能力需按需安装依赖并配置服务或设备。

## 功能概况

| 能力 | 当前实现 |
| --- | --- |
| 文档接入 | 文本、Markdown、CSV、JSON、XLSX；PDF、DOCX、扫描件和图片按解析依赖启用，保留章节、页码和表格元数据 |
| 商品查询 | SQLite 保存 SKU、名称、价格和库存；支持精确识别、预算筛选和缺货信息 |
| 知识问答 | 按角色检索、证据组装、引用校验、缺少依据时拒答；真实模式可使用 BGE、BM25、RRF 与 Cross-Encoder |
| 版本管理 | 内容去重、商品快照、历史激活、软删除；文档、商品与索引在构建成功后共同发布 |
| 权限隔离 | `guest`、`operator`、`admin` 角色及文档/商品 ACL；检索时核验版本和权限有效性 |
| 桌面交互 | 问答与知识库管理双视图，SSE 答案显示、上传任务轮询、后台线程取消与关闭管理 |
| 多模态 | 百度 ASR/TTS、语音活动检测、热词纠错、低置信确认、分句播放与打断；可选人员存在检测 |
| 评测诊断 | 检索、引用、拒答、解析、延迟评测，健康与就绪接口 |

### 2026-10-04 完善内容

- 商品回滚、删除与索引发布保持一致；索引失败继续使用旧知识，支持跨进程刷新和发布冲突检测。
- 修复同名商品/SKU 混淆、预算绕过、价格精度和超大预算溢出。
- 修复文本切片、CSV/JSON、DOCX/Docling 阅读顺序，校验非法商品数字和未计算的 Excel 公式。
- 完善上传文件身份、SSE 换行、语音队列隔离、界面线程与任务状态反馈。
- 修正重复文档和无标注数据导致的评测分数虚高。详见 [程序审查报告](docs/audit-2026-10-04.md)。

## 快速开始：离线文字问答

要求 **Python 3.11 或 3.12**，主要运行与验证平台为 Windows。以下命令使用 **Windows PowerShell**，在仓库根目录执行，直接调用虚拟环境程序，无需激活脚本。

### 1. 安装核心依赖

```powershell
git clone https://github.com/Deng50/sell-rag.git
cd sell-rag
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.lock
.\.venv\Scripts\python.exe -m pip install -e . --no-deps
```

已有仓库时跳过克隆；使用 Python 3.11 时，将创建环境命令中的 `-3.12` 改为 `-3.11`。

### 2. 导入自带示例

```powershell
$env:SELL_RAG_FAKE_PROVIDERS = "true"
.\.venv\Scripts\sell-rag.exe ingest .\tests\fixtures\products.csv
.\.venv\Scripts\sell-rag.exe ingest .\tests\fixtures\knowledge.md
```

命令会创建本地 `runtime/` 数据、解析文件并发布索引。示例商品包含清泉饮用水、每日坚果和校园纪念杯，知识文件包含校史和售卖说明。

### 3. 启动服务并提问

```powershell
.\.venv\Scripts\sell-rag.exe serve
```

打开 [API 文档](http://127.0.0.1:8765/docs)，在 `POST /v1/query` 中输入：

```json
{"query": "清泉饮用水多少钱？"}
```

也可在另一个 PowerShell 终端调用接口：

```powershell
$body = @{ query = "清泉饮用水多少钱？" } | ConvertTo-Json
Invoke-RestMethod -Uri "http://127.0.0.1:8765/v1/query" -Method Post -ContentType "application/json; charset=utf-8" -Body ([System.Text.Encoding]::UTF8.GetBytes($body))
```

可以继续尝试“校园纪念杯还有库存吗？”或“北洋大学堂是什么时候创建的？”。Fake 模式用于验证流程和结构化数据，不能代表真实模型的语言理解与回答质量。

## 桌面客户端与可选能力

服务保持运行，另开终端进入同一仓库：

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[ui]"
.\.venv\Scripts\sell-rag.exe ui
```

用户视图用于问答；知识库管理视图可上传文档并查看处理结果。管理操作需要令牌，服务端与界面进程应配置相同的 `SELL_RAG_OPERATOR_TOKEN`。语音、摄像头需另装对应依赖。

| 依赖组 | 安装命令 | 用途 |
| --- | --- | --- |
| 文档 | `.\.venv\Scripts\python.exe -m pip install -e ".[document]"` | Docling、PDF、DOCX；完整 OCR 可能还需模型资源 |
| 检索 | `.\.venv\Scripts\python.exe -m pip install -e ".[retrieval]"` | BGE、Cross-Encoder、FAISS、PyTorch |
| 音频 | `.\.venv\Scripts\python.exe -m pip install -e ".[audio]"` | 录音、VAD、播放 |
| 视觉 | `.\.venv\Scripts\python.exe -m pip install -e ".[vision]"` | 摄像头与人员检测 |
| 开发 | `.\.venv\Scripts\python.exe -m pip install -e ".[dev]"` | 测试、覆盖率与静态检查 |

真实模型首次使用可能下载权重，GPU 运行需匹配的 PyTorch/CUDA。可选依赖不由核心 `requirements.lock` 完整锁定。

## 配置与真实服务

参数在 [configs/default.yaml](configs/default.yaml)，密钥和开关从**进程环境变量**读取。当前程序不会自动加载根目录 `.env`；[.env.example](.env.example) 仅为变量参考。

```powershell
$env:SELL_RAG_OPERATOR_TOKEN = "替换为你的操作员令牌"
$env:SELL_RAG_ADMIN_TOKEN = "替换为你的管理员令牌"
```

启用真实适配器时，在服务启动前配置：

```powershell
$env:SELL_RAG_FAKE_PROVIDERS = "false"
$env:ZHIPUAI_API_KEY = "填写你的智谱密钥"
$env:BAIDU_API_KEY = "填写你的百度语音密钥"
$env:BAIDU_SECRET_KEY = "填写你的百度语音密钥"
$env:SELL_RAG_DEVICE = "cpu"
```

准备好 GPU 后可改为 `cuda`。可用 `SELL_RAG_CONFIG` 指定其他 YAML 文件。各终端从同一仓库根目录启动，避免相对路径产生不同数据目录。

## 导入与管理自己的知识

```powershell
.\.venv\Scripts\sell-rag.exe ingest .\data\商品目录.xlsx
.\.venv\Scripts\sell-rag.exe ingest .\data\内部说明.pdf --acl operator,admin
.\.venv\Scripts\sell-rag.exe reindex
.\.venv\Scripts\sell-rag.exe rollback DOCUMENT_ID VERSION
```

替换上述文件路径和版本参数为实际值。CLI 是本机管理入口；HTTP 管理接口需要认证。商品表可参照 [示例 CSV](tests/fixtures/products.csv)，包含编号、名称、价格、库存等字段。库存不能为负数或小数，Excel 公式须先计算并保存结果。

上传接口返回任务编号，通过 `/v1/jobs/{job_id}` 查看结果。需要发布一致性时使用 CLI/API 管理流程。旧数据库从未保存的商品历史不能凭空恢复，缺少快照时须重新摄取相应原始文件。

## 常用接口

| 接口 | 用途 |
| --- | --- |
| `POST /v1/query`、`POST /v1/query/stream` | 同步问答 / SSE 回答、引用和完成事件 |
| `POST /v1/documents`、`GET /v1/documents` | 上传与查询文档 |
| `POST /v1/documents/{document_id}/versions/{version}/activate` | 激活历史版本 |
| `DELETE /v1/documents/{document_id}` | 软删除并更新索引 |
| `GET /v1/jobs/{job_id}` | 查询后台任务 |
| `GET /v1/products`、`POST /v1/products` | 商品查询与维护 |
| `POST /v1/speech/transcribe`、`POST /v1/speech/synthesize` | 语音识别与合成 |
| `POST /v1/feedback`、`POST /v1/evaluations` | 反馈与评测 |
| `GET /health`、`GET /ready` | 健康与就绪检查 |

管理请求附带 `X-Role: operator` 或 `admin`，以及对应的 `Authorization: Bearer <令牌>`。请求体角色不能代替认证；普通用户默认为 `guest`。字段和具体权限以运行后的 `/docs` 为准。

## 架构与目录

```text
文件 → 校验/解析 → 切片与商品快照 → 构建索引 → 共同发布
问题 → 查询路由 → SQLite / 角色隔离检索 → 证据与引用 → 回答

configs/                       默认参数
src/sell_rag/ingestion/         文档校验、解析和切片
src/sell_rag/storage/           SQLite、版本和商品快照
src/sell_rag/indexing/          模型适配与索引发布
src/sell_rag/retrieval/         商品路由、检索和重排
src/sell_rag/generation/        生成与引用回答
src/sell_rag/multimodal/        语音和人员检测
src/sell_rag/api/、ui/          HTTP 与桌面入口
src/sell_rag/evaluation/        本地评测
tests/                         测试和示例文件
runtime/                       本地数据库、文档、索引和日志
```

## 测试与当前边界

```powershell
.\.venv\Scripts\python.exe -m pip install -e ".[dev,ui]" python-docx pypdf
$env:QT_QPA_PLATFORM = "offscreen"
.\.venv\Scripts\python.exe -m pytest --cov=sell_rag --cov-report=term
.\.venv\Scripts\python.exe -m ruff check --isolated --select E9,F63,F7,F82 src tests
.\.venv\Scripts\sell-rag.exe evaluate .\tests\fixtures\evaluation.jsonl
```

2026-10-04 审查时 **60 项测试通过，语句覆盖率 78%**。默认测试不访问云 API 或下载大型模型；云服务、完整 OCR、GPU、摄像头和麦克风仍需实机验收。

- SSE 发送引用校验后的分段答案，首段需等待生成完成，不是上游模型首 token 即时输出。
- Fake TTS 返回有效静音 WAV，实际朗读需要百度语音服务。
- 缺少人工相关性标注的评测指标返回 `null`。API 评测文件限于 `tests/fixtures/` 或 `runtime/evaluations/`。
- 摄像头只检测人员是否出现，画面不持久化，不推断身份、年龄或性别。
- 管理接口 401 时，检查令牌配置、角色和认证头是否一致。
- `runtime/`、密钥和模型不提交 Git，业务数据需要单独备份。

## 许可证

当前仓库未提供独立开源许可证文件；使用和分发前请与作者确认授权范围。
