# 本地化 RAG 智能知识库 Agent 系统

一套完全本地化部署的企业级智能知识库问答系统，基于 LangChain + Chroma + Ollama 技术栈构建，支持多路检索、重排序精排、多轮对话、Agent 工具调用等核心能力，数据全部本地存储，无需联网，保障数据隐私安全。

## ✨ 核心功能特性

### 🧠 RAG 问答能力
- **多路混合检索**：向量语义检索 + BM25 关键词检索，基于 RRF 倒数秩融合算法加权排序，解决纯语义排序错位问题
- **二阶段精排**：集成 `BAAI/bge-reranker-base` 交叉编码器重排序模型，大幅提升 Top N 召回准确率
- **查询改写**：支持基于对话历史的问题补全与指代消解，自动优化检索问句，提升多轮场景召回率
- **强约束防幻觉**：工业级提示词体系，强制基于参考资料作答，支持原文引用溯源，有效降低幻觉

### 🤖 Agent 智能体
- **ReAct 思考框架**：标准的 Thought-Action-Observation 推理循环，可自主判断是否调用工具
- **可扩展工具体系**：内置知识库问答、数学计算器、系统时间等工具，支持快速扩展自定义工具
- **多轮记忆机制**：支持会话级历史记忆，结合查询改写实现连贯的多轮工具调用对话

### 🔧 工程化能力
- **标准 RESTful API**：基于 FastAPI 封装全量能力，提供统一响应格式与全局异常处理
- **SSE 流式输出**：支持逐字返回答案，对齐主流大模型交互体验
- **会话管理**：服务端会话生命周期管理，自动截断历史长度，平衡效果与性能
- **向量库持久化**：基于 Chroma 嵌入式向量库，零额外部署成本，开箱即用

### 📄 知识库管理
- 支持 Markdown、TXT、PDF 等多格式文档解析
- 可配置文本分块大小与重叠长度
- 支持多集合隔离管理，元数据过滤检索

## 🛠 技术栈

| 层级 | 技术选型 |
|------|----------|
| 大模型推理 | Ollama + Qwen2.5:7b |
| 嵌入模型 | nomic-embed-text |
| 重排序模型 | BAAI/bge-reranker-base |
| 应用框架 | LangChain |
| 向量数据库 | ChromaDB |
| 后端服务 | FastAPI + Uvicorn |
| Agent 框架 | 自研 ReAct 实现 |
| 开发语言 | Python 3.10+ |

## 📁 项目结构

```
rag-agent-local/
├── api/                    # 后端接口层
│   └── app.py              # FastAPI 服务入口
├── core/                   # 业务核心层
│   └── react_agent.py      # ReAct Agent 实现
├── data/                   # 数据目录
├── utils/                  # 工具层
│   ├── document_loader.py  # 文档加载解析
│   ├── rag_base.py         # RAG 核心能力
│   └── text_splitter.py    # 文本分块
├── chroma_db/              # 向量库持久化
├── app_cmd.py              # 命令行交互入口
├── requirements.txt        # 依赖清单
└── README.md
```

## 🚀 快速开始

### 环境要求
- Python 3.10 ~ 3.12
- 已安装并启动 Ollama 服务
- 已拉取 `qwen2.5:7b` 与 `nomic-embed-text` 模型
- 6GB 以上显存（7B 模型 + 重排序模型可流畅运行）

### 1. 克隆项目
```bash
git clone [https://github.com/WuJie-Jay/rag-agent-local.git](https://github.com/WuJie-Jay/rag-agent-local.git)
cd rag-agent-local
```

### 2. 安装依赖
```bash
pip install -r requirements.txt
```

### 3. 启动 Ollama 服务
确保本地 Ollama 服务正常运行，默认端口 11434。

### 4. 构建测试知识库
```bash
python app.py
# 选择功能1，传入文档路径构建知识库
```

### 5. 启动后端 API 服务
```bash
uvicorn api.app:app --host 0.0.0.0 --port 8000 --reload
```

启动后访问接口文档：`[http://localhost:8000/docs](http://localhost:8000/docs)`，可在线调试所有接口。

### 6. 命令行模式
也可以直接通过命令行体验全部功能：
```bash
python app_cmd.py
```
支持三种模式：
1. 构建知识库
2. 基础 RAG 问答
3. Agent 智能对话（支持工具调用）

## 📡 核心 API 接口

| 接口 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 服务健康检查 |
| `/api/kb/build` | POST | 构建知识库，文档向量化入库 |
| `/api/kb/search` | POST | 相似度检索，返回原始片段与元数据 |
| `/api/chat/qa` | POST | 单轮 RAG 问答（非流式） |
| `/api/chat/stream` | POST | 流式 RAG 问答，SSE 协议逐字返回 |
| `/api/chat/session` | POST | 多轮会话问答，自动查询改写 |

完整接口参数与响应格式见在线文档 `/docs`。

## 📈 优化效果

在标准测试集上的检索侧优化对比：

| 指标 | 纯向量检索 | 混合检索 + Rerank | 提升幅度 |
|------|------------|-------------------|----------|
| 目标段落平均排名 | 第 4.2 名 | 第 1.8 名 | 提升 57% |
| Top 3 命中率 | 60% | 100% | +40% |
| Top 5 命中率 | 80% | 100% | +20% |

## 🗺 开发路线

### ✅ 已完成
- [x] 基础向量检索与知识库构建
- [x] BM25 + 向量混合检索（RRF 融合）
- [x] bge-reranker 二阶段精排
- [x] 查询改写与多轮指代消解
- [x] ReAct Agent 框架与基础工具
- [x] FastAPI 后端接口封装
- [x] SSE 流式输出
- [x] 会话管理与统一异常处理

### 🚧 规划中
- [ ] Streamlit 可视化前端界面
- [ ] 多格式文档批量上传入库
- [ ] 多知识库隔离管理
- [ ] 对话历史持久化
- [ ] Docker 一键部署
- [ ] 效果评估可视化面板
- [ ] 向量库平滑升级 Milvus 支持

## 💡 设计亮点

1. **全本地化部署**：所有模型、数据、服务全部运行在本地，数据不出域，隐私安全有保障
2. **分层解耦架构**：接口层、业务层、工具层清晰分离，便于扩展与维护
3. **渐进式优化**：从基础向量检索到混合检索、重排序、查询改写，每一步优化都可量化验证
4. **最小依赖设计**：向量库嵌入式部署，无需额外中间件，降低部署与学习成本
