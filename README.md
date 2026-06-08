# 🔬 科研灵感助手 (Research Inspiration Agent)

> 基于 LLM + RAG 的科研辅助 Agent，帮助研究人员进行文献知识库管理、论文检索、研究方向分析和科研方案讨论。

[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📖 目录

- [项目简介](#项目简介)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [使用指南](#使用指南)
- [已知局限与 v2.0 规划](#已知局限与-v20-规划)
- [版本历史](#版本历史)

---

## 项目简介

**科研灵感助手**是一个面向学术研究人员的 AI Agent。以**研究方向**为锚点，提供从**文献入库 → 知识检索 → 联网搜索 → 综述分析 → 方案讨论**的全流程辅助。

核心定位：**知识库论文分析为主，联网搜索为辅**，帮助研究人员系统性地理解某一方向的研究现状、发现研究空白、验证科研想法。

### 典型使用场景

```
用户: "帮我分析一下部分伪造音频检测的研究现状"

Agent → search_knowledge_base("部分伪造音频检测")
      → 混合检索 (6 个查询变体 × BM25+语义)
      → 检索到 5 篇最相关论文片段
      → 按细分领域（伪造内容检测 vs 伪造边界定位）组织回复
      → 标注引用来源（论文名称 + 章节）
      → 结尾总结研究空白与潜在创新方向
```

---

## 核心功能

### 1. 📚 知识库管理

| 功能 | 说明 |
|------|------|
| **PDF 智能解析** | IBM Docling 将 PDF 转为结构化 Markdown，自动识别标题层级、表格、公式、OCR 扫描页 |
| **MD5 去重** | 上传前计算文件 MD5 哈希，已存在的论文自动拒绝入库 |
| **按章节分块** | 按 `##` / `###` Markdown 标题切分为逻辑分块（1000 字符/块），保留文档层级结构 |
| **向量化存储** | DashScope `text-embedding-v4` 将 chunk 转为 1024 维向量，存入 Chroma |
| **BM25 索引** | jieba 分词后构建 BM25Okapi 索引，与语义检索互补 |

### 2. 🔍 混合 RAG 检索（三层增强）

```
用户查询
    │
    ▼
L1: LLM 查询改写 → 5 个同义变体
    "partial spoofing" / "partial fake" / "audio forgery localization" / ...
    │
    ├─→ 每个变体独立检索:
    │     BM25 (关键词, jieba分词)
    │      + 
    │     语义检索 (向量相似度, Chroma cosine)
    │     → RRF(k=60) 融合 → 各取 top-15
    │
    ▼
L2: 全局去重 → ~50 候选池
    │
    ▼
L3: Cross-Encoder 精排 → top-5
    (BAAI/bge-reranker-large, 不可用时自动回退 RRF)
```

### 3. 🌐 联网论文搜索

| 数据源 | 状态 | 说明 |
|--------|:---:|------|
| **OpenAlex** | ✅ 主力 | 免费开放学术索引，无速率限制，PDF 下载链接丰富 |
| **arXiv** | ✅ 可用 | 双路径请求（urllib 直连 + 显式代理 `127.0.0.1:7897`） |
| **Semantic Scholar** | ⚠️ 限速 | 120s 冷却计时，可用时自动加入并行检索 |

- **三源并行**：ThreadPoolExecutor 同时检索，单源失败不影响其他源
- **年份过滤**：自动从 query 中提取年份提示，或指定 `since_year`
- **摘要翻译**：批量调用 LLM 将英文 abstract 翻译为中文
- **论文卡片**：前端展示标题、作者、年份、中英文摘要、PDF 下载按钮
- **一键入库**：点击按钮下载 PDF 并加入知识库

### 4. 💬 会话管理

- **全量持久化**：MySQL 存储所有消息、摘要、知识库文档和分块，重启不丢失
- **会话恢复**：切换历史会话时自动恢复聊天记录和论文卡片（论文元数据以 JSON 嵌入消息存储）
- **自动摘要**：消息数 ≥ 20 条自动触发 LLM 生成四段式摘要（主题 / 关键讨论 / 涉及论文 / 待办事项），摘要自动注入后续上下文

### 5. 📋 全链路日志追踪

- **四通道日志**：控制台彩色 + 按天滚动文件 `logs/agent.log` + 错误日志 `logs/error.log` + 内存环形缓冲（500 条）
- **trace_id**：每个请求自动生成唯一 ID，跨 LLM 调用和 Tool 执行全程携带
- **前端面板**：Streamlit 侧边栏实时日志，按级别/标签筛选，支持一键导出

---

## 系统架构

```
┌─────────────────────────────────────────────────┐
│               Streamlit 前端 (app.py)              │
│   聊天界面 · 知识库上传 · 会话管理 · 日志面板       │
├─────────────────────────────────────────────────┤
│          Agent 核心 (agent_core.py)                │
│    ReAct 模式 · 上下文编排 · Tool 路由              │
├──────────────────┬──────────────────────────────┤
│   tools.py       │   paper_search_mcp.py (MCP)   │
│   知识库检索 (RAG) │   联网论文搜索                 │
├──────────────────┴──────────────────────────────┤
│               LLM 服务 (llm_service.py)            │
│          Qwen3-max (DashScope API)                │
├─────────────────────────────────────────────────┤
│                  知识层                            │
│  retriever · reranker · query_rewriter            │
│  document_processor (Docling) · embedding_service │
│  vector_store (Chroma)                            │
├─────────────────────────────────────────────────┤
│                  数据层                            │
│  MySQL (sessions · messages · summaries          │
│         documents · chunks)                       │
└─────────────────────────────────────────────────┘
```

> **注意**：本项目**没有** FastAPI 或独立 HTTP 服务端。Streamlit 自身即 Web Server，所有 Python 模块在同一进程内直接调用，无网络开销。

---

## 快速开始

### 环境要求

- Python 3.10+
- MySQL 8.0+
- [DashScope API Key](https://dashscope.aliyun.com/) （通义千问 + Embedding）
- [Clash Verge](https://github.com/clash-verge-rev/clash-verge-rev) 等代理工具（联网搜索需要）

### 1. 克隆项目

```bash
git clone https://github.com/your-username/research-inspiration-agent.git
cd research-inspiration-agent
```

### 2. 安装依赖

```bash
conda create -n agent_rag python=3.10
conda activate agent_rag
pip install -r requirements.txt
```

### 3. 配置环境变量

```bash
cp .env.example .env
```

编辑 `.env`，至少填入：

```env
DASHSCOPE_API_KEY=your-dashscope-api-key
MYSQL_PASSWORD=your-mysql-password
```

### 4. 创建数据库

```sql
CREATE DATABASE IF NOT EXISTS research_assistant
  CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

### 5. 启动

```bash
conda activate agent_rag
streamlit run app.py
```

浏览器打开 `http://localhost:8501`

---

## 项目结构

```
research-inspiration-agent/
├── app.py                              # Streamlit 前端入口
├── config.py                           # 全局配置 + 环境变量
├── requirements.txt                    # Python 依赖
├── .env.example                        # 环境变量模板
├── README.md
├── 项目架构与工作流程详解.md              # 详细架构文档
│
├── agent/                              # Agent 核心
│   ├── agent_core.py                   # ReAct 主循环 (SYSTEM_PROMPT + Tool 路由)
│   ├── llm_service.py                  # Qwen3-max (OpenAI 兼容协议 + 重试)
│   ├── tools.py                        # search_knowledge_base Tool
│   ├── paper_search_mcp.py             # search_papers_online Tool (MCP 风格)
│   └── middleware.py                   # 工具调用追踪 + 日志
│
├── knowledge_base/                     # 知识库与检索
│   ├── document_processor.py           # Docling PDF→MD + 章节分块
│   ├── embedding_service.py            # DashScope text-embedding-v4
│   ├── vector_store.py                 # Chroma 向量存储
│   ├── retriever.py                    # BM25 + 语义 + RRF 混合检索
│   ├── reranker.py                     # Cross-Encoder 重排序
│   ├── query_rewriter.py              # LLM 查询改写 (多角度变体)
│   └── paper_search.py                 # 三源论文检索 (arXiv+OA+S2)
│
├── session/                            # 会话与摘要
│   ├── session_manager.py              # CRUD + 上下文构建
│   └── summarizer.py                   # 自动摘要触发 + LLM 生成
│
├── database/                           # 数据持久化
│   ├── connection.py                   # SQLAlchemy 引擎 + Session 工厂
│   └── models.py                       # ORM: sessions/messages/summaries/documents/chunks
│
├── utils/                              # 工具
│   ├── logger.py                       # 统一日志系统 (4 通道)
│   └── helpers.py                      # MD5 / token 估算 / trace_id
│
├── logs/                               # 日志文件
├── uploads/                            # 上传 PDF/MD 存储
└── chroma_db/                          # Chroma 持久化目录
```

---

## 技术栈

| 层级 | 技术选型 |
|------|---------|
| **前端** | Streamlit |
| **LLM** | 通义千问 Qwen3-max (DashScope OpenAI 兼容 API) |
| **Embedding** | DashScope text-embedding-v4 (1024 维) |
| **向量库** | Chroma (持久化) |
| **数据库** | MySQL 8.0 + SQLAlchemy ORM + PyMySQL |
| **PDF 解析** | IBM Docling (布局分析 + OCR + 表格识别 + 公式保留) |
| **关键词检索** | rank-bm25 + jieba 分词 |
| **语义精排** | BAAI/bge-reranker-large (sentence-transformers) |
| **论文检索** | arXiv API + OpenAlex API + Semantic Scholar API |
| **日志** | Python logging + colorama + TimedRotatingFileHandler |
| **工具协议** | OpenAI Function Calling + MCP 风格 Tool Schema |

---

## 使用指南

### 1. 构建知识库

在左侧边栏上传论文 PDF。系统自动完成：
- MD5 查重
- Docling 解析 → Markdown
- 按 `##` / `###` 标题分块
- 向量化 → Chroma
- 重建 BM25 索引

### 2. 检索与提问

在聊天框输入科研问题，Agent 自动：
- 调用 `search_knowledge_base` 从知识库检索
- 基于检索结果分析回答
- 引用来源（论文名称 + 章节）

### 3. 联网搜索

当知识库无相关内容时，Agent 调用 `search_papers_online`：
- 三源并行检索
- 前端展示论文资料卡片
- 可下载 PDF 并一键加入知识库

### 4. 会话管理

- **新建会话**：点击"➕ 新建会话"
- **切换会话**：点击历史会话，自动恢复聊天记录和论文卡片
- **删除会话**：点击 🗑️

---

## 已知局限与 v2.0 规划

### 当前痛点

#### 1. RAG 检索覆盖面不足

**现象**：部分相关论文的 chunk 未被检索到。例如知识库中明确包含用 wav2vec 做前端特征提取的论文，但提问后检索结果未能召回。

**根因**：
- 当前依赖 **纯文本匹配**（BM25 关键词 + 语义向量相似度）
- 同一概念在学术文献中表述差异巨大（"wav2vec 2.0" ↔ "self-supervised speech representation" ↔ "pre-trained acoustic model"），仅靠查询改写无法穷举所有变体
- 论文之间**孤立存储**——系统不知道 A 论文和 B 论文用了同一个方法，无法通过"方法 → 论文"的反向链接发现关联

**v2.0 方向**：引入 **知识图谱 (Knowledge Graph)**，将论文实体（方法、数据集、评价指标等）及关系（`uses_method`、`evaluates_on`、`improves_upon`、`cites`）结构化存储，检索升级为**图谱结构化查询 + 语义文本检索**的混合范式。

#### 2. 联网搜索关键词精准度不足

**现象**：LLM 生成的搜索查询包含过多核心关键词（如 `"partial audio deepfake detection wav2vec self-supervised representation"`），导致检索结果中混入相关性很低的论文（如 "The Power of Generative AI: A Review of Requirements, Models, Input–Output Formats, Evaluation Metrics, and Challenges" 这类仅命中个别词的泛 AI 综述）。

**根因**：
- LLM 倾向于将用户问题的所有信息点塞入一条查询
- 学术搜索引擎对过长的多关键词查询排序效果不佳
- 联网搜索结果缺少本地二次精排（当前直接按引用量排序）

**v2.0 方向**：引入 **联网搜索结果重排序**（用 Cross-Encoder 或 LLM 对搜索返回的论文按与用户问题的相关性重新打分），以及**分步检索**策略（先搜核心术语，再在候选池中筛选）。

### v2.0 规划

- [ ] **知识图谱集成**：论文实体自动提取 + 图谱存储 + `query_paper_graph` Tool
- [ ] **联网搜索精排**：Cross-Encoder / LLM 对搜索结果按相关性重排序
- [ ] **检索质量提升**：图谱辅助 RAG，结构化关系 + 语义文本混合检索
- [ ] **综述生成增强**：基于图谱的跨论文对比 + 性能数据自动汇总
- [ ] **前端升级**：Vue.js 替换 Streamlit，支持更丰富的交互体验
- [ ] **多数据源**：中国知网、PubMed、DBLP 等
- [ ] **多用户 + 协作**：用户系统、知识库隔离、团队共享

---

## 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|---------|
| **v1.0.0** | 2026-06 | 初始发布：知识库管理（Docling PDF→MD）、混合 RAG 检索（查询改写 + BM25 + 语义 + CrossEncoder 重排）、三源联网论文搜索（arXiv + OpenAlex + S2）、MySQL 全量持久化、自动摘要、全链路日志追踪 (trace_id)、论文卡片前端 |
