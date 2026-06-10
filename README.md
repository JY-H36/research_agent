# 🔬 科研灵感助手 (Research Inspiration Agent)

> 基于 LLM + RAG + Knowledge Graph 的科研辅助 Agent，帮助研究人员进行文献知识库管理、论文检索、知识图谱构建和科研方案讨论。

[![Version](https://img.shields.io/badge/version-2.0.0-blue)](https://github.com)
[![Python](https://img.shields.io/badge/python-3.10+-green)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/framework-Streamlit-red)](https://streamlit.io/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## 📖 目录

- [项目简介](#项目简介)
- [v1.0 痛点与 v2.0 改进](#v10-痛点与-v20-改进)
- [核心功能](#核心功能)
- [系统架构](#系统架构)
- [知识图谱设计](#知识图谱设计)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [技术栈](#技术栈)
- [使用指南](#使用指南)
- [版本历史](#版本历史)

---

## 项目简介

**科研灵感助手**是一个面向学术研究人员的 AI Agent。以**研究方向**为锚点，提供从**文献入库 → 实体提取 → 知识图谱构建 → 图谱查询 → 联网搜索 → 综述分析 → 方案讨论**的全流程辅助。

核心定位：**知识库论文分析为主，联网搜索为辅**，帮助研究人员系统性地理解某一方向的研究现状、发现研究空白、验证科研想法。

### 典型使用场景

```
用户: "哪些论文用了 wav2vec 2.0 做特征提取？在 ASVspoof 2019 LA 上表现如何？"

Agent → query_paper_graph("使用了 wav2vec 2.0 且在 ASVspoof 2019 上评测的论文")
      → 知识图谱结构化查询（SQL JOIN + NetworkX 图遍历）
      → 返回 3 篇论文，含特征提取器变体、网络框架、损失函数、实验数据集及性能指标
      → 按数据集汇总性能对比表
      → 标注来源（论文名称 + 实体关系路径）
```

---

## v1.0 痛点与 v2.0 改进

### 痛点 1：同义近义词理解不充分

**现象**：知识库中存在 "wav2vec 2.0 Base" 和 "Wav2Vec 2.0" 两个相同方法的实体，系统无法识别它们本质上是同一个东西。原因在于 v1.0 的知识图谱缺乏语义层面的消歧能力，仅依赖字符串相似度匹配。

**v2.0 解决方案 —— 三级级联实体消歧**：

```
入库新实体
    │
    ▼
L1: 精确名称匹配 (DB 查询, O(1))
    │  命中 → 直接复用已有实体
    │  未命中 ↓
L2: 模糊规则匹配 (子串包含、token 重叠 >75%、别名匹配、归一化等号比较)
    │  命中 → 复用，并补充描述
    │  未命中 ↓
L3: LLM 语义判断 (结合实体 name + description + 上下文 → 判定是否应为同一实体)
    │  确认重复 → 复用
    │  判定不同 → 创建新实体
```

关键提升：L3 层利用 LLM 的语义理解能力，可以正确判断 "wav2vec 2.0 Base" ≈ "Wav2Vec 2.0"、"ADD 2023" ≈ "Audio Deepfake Detection 2023" 这类字符串距离很远但语义等价的实体对。

### 痛点 2：知识图谱实体粒度不够精细

**现象**：v1.0 将所有"方法"统归为 `Method` 实体的不同 `category`（如 `feature_extractor`、`network_architecture`、`loss_function`）。这导致：

- 无法区分"这篇论文提出了什么方法"和"这篇论文用了什么方法"
- 特征提取器、网络框架、损失函数本质上是不同类型的一等公民，不应混在同一张表里
- 查询"哪些论文用了 ResNet 作为骨架网络"需要额外的 category 过滤，效率低下

**v2.0 解决方案 —— 实体体系重构**：

| 维度 | v1.0 | v2.0 |
|------|------|------|
| **论文提出的方法** | `Method` (category=framework) | 直接存为 `Paper` 的 `method_name` + `method_summary` 属性 |
| **特征提取器** | `Method` (category=feature_extractor) | 独立实体 `FeatureExtractor`（通过 `Method` 表 + category 区分） |
| **网络框架** | `Method` (category=network_architecture) | 独立实体 `NetworkArchitecture`（通过 `Method` 表 + category 区分） |
| **损失函数** | `Method` (category=loss_function) | 独立实体 `LossFunction`（通过 `Method` 表 + category 区分） |
| **数据集** | `Dataset` | `Dataset`（不变，但增加了 LLM 语义消歧） |
| **作者/发表地/指标** | 独立实体 | 简化为 `Paper` 的 JSON 属性（降低图复杂度） |

核心理念变化：**论文本身即是其提出的方法**——`Paper` 实体直接承载 `method_name` 和 `method_summary`，无需额外创建 `Method(category=framework)` 实体。而特征提取器、网络框架、损失函数作为三个独立的实体类型，各自拥有独立的消歧和查询路径。

---

## 核心功能

### 1. 📚 知识库管理

| 功能 | v1.0 | v2.0 增强 |
|------|:---:|------|
| **PDF 智能解析** | ✅ | IBM Docling → Markdown（不变） |
| **MD5 去重** | ✅ | 不变 |
| **章节分块** | ✅ | 不变 |
| **向量化存储** | ✅ | 不变 |
| **知识图谱提取** | ❌ | 🆕 LLM Agent 自动提取论文结构化信息（方法、数据集、实验结果） |
| **知识库管理页** | ❌ | 🆕 Streamlit 独立页面：论文列表查看、详情展开、删除管理 |
| **Excel 一键导出** | ❌ | 🆕 全部论文信息导出为 Excel（标题、关键词、方法、数据集等 11 列） |

### 2. 🔍 混合 RAG 检索（三层增强）

```
用户查询
    │
    ▼
L1: LLM 查询改写 → 5 个同义变体
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

### 3. 🧠 知识图谱（v2.0 核心新增）

#### 3a. LLM Agent 论文信息提取

上传论文 PDF 后，`paper_extraction_agent` 自动从完整 Markdown 中一次性提取：

```json
{
  "title": "论文标题",
  "keywords": [...],
  "proposed_method": { "name": "本文提出的方法", "summary": "3-5 句话概述" },
  "feature_extractors": [{ "name": "...", "description": "...", "variant": "..." }],
  "network_architectures": [{ "name": "...", "description": "..." }],
  "loss_functions": [{ "name": "...", "description": "..." }],
  "experiment_datasets": [{ "name": "...", "description": "...", "role": "eval|train|both" }],
  "experiment_results": [{ "dataset": "...", "results": [{ "model": "...", "metrics": {...} }] }]
}
```

#### 3b. 实体消歧（三级级联）

入库时实时消歧：L1 精确匹配 → L2 模糊规则 → L3 LLM 语义判断。避免事后大批量归一化的性能和准确度问题。

#### 3c. 图谱结构化查询

Agent 可调用 `query_paper_graph` 工具执行 10 种图谱查询：

| 查询类型 | 示例问题 |
|----------|---------|
| 哪些论文用了某方法 | "哪些论文用了 wav2vec 2.0？" |
| 某方法被哪些论文使用 | "AASIST 方法被哪些论文使用？" |
| 某数据集上的评测论文 | "ASVspoof 2019 LA 上评测了哪些论文？" |
| 数据集上的性能排名 | "ASVspoof 2019 LA 上 EER 最低的论文？" |
| 某作者的论文 | "XXX 发表了哪些论文？" |
| 某研究任务的论文 | "部分伪造检测方向有哪些论文？" |
| 论文详情 | "论文 XXX 的完整信息" |
| 方法共现分析 | "哪些方法经常与 wav2vec 2.0 组合使用？" |
| 相关论文发现 | "与这篇论文最相关的其他论文" |
| 图谱统计 | "知识库中有多少论文/方法/数据集？" |

#### 3d. 知识库管理页面（Streamlit）

- 📋 论文列表（可搜索）
- 🔍 论文详情展开（提出方法 → 特征提取器 → 网络框架 → 损失函数 → 数据集 → 实验结果）
- 🗑️ 论文删除（含关联实体级联清理）
- 📥 Excel 一键导出

### 4. 🌐 联网论文搜索

| 数据源 | 状态 | 说明 |
|--------|:---:|------|
| **OpenAlex** | ✅ 主力 | 免费开放学术索引，无速率限制 |
| **arXiv** | ✅ 可用 | 双路径请求（直连 + 代理） |
| **Semantic Scholar** | ⚠️ 限速 | 120s 冷却计时，可用时自动加入 |

- **三源并行**：ThreadPoolExecutor 同时检索
- **论文卡片**：前端展示标题、作者、年份、摘要、PDF 下载
- **一键入库**：下载 PDF → 自动解析 → 知识图谱提取

### 5. 💬 会话管理

- **MySQL 全量持久化**：消息、摘要、文档、分块，重启不丢失
- **自动摘要**：消息数 ≥ 20 条自动触发 LLM 生成四段式摘要
- **会话恢复**：切换历史会话自动恢复聊天记录和论文卡片

### 6. 📋 全链路日志追踪

- 四通道日志：控制台彩色 + 按天滚动 + 错误日志 + 内存环形缓冲
- `trace_id` 全程携带，前端面板实时筛选

---

## 系统架构

```
┌─────────────────────────────────────────────────────────┐
│                  Streamlit 前端 (app.py)                   │
│     💬 对话 · 📚 知识库管理 · 会话管理 · 日志面板           │
├─────────────────────────────────────────────────────────┤
│              Agent 核心 (agent_core.py)                    │
│       ReAct 模式 · 上下文编排 · Tool 路由                   │
├───────────────────────┬─────────────────────────────────┤
│   tools.py             │   paper_search_mcp.py (MCP)      │
│   search_knowledge_base│   search_papers_online           │
│   query_paper_graph 🆕 │   三源并行检索                    │
├───────────────────────┴─────────────────────────────────┤
│                 LLM 服务 (llm_service.py)                  │
│              Qwen3-max (DashScope API)                     │
├─────────────────────────────────────────────────────────┤
│                    知识层                                  │
│  ┌───────────────────┐  ┌──────────────────────────────┐ │
│  │ RAG 检索           │  │ 知识图谱 🆕                    │ │
│  │ retriever          │  │ paper_extraction_agent       │ │
│  │ reranker           │  │ entity_resolver (L1/L2/L3)   │ │
│  │ query_rewriter     │  │ entity_normalizer            │ │
│  │ vector_store       │  │ graph_store (NetworkX+MySQL) │ │
│  └───────────────────┘  │ graph_query (10 种查询)       │ │
│                         └──────────────────────────────┘ │
├─────────────────────────────────────────────────────────┤
│                    数据层                                  │
│  MySQL (sessions · messages · summaries · documents      │
│         · chunks · kg_papers · kg_methods ·              │
│         kg_datasets · kg_metrics · kg_tasks ·            │
│         9 张关系表)  +  Chroma 向量库                      │
└─────────────────────────────────────────────────────────┘
```

---

## 知识图谱设计

### 实体（8 张表）

| 实体 | 表名 | 核心字段 | 说明 |
|------|------|---------|------|
| **Paper** | `kg_papers` | title, authors(JSON), keywords(JSON), method_name, method_summary, year, venue_name | 论文即方法，直接承载 proposed method |
| **Method** | `kg_methods` | name, aliases(JSON), category, description | 特征提取器 / 网络框架 / 损失函数 |
| **Dataset** | `kg_datasets` | name, description, domain, task | 实验数据集 |
| **Metric** | `kg_metrics` | name, full_name, direction, unit | 评价指标定义 |
| **Task** | `kg_tasks` | name, description, parent_task_id, level | 研究任务树 |
| **Author** | `kg_authors` | name, affiliation, orcid | 作者（备查，不作为主要查询路径） |
| **Venue** | `kg_venues` | name, abbreviation, type, rank | 发表地（备查） |

### 关系（9 张表）

```
Paper ──USES_METHOD──→ Method     (特征提取器/网络框架/损失函数)
Paper ──EVALUATES_ON──→ Dataset   (评测 + metrics JSON)
Paper ──TRAINS_ON─────→ Dataset   (训练)
Paper ──BELONGS_TO────→ Task      (研究任务)
Paper ──REPORTS_METRIC→ Metric    (指标报告)
Paper ──CITES─────────→ Paper     (引用)
Method──IMPROVES_UPON─→ Method    (方法改进)
Paper ──PUBLISHED_IN──→ Venue     (发表)
Paper ──WRITTEN_BY────→ Author    (作者)
```

> **设计原则**：Paper 是图谱核心节点。其他实体均通过关系表围绕 Paper 组织。简化了 v1.0 中 Author/Venue/Metric 的独立实体路径，改为优先使用 Paper 的 JSON 属性存储。

---

## 快速开始

### 环境要求

- Python 3.10+
- MySQL 8.0+
- [DashScope API Key](https://dashscope.aliyun.com/)（通义千问 + Embedding）
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
├── app.py                              # Streamlit 前端入口（💬 对话 / 📚 知识库管理）
├── config.py                           # 全局配置（含知识图谱开关）
├── requirements.txt
├── .env.example
├── README.md
│
├── agent/                              # Agent 核心
│   ├── agent_core.py                   # ReAct 主循环 (SYSTEM_PROMPT + Tool 路由)
│   ├── llm_service.py                  # Qwen3-max (OpenAI 兼容协议 + 重试)
│   ├── tools.py                        # search_knowledge_base + query_paper_graph
│   ├── paper_search_mcp.py             # search_papers_online (MCP 风格)
│   └── middleware.py                   # 工具调用追踪 + 日志
│
├── knowledge_graph/ 🆕                  # 知识图谱模块（v2.0 核心新增）
│   ├── models.py                       # 8 张实体表 + 9 张关系表 ORM
│   ├── paper_extraction_agent.py       # LLM Agent 论文信息提取
│   ├── entity_extractor.py             # 提取 + 入图流水线（一键调用）
│   ├── entity_resolver.py              # 三级级联实体消歧 (L1/L2/L3)
│   ├── entity_normalizer.py            # 批量实体归一化
│   ├── entity_normalization_agent.py   # LLM 归一化 Agent
│   ├── graph_store.py                  # MySQL + NetworkX 图存储 CRUD
│   ├── graph_query.py                  # 10 种图谱结构化查询
│   ├── graph_tool.py                   # query_paper_graph Agent Tool
│   └── visualization.py                # 图谱可视化
│
├── knowledge_base/                     # 知识库与 RAG 检索
│   ├── document_processor.py           # Docling PDF→MD + 章节分块
│   ├── embedding_service.py            # DashScope text-embedding-v4
│   ├── vector_store.py                 # Chroma 向量存储
│   ├── retriever.py                    # BM25 + 语义 + RRF 混合检索
│   ├── reranker.py                     # Cross-Encoder 重排序
│   ├── query_rewriter.py               # LLM 查询改写
│   └── paper_search.py                 # 三源论文检索
│
├── session/                            # 会话与摘要
│   ├── session_manager.py              # CRUD + 上下文构建
│   └── summarizer.py                   # 自动摘要生成
│
├── database/                           # 数据持久化
│   ├── connection.py                   # SQLAlchemy 引擎 + Session 工厂
│   └── models.py                       # ORM: sessions/messages/summaries/documents/chunks
│
├── utils/                              # 工具
│   ├── logger.py                       # 四通道日志系统
│   └── helpers.py                      # MD5 / token 估算 / trace_id
│
├── logs/                               # 日志文件
├── uploads/                            # 上传文件存储
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
| **关系数据库** | MySQL 8.0 + SQLAlchemy ORM + PyMySQL |
| **图计算** | NetworkX (MultiDiGraph 内存图缓存) |
| **PDF 解析** | IBM Docling (布局分析 + OCR + 表格识别) |
| **关键词检索** | rank-bm25 + jieba 分词 |
| **语义精排** | BAAI/bge-reranker-large (sentence-transformers) |
| **Excel 导出** | openpyxl |
| **论文检索** | arXiv API + OpenAlex API + Semantic Scholar API |
| **日志** | Python logging + colorama + TimedRotatingFileHandler |

---

## 使用指南

### 1. 上传论文 → 自动构建知识图谱

在左侧边栏上传论文 PDF。系统自动完成：

```
PDF 上传
  → MD5 查重
  → Docling 解析 → Markdown
  → 章节分块 → 向量化 → Chroma
  → LLM Agent 提取结构化信息
  → 三级级联实体消歧
  → 写入知识图谱（Paper + Methods + Datasets + 关系）
  → 自动归一化
  → 前端展示提取结果
```

### 2. 知识库管理

切换到 **📚 知识库管理** 标签页：

- 查看所有已入库论文（支持搜索）
- 点击论文查看完整信息（提出方法、特征提取器、网络框架、损失函数、实验数据集、实验结果）
- 删除论文（级联清理关联实体、文档、向量数据）
- 一键导出全部论文信息为 Excel

### 3. 图谱查询

在 **💬 对话** 标签页的聊天框中，Agent 自动判断是否需要调用 `query_paper_graph`：

- "哪些论文用了 wav2vec 2.0？"
- "ASVspoof 2019 LA 上评测了哪些论文？"
- "部分伪造检测方向有哪些论文？"

### 4. 联网搜索

当知识库无相关内容时，Agent 自动调用 `search_papers_online`：

- 三源并行检索（arXiv + OpenAlex + Semantic Scholar）
- 前端展示论文卡片，可下载 PDF 并一键入库

### 5. 会话管理

- **新建会话**：点击 "➕ 新建会话"
- **切换会话**：点击历史会话，自动恢复聊天记录
- **删除会话**：点击 🗑️

---

## 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|---------|
| **v2.0** | 2026-06 | 🆕 知识图谱集成：LLM Agent 论文信息提取、三级级联实体消歧 (L1 精确 + L2 模糊 + L3 LLM 语义)、实体体系重构（特征提取器/网络框架/损失函数独立实体、论文即方法）、知识库管理页面、Excel 一键导出、`query_paper_graph` 工具（10 种图谱查询）、NetworkX 图缓存、实体批量归一化 |
| **v1.0** | 2026-06 | 初始发布：知识库管理（Docling PDF→MD）、混合 RAG 检索（查询改写 + BM25 + 语义 + CrossEncoder 重排）、三源联网论文搜索（arXiv + OpenAlex + S2）、MySQL 全量持久化、自动摘要、全链路日志追踪 (trace_id)、论文卡片前端 |
