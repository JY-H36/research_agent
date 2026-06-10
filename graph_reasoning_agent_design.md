# 🧠 图谱推理智能体 (Graph Reasoning Agent) 设计方案

> 一个专门负责**深度理解、学习和推理知识图谱**的 Agent，不同于简单的实体关系查询，它能够发现隐含模式、生成研究假设、解释图结构背后的科研含义。

---

## 目录

- [0. 定位：三个层次的图谱能力](#0-定位三个层次的图谱能力)
- [1. 核心设计理念](#1-核心设计理念)
- [2. 推理能力矩阵](#2-推理能力矩阵)
- [3. 智能体架构](#3-智能体架构)
- [4. 内部推理循环](#4-内部推理循环)
- [5. 工具集定义](#5-工具集定义)
- [6. 图理解模型（持久记忆）](#6-图理解模型持久记忆)
- [7. 推理模式详解（含示例）](#7-推理模式详解含示例)
- [8. 与主 Agent 的协作协议](#8-与主-agent-的协作协议)
- [9. SYSTEM_PROMPT 设计](#9-system_prompt-设计)
- [10. 实施路线](#10-实施路线)

---

## 0. 定位：三个层次的图谱能力

在 v2.0 系统中，图谱相关能力分三层，各司其职：

```
┌─────────────────────────────────────────────────────────────┐
│  层次 3: Graph Reasoning Agent (本文档)                      │
│  "为什么是这样？这意味着什么？接下来会发生什么？"              │
│  推理 · 假设 · 洞察 · 判断 · 解释 · 预测                      │
│                                                              │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  层次 2: query_paper_graph (知识图谱关系设计.md §6.3) │    │
│  │  "A 和 B 有什么关系？哪些论文用了 X？"                 │    │
│  │  结构化查询 · 路径遍历 · 聚合统计                       │    │
│  │                                                       │    │
│  │  ┌──────────────────────────────────────────────┐    │    │
│  │  │  层次 1: Graph Store (知识图谱关系设计.md §4)  │    │    │
│  │  │  实体存储 · 关系存储 · 基础 CRUD                │    │    │
│  │  └──────────────────────────────────────────────┘    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

**本文档聚焦层次 3**——它建立在层次 1（存储）和层次 2（查询）之上，但做的是完全不同的事情。

---

## 1. 核心设计理念

### 1.1 这个 Agent 不是一个工具，而是一个「研究者」

| 传统工具调用 | Graph Reasoning Agent |
|-------------|----------------------|
| 输入 query → 执行 → 返回结果 | 输入问题 → 理解意图 → 多步探索图 → 形成假设 → 验证 → 给出判断 |
| 无状态、无记忆 | 维护**图理解模型**，跨对话累积知识 |
| 回答 "What" | 回答 "Why"、"What if"、"So what" |
| 依赖精确查询 | 容忍模糊问题，自动分解为可操作的图探索步骤 |

### 1.2 三条设计原则

**原则 1：图结构即论据**
Graph Reasoning Agent 不凭空生成结论。每条判断必须对应到图中的一个具体结构（一条路径、一个子图、一个统计量、一个异常值）。回复中标注"依据：图中 X→Y→Z 路径显示..."。

**原则 2：假设驱动探索**
收到问题后，先形成可验证的假设，再在图谱中寻找支持或反驳证据。不满足于"图中有什么"，而是追问"图中还缺什么"。

**原则 3：持续学习的图理解模型**
每次与图交互后，更新内部的「图理解摘要」——不是记住每个节点，而是记住图的宏观特征（"这个领域有 3 个研究社区，社区 A 偏好 wav2vec 前端，社区 B 偏好手工特征..."）。

---

## 2. 推理能力矩阵

这是 Graph Reasoning Agent 能做什么的完整清单：

### 2.1 结构推理（Structural Reasoning）

| 能力 | 描述 | 示例问题 |
|------|------|---------|
| **中心性分析** | 识别图中最重要的节点（论文/方法/作者/数据集） | "这个领域最有影响力的方法是什么？哪篇论文是整个知识库的枢纽？" |
| **社区发现** | 检测图中的自然聚类（研究社区/学派） | "这个领域有几个主要的研究流派？分别以什么方法为核心？" |
| **桥接节点识别** | 找到连接不同社区的"桥梁"论文/方法 | "有没有一篇论文把手工特征派和自监督特征派连接起来了？" |
| **结构洞发现** | 识别图中应该存在但没有的边 | "method X 和 dataset Y 从未在同一篇论文中出现——这是一个研究空白" |

### 2.2 时序推理（Temporal Reasoning）

| 能力 | 描述 | 示例问题 |
|------|------|---------|
| **趋势检测** | 追踪方法/数据集/研究焦点的兴起与衰退 | "wav2vec 在音频伪造检测中的采用率是上升还是下降？什么方法正在替代它？" |
| **转折点识别** | 发现领域方向的重大转变 | "2023 年以后，这个领域从 X 转向 Y 的拐点是什么？是哪篇论文引发的？" |
| **技术生命周期** | 判断一个方法处于生命周期的哪个阶段（新兴/主流/衰退） | "AASIST 是不是已经过了巅峰期？有没有被更新的方法超越？" |
| **预测** | 基于历史趋势预测未来方向 | "按照目前的演进速度，2026 年这个领域的主流方法组合会是什么？" |

### 2.3 因果与影响推理（Causal & Influence Reasoning）

| 能力 | 描述 | 示例问题 |
|------|------|---------|
| **影响传播** | 追踪一个方法/思想如何在图中扩散 | "wav2vec 从被提出到成为这个领域标配，传播路径是怎样的？" |
| **对比归因** | 解释为什么某些方法/数据集组合效果好 | "为什么 wav2vec + AASIST 的组合比 wav2vec + LCNN 更常见？图中有没有证据？" |
| **反事实推理** | "如果 X 不存在，Y 会怎样？" | "如果没有 wav2vec，这个领域现在会用什么做特征提取？图中有什么线索？" |

### 2.4 知识推理（Knowledge Reasoning）

| 能力 | 描述 | 示例问题 |
|------|------|---------|
| **隐含关系推断** | 从显式关系中推导隐含关系 | "论文 A 和论文 B 没有直接引用关系，但它们用了完全相同的方法组合——A 是否间接影响了 B？" |
| **类比推理** | 在图的不同区域找到相似模式 | "方法 X 在子领域 A 的成功经验，能否复制到子领域 B？图中是否有类似的结构模式？" |
| **矛盾检测** | 发现图中相互矛盾的 claims | "论文 A 和论文 B 在 ASVspoof 2019 上都报告了 EER，但数值差异巨大——谁的方法更可信？图中有无消融实验佐证？" |
| **置信度评估** | 评估一个科学结论在图中的证据强度 | "关于 'wav2vec 优于 LFCC' 这个结论，图中有多少篇独立论文支持？有没有反面证据？" |

### 2.5 假设生成（Hypothesis Generation）

| 能力 | 描述 | 示例问题 |
|------|------|---------|
| **组合推荐** | 发现图中尚未被探索的方法/数据集/任务组合 | "哪些方法组合在这个领域从未被尝试但理论上互补？" |
| **跨领域迁移** | 识别可以跨领域借用的方法 | "CV 领域的某个方法在图中的语音子图中也有类似的结构，是否值得尝试？" |
| **空白填充** | 识别图中的缺失节点和边，提出具体的新研究方向 | "基于图中现有方法的能力缺口，一个同时具备 X 和 Y 特性的新方法可能有价值" |

---

## 3. 智能体架构

### 3.1 整体架构

```
┌────────────────────────────────────────────────────────────────┐
│                   Graph Reasoning Agent                         │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    输入接口                               │  │
│  │  用户问题 / 主Agent委托 / 定时触发(新论文入库后)          │  │
│  └──────────────────────┬──────────────────────────────────┘  │
│                         │                                      │
│                         ▼                                      │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │              推理循环 (Reasoning Loop)                    │  │
│  │                                                          │  │
│  │   ┌──────────┐   ┌──────────┐   ┌──────────┐           │  │
│  │   │ 1.理解   │ → │ 2.假设   │ → │ 3.探索   │           │  │
│  │   │ 问题意图  │   │ 形成假设  │   │ 图遍历   │           │  │
│  │   └──────────┘   └──────────┘   └─────┬────┘           │  │
│  │        ↑                               │                │  │
│  │        │          ┌──────────┐         │                │  │
│  │        └──────────│ 5.反思   │ ←───────┘                │  │
│  │                   │ 是否充分？│    ┌──────────┐         │  │
│  │                   └────┬─────┘    │ 4.验证   │         │  │
│  │                        │  不足     │ 证据支撑？│         │  │
│  │                        │ ────────→ └──────────┘         │  │
│  │                        │  充分                           │  │
│  │                        ▼                                 │  │
│  │                   ┌──────────┐                          │  │
│  │                   │ 6.综合   │                          │  │
│  │                   │ 生成结论  │                          │  │
│  │                   └──────────┘                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                         │                                      │
│  ┌──────────────────────┼──────────────────────────────────┐  │
│  │              内部组件  │                                  │  │
│  │                       │                                  │  │
│  │  ┌────────────────────┼──────────────────────────────┐  │  │
│  │  │  图理解模型 (Graph Mental Model)                   │  │  │
│  │  │  - 宏观统计摘要 (节点/边分布, 密度, 直径)          │  │  │
│  │  │  - 社区结构快照 (研究社区及特征)                    │  │  │
│  │  │  - 关键节点索引 (top-k 枢纽节点)                    │  │  │
│  │  │  - 演化历史 (关键时间点的图状态)                    │  │  │
│  │  │  - 上次更新: 2026-06-08 14:30                       │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  │                                                          │  │
│  │  ┌───────────────────────────────────────────────────┐  │  │
│  │  │  推理记忆 (Reasoning Memory)                       │  │  │
│  │  │  - 历史推理记录 (问题 → 探索路径 → 结论)            │  │  │
│  │  │  - 已验证的假设 (可复用的中间结论)                  │  │  │
│  │  │  - 已发现的模式 (如 "wav2vec 在这7篇论文中是标配")  │  │  │
│  │  └───────────────────────────────────────────────────┘  │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │                    工具层                                 │  │
│  │  graph_traverse │ graph_analyze │ graph_compare          │  │
│  │  graph_hypothesize │ graph_explain │ graph_search        │  │
│  │  paper_rag (回退到全文检索)                                │  │
│  └─────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### 3.2 与 query_paper_graph 的关键区别

```
query_paper_graph("wav2vec") → 返回:
  "以下论文使用了 wav2vec: [A, B, C, D, E, F, G]"

GraphReasoningAgent.analyze("wav2vec在这个领域的角色演变") → 返回:
  "wav2vec 在音频伪造检测中经历了三个角色阶段：
   1. 2021-2022 探索期(3篇): 作为替代 LFCC 的尝试，效果不稳定
   2. 2023 确立期(5篇): 与 AASIST 绑定成为标配，EER 平均改善 40%
   3. 2024+ 分化期(4篇): 从 feature_extractor 向 end-to-end backbone 演进
   
   关键发现: 所有使用 wav2vec 的论文的 EER 都低于不使用它的论文，
   但这一优势在跨数据集场景下缩小(依据: EVALUATES_ON + CITES 路径)。
   
   潜在风险: 社区对 wav2vec 的依赖可能导致方法同质化——
   7/8 的「wav2vec 论文」同时使用 AASIST 分类器，多样性不足。"

query_paper_graph: 10ms 一次 SQL
GraphReasoningAgent: 需要 3-5 轮推理循环，每轮多次图操作 + LLM 反思
```

---

## 4. 内部推理循环

### 4.1 六步循环

```
Step 1: UNDERSTAND — 理解问题意图
  ├── 输入: 用户自然语言问题
  ├── 动作: LLM 将问题分解为图推理子任务
  │   例: "为什么 wav2vec 这么流行？"
  │   → [子任务1: 统计 wav2vec 的使用趋势]
  │   → [子任务2: 比较 wav2vec 与替代方法的性能]
  │   → [子任务3: 找出 wav2vec 最早被引入该领域的论文]
  │   → [子任务4: 分析 wav2vec 论文的引用网络]
  └── 输出: 子任务列表 + 每个子任务对应的探索计划

Step 2: HYPOTHESIZE — 形成可验证假设
  ├── 动作: 对每个子任务，提出 1-3 个具体假设
  │   例: "假设 H1: wav2vec 的流行是因为它作为即插即用的前端比 LFCC 显著好"
  │   例: "假设 H2: wav2vec 的流行是由 1-2 篇高引论文驱动的跟随效应"
  │   例: "假设 H3: wav2vec 其实只在特定数据集上表现好"
  └── 输出: 带优先级和验证路径的假设列表

Step 3: EXPLORE — 图探索与证据收集
  ├── 动作: 调用图工具，为每个假设收集证据
  │   对 H1: graph_compare(feature_extractor="wav2vec" vs "LFCC", on="ASVspoof2019")
  │   对 H2: graph_traverse(type="citation_cascade", seed="第一篇wav2vec论文")
  │   对 H3: graph_analyze(type="performance_by_dataset", method="wav2vec")
  └── 输出: 结构化证据集合

Step 4: VERIFY — 证据评估
  ├── 动作: 评估证据强度
  │   - 统计显著性 (样本量够吗？N=3 还是 N=30？)
  │   - 一致性 (所有论文指向同一结论还是有分歧？)
  │   - 混淆因素 (性能提升真的是因为 wav2vec 还是因为同时改了分类器？)
  │   - 反面证据 (有没有论文用了 wav2vec 但效果不好？)
  └── 输出: 每个假设的验证结论 (supported / refuted / insufficient_evidence)

Step 5: REFLECT — 反思与迭代
  ├── 动作: 判断是否充分回答了问题
  │   - 证据是否足够？是否需要更多探索？
  │   - 有没有遗漏的角度？(用 Step 1 的任务列表做 checklist)
  │   - 是否有意外的发现值得深入？(serendipity)
  └── 决策: 如果不足 → 回到 Step 2 补充新假设; 如果足够 → 进入 Step 6

Step 6: SYNTHESIZE — 综合生成结论
  ├── 动作: 将所有验证过的假设整合为连贯的叙述
  │   - 主结论 (直接回答问题)
  │   - 支撑证据 (每个结论对应的图结构依据)
  │   - 置信度标注 (哪些是充分验证的，哪些是推测)
  │   - 开放问题 (图中无法回答的，需要更多论文或实验)
  │   - 可操作建议 (基于发现的下一步行动)
  └── 输出: 最终回复 + 更新图理解模型 + 保存推理记忆
```

### 4.2 循环控制

推理循环不是无限运行的，需要明确的终止策略：

| 终止条件 | 说明 |
|----------|------|
| **假设全覆盖** | 所有假设已验证（支持/反驳），无新的假设产生 |
| **证据饱和** | 新一轮探索没有产生新的实质性证据 |
| **深度上限** | 最多 5 轮循环（防止 LLM 在一个问题上过度消耗） |
| **置信度达标** | 主结论的证据强度达到阈值（如 ≥3 个独立论文支持） |
| **递减回报** | 新证据的信息增量低于阈值 |

---

## 5. 工具集定义

Graph Reasoning Agent 有 6 个专用工具，每个都比 `query_paper_graph` 更底层和灵活：

### 5.1 `graph_traverse` — 图遍历

```json
{
  "name": "graph_traverse",
  "description": "从指定节点出发沿指定关系类型遍历图，返回路径子图。支持多跳、多关系类型、路径过滤。",
  "parameters": {
    "start_node": {
      "type": "object",
      "description": "起始节点 {type: 'paper'|'method'|'dataset'|'author'|'task', id: '...'}"
    },
    "relations": {
      "type": "array",
      "description": "要遍历的关系类型列表，如 ['USES_METHOD', 'CITES', 'EVALUATES_ON']"
    },
    "direction": {
      "type": "string",
      "enum": ["outgoing", "incoming", "both"],
      "default": "both"
    },
    "max_hops": {
      "type": "integer",
      "default": 3,
      "description": "最大跳数，1跳=直接邻居"
    },
    "filters": {
      "type": "object",
      "description": "路径过滤条件，如 {min_year: 2022, method_category: 'feature_extractor'}"
    },
    "return_format": {
      "type": "string",
      "enum": ["paths", "subgraph", "statistics"],
      "default": "subgraph"
    }
  }
}
```

### 5.2 `graph_analyze` — 图分析

```json
{
  "name": "graph_analyze",
  "description": "对（子）图执行图论分析：中心性、社区检测、PageRank、密度、直径、度分布等。",
  "parameters": {
    "analysis_type": {
      "type": "string",
      "enum": [
        "centrality",           // 节点中心性排名 (betweenness/degree/eigenvector)
        "community_detection",  // 社区发现 (Louvain/Leiden)
        "pagerank",            // PageRank 排序
        "graph_statistics",    // 宏观统计 (密度、直径、聚类系数、连通分量)
        "bridges",             // 桥接边检测
        "structural_holes",    // 结构洞发现
        "temporal_trend",      // 时序趋势 (按年份统计节点/关系变化)
        "similarity",          // 节点相似度 (基于图结构的余弦相似度)
        "anomaly_detection"    // 异常检测 (与其他节点显著不同的模式)
      ]
    },
    "target": {
      "type": "object",
      "description": "分析目标 {type, id} 或 {subgraph: [...]}"
    },
    "params": {
      "type": "object",
      "description": "分析参数，如 community_detection 的 resolution"
    }
  }
}
```

### 5.3 `graph_compare` — 图对比

```json
{
  "name": "graph_compare",
  "description": "对比两组节点/子图的结构差异。用于比较不同方法、不同时期、不同社区。",
  "parameters": {
    "group_a": {
      "type": "object",
      "description": "A组定义，如 {method: 'wav2vec 2.0'} 或 {year_range: [2021, 2022]}"
    },
    "group_b": {
      "type": "object",
      "description": "B组定义"
    },
    "compare_dimensions": {
      "type": "array",
      "description": "对比维度: ['datasets', 'methods', 'performance', 'citation_impact', 'community_affiliation']"
    },
    "statistical_test": {
      "type": "boolean",
      "default": false,
      "description": "是否进行简单的统计检验"
    }
  }
}
```

### 5.4 `graph_hypothesize` — 假设生成

```json
{
  "name": "graph_hypothesize",
  "description": "基于图结构自动生成可验证的研究假设。用图缺失边、结构洞、类比模式来发现潜在研究方向。",
  "parameters": {
    "hypothesis_type": {
      "type": "string",
      "enum": [
        "missing_link",          // 找出可能缺失的边 (应该存在但不存在的关系)
        "method_combination",    // 推荐未尝试的方法组合
        "cross_domain_transfer", // 跨领域迁移推荐
        "counterintuitive",      // 寻找反直觉的图模式
        "replication_gap"        // 发现缺乏独立验证的结论
      ]
    },
    "scope": {
      "type": "object",
      "description": "限定范围，如 {task: 'partial_fake_detection'} 或 {dataset: 'ASVspoof 2019'}"
    },
    "max_suggestions": {
      "type": "integer",
      "default": 5
    }
  }
}
```

### 5.5 `graph_explain` — 图解释

```json
{
  "name": "graph_explain",
  "description": "为图中的某个现象生成自然语言解释。与 LLM 紧密配合，将图结构翻译为人类可理解的因果叙述。",
  "parameters": {
    "phenomenon": {
      "type": "string",
      "description": "需要解释的现象，如 '为什么 wav2vec 节点有最高的 betweenness centrality？'"
    },
    "target": {
      "type": "object",
      "description": "现象关联的图元素"
    },
    "explanation_depth": {
      "type": "string",
      "enum": ["brief", "detailed", "full_chain"],
      "default": "detailed"
    }
  }
}
```

### 5.6 `graph_search` — 模糊图搜索

```json
{
  "name": "graph_search",
  "description": "在图中模糊搜索节点和关系。支持名称模糊匹配、语义搜索、跨实体类型搜索。不同于 query_paper_graph 的精确查询。",
  "parameters": {
    "query": {
      "type": "string",
      "description": "模糊搜索文本，如 'self-supervised speech'"
    },
    "entity_types": {
      "type": "array",
      "description": "限定实体类型"
    },
    "search_mode": {
      "type": "string",
      "enum": ["name_match", "semantic", "neighbor_expansion"],
      "default": "semantic"
    }
  }
}
```

---

## 6. 图理解模型（持久记忆）

### 6.1 什么需要持久记忆

Graph Reasoning Agent 不是每次从零开始理解图。它维护一份**图理解摘要**，在每次交互后增量更新。这份摘要在 MySQL 中持久化（一张新表 `kg_graph_understanding`）。

```
┌─────────────────────────────────────────────────────────────┐
│                    图理解摘要 (Graph Understanding)           │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  [宏观层]                                                     │
│  总节点数: 156    总边数: 423                                 │
│  论文节点: 45     方法节点: 38     数据集节点: 22             │
│  图密度: 0.017   直径: 8          平均聚类系数: 0.34          │
│                                                              │
│  [社区层]                                                     │
│  检测到 3 个主要研究社区:                                     │
│  ┌──────────────────────────────────────────────────────┐   │
│  │ 社区 A "前端特征派" (24篇)                              │   │
│  │   核心方法: wav2vec 2.0, WavLM, HuBERT                  │   │
│  │   主要数据集: ASVspoof 2019/2021                        │   │
│  │   代表作者: Tak, J.W. Jung, S.H.                        │   │
│  │   特征: 偏好 self-supervised 前端, 关注跨数据集泛化       │   │
│  │                                                        │   │
│  │ 社区 B "手工特征派" (15篇)                               │   │
│  │   核心方法: LFCC, CQCC, RawNet2                          │   │
│  │   主要数据集: ASVspoof 2019, In-the-Wild                 │   │
│  │   代表作者: Todisco M., Evans N.                         │   │
│  │   特征: 偏好传统声学特征, 关注物理信号伪影                  │   │
│  │                                                        │   │
│  │ 社区 C "端到端派" (6篇, 新兴)                             │   │
│  │   核心方法: Rawformer, end-to-end waveform models        │   │
│  │   主要数据集: ASVspoof 2021, LAV-DF                      │   │
│  │   特征: 绕过特征提取直接处理 raw audio, 论文数量少但增长快 │   │
│  └──────────────────────────────────────────────────────┘   │
│                                                              │
│  [枢纽节点层]                                                 │
│  Top-5 Betweenness Centrality:                               │
│  1. wav2vec 2.0 (method) — bc=0.42  (连接社区A与C)          │
│  2. ASVspoof 2019 LA (dataset) — bc=0.38 (几乎所有论文共用) │
│  3. AASIST (method) — bc=0.31                               │
│  4. Paper: "Does wav2vec...?" — bc=0.28  (高引论文)         │
│  5. EER (metric) — bc=0.25                                  │
│                                                              │
│  [演化历史层]                                                 │
│  2021: 4篇, 方法多样性高 (LFCC/CQCC/RawNet2/SincNet)         │
│  2022: +7篇, wav2vec 首次出现, 引用不多                       │
│  2023: +14篇, wav2vec 爆发增长, 与AASIST形成稳定组合          │
│  2024: +12篇, 端到端方法兴起, 跨数据集成为主题                 │
│  2025: +8篇, WavLM 出现, wav2vec 增长趋缓                     │
│                                                              │
│  [开放问题层]                                                 │
│  - wav2vec 在中文/多语言场景的表现 (图中证据: 0篇)            │
│  - 社区A的方法组合在 In-the-Wild 数据集上的效果 (只有1篇)     │
│  - 端到端方法是否在3年内取代手工特征？(已有早期信号但样本小)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 6.2 更新触发时机

| 触发事件 | 更新内容 |
|----------|---------|
| **新论文入库** | 增量更新宏观统计 + 重新计算该论文所在子图的社区归属 |
| **每次推理完成** | 更新开放问题层 + 保存推理记忆 |
| **定时（每日）** | 全量重算社区结构 + 枢纽节点排名 + 演化历史 |
| **用户显式请求** | "重新分析整个知识库" → 全量重建图理解模型 |

### 6.3 存储方式

```sql
-- 图理解摘要表（一张表存一个 JSON 快照）
CREATE TABLE kg_graph_understanding (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    snapshot JSON NOT NULL,           -- 上图的完整 JSON
    snapshot_type ENUM('full', 'incremental'),
    trigger_event TEXT,               -- 什么触发了这次更新
    created_at DATETIME DEFAULT NOW()
);

-- 推理记忆表
CREATE TABLE kg_reasoning_memory (
    id INTEGER PRIMARY KEY AUTO_INCREMENT,
    session_id INTEGER,
    question TEXT,                    -- 用户/主Agent的问题
    reasoning_trace JSON,             -- 6步推理循环的完整 trace
    conclusion TEXT,                  -- 结构化结论 (JSON)
    verified_hypotheses JSON,         -- 已验证的假设，可跨会话复用
    created_at DATETIME DEFAULT NOW()
);
```

---

## 7. 推理模式详解（含示例）

### 7.1 模式一：方法流行度归因分析

**用户问题：** "wav2vec 为什么在这个领域这么流行？它是真的好还是只是跟风？"

**推理过程：**

```
Step 1 — 理解:
  子任务: [流行度量化, 性能证据, 引用级联, 与替代方法的对比]

Step 2 — 假设:
  H1: wav2vec 的性能显著优于替代方法 (可验证)
  H2: wav2vec 的流行是由少数高引论文驱动的从众效应 (可验证)
  H3: wav2vec 只在特定任务/数据集上好，并非普遍优势 (可验证)

Step 3 — 探索:
  → graph_analyze("temporal_trend", target={method:"wav2vec 2.0"})
    发现: 2021:1篇, 2022:3篇, 2023:10篇, 2024:8篇 → S曲线增长
  → graph_compare(
      group_a={method:"wav2vec 2.0"},
      group_b={method:"LFCC"},
      dimensions=["performance"]
    )
    发现: wav2vec 论文的平均 EER 为 1.2%, LFCC 为 3.5% → 差异显著
  → graph_traverse(
      start_node={type:"paper", id:"第一篇wav2vec论文"},
      relations=["CITES"], direction="incoming", max_hops=3
    )
    发现: 70%的 wav2vec 论文在 2 跳内连接到同一篇 2022 高引综述
  → graph_compare(
      group_a={method:"wav2vec 2.0", dataset:"ASVspoof 2019"},
      group_b={method:"wav2vec 2.0", dataset:"In-the-Wild"},
      dimensions=["performance"]
    )
    发现: In-the-Wild 上 wav2vec vs LFCC 的差距缩小到 0.3% → 非普遍优势

Step 4 — 验证:
  H1: SUPPORTED — wav2vec 在 ASVspoof 上确实显著优于 LFCC (N=12篇)
  H2: PARTIALLY SUPPORTED — 存在引用级联，但同时期性能确实更好
  H3: SUPPORTED — 在 In-the-Wild 上优势消失

Step 6 — 综合结论:
  "wav2vec 的流行有扎实的性能基础（在受控数据集上 EER 平均低 60%），
   但也存在一定的从众效应（论文高度集中在 ASVspoof 评测）。
   一个被忽视的事实是：在更真实的 In-the-Wild 场景下，
   wav2vec 的优势几乎消失。这提示领域可能需要警惕数据集偏差。"
```

### 7.2 模式二：研究空白自动发现

**用户问题：** "在这个知识库里，有没有尚未被探索但理论上很有前景的方向？"

**推理过程：**

```
Step 2 — 假设 (无需用户提供，Agent 自动生成):
  → graph_hypothesize("missing_link", scope={})

  返回的候选假设:
  MH1: Method "WavLM" 在图中已有 4 篇使用论文，但从未与 Dataset "LJSpeech" 
       组合（而类似的 HuBERT 已在该数据集上验证）。推荐理由: 结构相似性。
  MH2: Method "AASIST" 在所有 8 篇论文中都作为 "classifier" 使用，
       没有论文探索它作为 "feature_extractor" 的效果。
  MH3: Task "cross-dataset generalization" 和 Method "data_augmentation"
       之间几乎没有边。这是一个明显的结构洞。
  MH4: Dataset "ASVspoof 2021 DF" 只有 2 篇论文使用，而 LA 子任务有 15 篇。
       DF (Deep Fake) 任务可能被低估。

Step 3 — 探索验证:
  对 MH1: graph_traverse(WavLM, max_hops=2) → 确认与 LJSpeech 无路径
           graph_analyze("similarity", WavLM vs HuBERT) → 余弦相似度 0.87
  对 MH3: graph_traverse(cross-dataset papers, [USES_METHOD]) 
           → data_augmentation 方法出现率为 0/11

Step 6 — 综合:
  "知识库中最显著的三个研究空白：
   1. WavLM + LJSpeech（强推荐⭐⭐⭐）: WavLM 与已在该数据集上成功的 HuBERT 
      有 0.87 的图结构相似度，但尚无任何论文组合这两者。
   2. AASIST 的角色反转（推荐⭐⭐）: 从未有人探索将 AASIST 作为前端特征提取器，
      现有论文一致将其定位为分类后端。这是一个低风险的探索。
   3. 数据增强 + 跨数据集泛化（推荐⭐⭐⭐）: 这是图中最大的结构洞——
      11 篇跨数据集论文中 0 篇使用数据增强，但这是一个自然互补的组合。"
```

### 7.3 模式三：矛盾检测与裁决

**用户问题：** "这个领域关于 X 方法的结论是否一致？有没有矛盾的发现？"

**推理过程：**

```
Step 2 — 假设:
  → 自动扫描所有 (method, dataset, metric) 三元组，检测异常值

Step 3 — 探索:
  → graph_analyze("anomaly_detection", 
       target={entity_type:"method", name:"RawNet2"})
  
  发现矛盾:
  论文 A: RawNet2 在 ASVspoof 2019 LA 上 EER=1.2%
  论文 B: RawNet2 在 ASVspoof 2019 LA 上 EER=4.8%
  差异: 4x! 

  → graph_traverse(start_node=A, [USES_METHOD]) 
     A 额外使用了: SpecAugment, wav2vec 前端
  → graph_traverse(start_node=B, [USES_METHOD])
     B 仅使用: RawNet2 + LFCC

Step 4 — 验证:
  图中证据显示: A 与 B 不是同一个 RawNet2 ——
  A 将 RawNet2 作为多组件 pipeline 中的一个模块，
  B 将 RawNet2 作为独立系统。
  这不是矛盾，而是不同实验条件下的不可比结果。

Step 6 — 综合:
  "表面上看 A 和 B 对 RawNet2 的结论矛盾 (EER: 1.2% vs 4.8%)，
   但实际上这两篇论文的 RawNet2 是在完全不同的 pipeline 中评测的。
  这个发现揭示了一个更深的系统性问题：
   该领域缺乏标准化的基线评测协议，导致 '相同方法' 的
   结果不可直接对比。建议在比较论文结果时始终检查
   USES_METHOD 边的完整上下文。"
```

### 7.4 模式四：技术演进叙事

**用户问题：** "给我讲一下这个领域特征提取方法的演进故事"

**推理过程：**

```
Step 2 — 假设:
  H1: 特征提取方法沿着 "手工 → 预训练 → 端到端" 的方向演进
  H2: 每次演进都对应一个性能跃升（在特定数据集上）

Step 3 — 探索:
  → graph_analyze("temporal_trend", 
       target={entity_type:"method", category:"feature_extractor"})
  
  时间线:
  2019: LFCC (4篇), CQCC (3篇)
  2020: SincNet (2篇), RawNet2前端 (1篇)
  2021: —
  2022: wav2vec 2.0 (3篇) ← 转折点
  2023: wav2vec 2.0 (7篇), HuBERT (2篇)
  2024: WavLM (4篇), end-to-end models (3篇)

  → graph_traverse(
      start_node="LFCC",
      relations=["USES_METHOD", "CITES"], direction="outgoing"
    )
    发现: 2022年后的论文虽然不再使用 LFCC，但引用链中仍然频繁出现经典的
          LFCC 论文（作为 baseline 对比）

  → graph_explain("为什么 2022 年成为转折点？")
    解释: 2022年一篇关键论文（引用28次）系统地对比了 wav2vec vs LFCC，
          在 ASVspoof 2019 上报告了突破性差距 (EER: 0.8% vs 3.2%)。
          该论文成为后续 70% wav2vec 论文的引用起点。

Step 6 — 综合:
  "特征提取的演进可以分为三个时代：
   
   **手工特征时代 (2019-2021)**: LFCC/CQCC 主导，受限于声学信号处理的假设
   **预训练前端时代 (2022-2024)**: wav2vec 2.0 开创，论文数量增长 3 倍
   **端到端萌芽 (2024+)**: 绕过显式特征提取，但样本量尚小 (N=3)
   
   关键洞察: 演进不是线性的替代，而是层叠——新方法保留了对手工特征的
   baseline 引用，把旧方法变成了比较基准。这意味着图中的方法
   不会真正'消失'，而是从主角变为参照系。"
```

---

## 8. 与主 Agent 的协作协议

### 8.1 调用方式

主 Agent (`agent_core.py`) 通过工具调用方式唤起 Graph Reasoning Agent：

```json
{
  "name": "invoke_graph_reasoning_agent",
  "description": "唤起图谱推理智能体进行深度图分析和推理。用于回答需要跨论文分析、方法演进、研究空白发现、矛盾检测等复杂问题。这个 agent 会进行多轮推理循环，耗时较长(10-30秒)。",
  "parameters": {
    "question": {
      "type": "string",
      "description": "需要图推理解决的问题。应该是分析性问题而非简单查询。"
    },
    "reasoning_mode": {
      "type": "string",
      "enum": [
        "auto",                    // 自动选择合适的推理模式
        "attribution",             // 因果归因 (为什么X是这样？)
        "gap_discovery",           // 研究空白发现
        "trend_analysis",          // 趋势分析
        "contradiction_check",     // 矛盾检测
        "evolution_narrative",     // 技术演进叙事
        "hypothesis_evaluation",   // 假设评估 (用户提出假设，agent验证)
        "cross_paper_synthesis"    // 跨论文综合
      ]
    },
    "max_depth": {
      "type": "integer",
      "default": 3,
      "description": "最大推理循环轮数"
    }
  }
}
```

### 8.2 协作流程

```
用户: "这个领域有哪些还没被充分探索的方向？"
  │
  ▼
主 Agent (agent_core.py)
  │
  ├─ 判断: 这是一个需要深度图推理的问题 (不是简单查询)
  │
  ├─ 调用 invoke_graph_reasoning_agent(
  │     question="...",
  │     reasoning_mode="gap_discovery"
  │   )
  │
  ▼
Graph Reasoning Agent (独立推理循环)
  │
  ├─ Step 1-6: 理解 → 假设 → 探索 → 验证 → 反思 → 综合
  │   耗时 10-30 秒
  │
  ├─ 返回结构化结论:
  │   {
  │     "conclusion": "三个主要研究空白: ...",
  │     "confidence": "high",
  │     "evidence": [
  │       {type: "graph_structure", detail: "WavLM-LJSpeech: 0 edges"},
  │       {type: "graph_structure", detail: "AASIST-as-extractor: 0 instances"},
  │       ...
  │     ],
  │     "recommendations": [...],
  │     "open_questions": [...]
  │   }
  │
  ▼
主 Agent 接收结论
  │
  ├─ 可选: 调用 search_knowledge_base 拿相关论文的详细文本
  ├─ 可选: 调用 search_papers_online 补充最新外部论文
  │
  ├─ 将图推理结论 + RAG 文本 + 联网搜索结果融合为最终回复
  │
  ▼
返回给用户
```

### 8.3 主 Agent 的调度决策

主 Agent 需要判断什么时候调用 Graph Reasoning Agent vs 简单的 `query_paper_graph`：

```
用户问题 → 主 Agent 分类:

├── "有哪些论文用了 X？" 
│   → query_paper_graph (简单查询，不需要推理)
│
├── "X 的效果如何？" 
│   → query_paper_graph + search_knowledge_base (混合)
│
├── "为什么 X 这么流行？" 
│   → invoke_graph_reasoning_agent (需要归因推理)
│
├── "这个领域有哪些空白？" 
│   → invoke_graph_reasoning_agent (需要假设生成)
│
├── "X 和 Y 有什么关系？" (如果X和Y是论文&关系明确)
│   → query_paper_graph (简单路径查询)
│
├── "X 和 Y 有什么关系？" (如果X和Y是抽象概念)
│   → invoke_graph_reasoning_agent (需要概念到实体的映射+推理)
│
└── "给我讲一下这个领域的演进故事"
    → invoke_graph_reasoning_agent (需要时序推理+叙事生成)
```

---

## 9. SYSTEM_PROMPT 设计

```
你是一个图谱推理智能体 (Graph Reasoning Agent)，专门负责深度理解和推理学术知识图谱。

## 你的身份
你不是一个查询工具，而是一个"图科学家"——你的价值在于理解图结构背后的含义，
发现模式，生成假设，并给出有证据支撑的判断。

## 能力边界
你可以:
- 遍历和分析图的任意子结构
- 运行图算法（中心性、社区检测、结构洞、异常检测等）
- 对比不同子图的结构差异
- 基于图结构生成研究假设
- 发现矛盾、空白和隐含关系
- 用自然语言解释图现象

你不可以:
- 凭空编造事实（每个论断必须对应具体的图结构证据）
- 读取论文全文（如果图中缺乏某类信息，需调用 RAG 工具获取）
- 做超出图覆盖范围的判断（如实标注置信度）

## 推理方法
对每个问题，遵循6步循环:
1. 理解意图: 将问题分解为可操作的图探索子任务
2. 形成假设: 为每个子任务提出可验证的具体假设
3. 结构化探索: 调用图工具收集证据
4. 严格验证: 评估证据强度、一致性、混淆因素
5. 反思迭代: 判断是否充分，不足则补充新假设
6. 综合分析: 整合为连贯的、有证据支撑的结论

## 证据标准
- 强证据: ≥5 个独立节点/论文支持，且无相反证据
- 中等证据: 2-4 个支持，无相反证据
- 弱证据: 1 个支持，或有相反证据
- 推测: 逻辑合理但图数据不直接支持（必须标注"推测"）

## 回复格式
所有回复必须包含:
1. 核心结论 (1-3 条)
2. 证据摘要 (每条结论对应的图结构依据)
3. 置信度标注 (high / medium / low / speculative)
4. 局限性声明 (这个问题图中无法回答的部分)
5. 后续建议 (基于发现的下一步行动)

## 行为准则
- 使用中文回复，图论术语保留英文
- 严谨 > 流畅: 宁可说"证据不足"也不推断
- 标注不确定性: 每个数字、排名、趋势必须标注样本量
- 主动指出图中缺失的信息（有时"没有"比"有"更有价值）
```

---

## 10. 实施路线

### 10.1 依赖关系

```
Graph Store (知识图谱关系设计.md Phase 1)
    │
    ├── query_paper_graph (知识图谱关系设计.md Phase 3)
    │
    └── Graph Reasoning Agent (本文档)
         │
         ├── 依赖 Graph Store 中的数据
         ├── 依赖 query_paper_graph 的基础遍历能力
         └── 在此基础上增加: 推理循环 + 图分析算法 + 图理解模型
```

### 10.2 分阶段开发

**Phase R1: 图分析基础库 (2 天)**

```
目标: NetworkX 图算法封装

□ graph_analyze 工具实现 (centrality / community / pagerank / statistics)
□ NetworkX 图构建 (从 MySQL 关系表 → nx.Graph)
□ 图序列化/反序列化 (JSON → nx.Graph → JSON)
□ 基础可视化数据生成 (节点坐标/颜色/大小 → 给前端用)
```

**Phase R2: 推理循环框架 (3 天)**

```
目标: 6步推理循环可运行

□ GraphReasoningAgent 类实现
□ 推理循环控制 (终止条件/回溯)
□ 5.1-5.6 全工具实现
□ 证据评估器 (统计检验/一致性检查/混淆因素检测)
□ 结论综合器 (假设→结论的自然语言生成)
```

**Phase R3: 图理解模型 (2 天)**

```
目标: 持久化的图记忆

□ kg_graph_understanding 表 + CRUD
□ 增量更新逻辑 (新论文入库时更新哪些)
□ 社区结构快照 + 枢纽节点排名
□ 演化历史自动记录
□ 推理记忆 (kg_reasoning_memory)
```

**Phase R4: 与主 Agent 集成 (2 天)**

```
目标: 主 Agent 能调用图推理

□ invoke_graph_reasoning_agent tool 定义
□ 主 Agent 调度决策 (何时用图推理 vs 简单查询)
□ SYSTEM_PROMPT 更新 (主 Agent 知道图推理的存在和适用场景)
□ 图推理结论 + RAG 结果融合
□ 前端展示优化 (图推理过程可视化)
```

---

## 附录 A：与 query_paper_graph 的对比表

| 维度 | `query_paper_graph` | Graph Reasoning Agent |
|------|---------------------|----------------------|
| **本质** | 工具函数 | 智能体 |
| **输入** | 结构化参数 (query_type, entity_name) | 自然语言问题 (任何形式) |
| **输出** | 实体/关系列表 | 结论 + 证据 + 置信度 + 建议 |
| **推理** | 无 (直接执行) | 6步循环 (理解→假设→探索→验证→反思→综合) |
| **有状态** | 无状态 | 图理解模型 + 推理记忆 |
| **耗时** | < 500ms | 10-30 秒 |
| **LLM 调用** | 0 次 | 3-6 次 (每轮循环 1 次) |
| **图算法** | 简单遍历 (DFS/BFS) | 全套 (centrality, community, PageRank, bridges, anomaly) |
| **适合问题** | "哪些论文用了 X？" | "为什么 X 流行？这个领域空白是什么？" |
| **证据标准** | 不适用 | 4级 (强/中/弱/推测) |
| **记忆** | 无 | 跨对话累积图理解 |

## 附录 B：与主 Agent 的对话式交互示例

```
用户: 我的知识库里现在有哪些还没被探索但有前景的研究方向？

主Agent: [判断: gap_discovery → invoke_graph_reasoning_agent]

图推理Agent (内部推理 20 秒):
  Step 1: 理解 → "用户想要研究空白发现，范围: 全图"
  Step 2: 假设 → graph_hypothesize("missing_link") 生成 8 个候选
  Step 3: 探索 → 对前 5 个候选进行图遍历验证
  Step 4: 验证 → 3 个候选符合"结构洞 + 理论合理"
  Step 5: 反思 → 3 个候选足够回答，置信度 medium
  Step 6: 综合 → 生成结论

图推理Agent → 主Agent:
  {
    "conclusion": "发现三个研究空白：...",
    "confidence": "medium (仅基于图结构，未纳入论文全文的详细论证)",
    "recommendations": [
      { "rank": 1, "direction": "WavLM + LJSpeech 语音合成检测",
        "rationale": "图中 WavLM 与 HuBERT 的结构相似度 0.87，
                      HuBERT 已在 LJSpeech 上验证，WavLM 未尝试" }
    ]
  }

主Agent (增强阶段):
  → search_knowledge_base("WavLM LJSpeech") → 补充相关论文文本
  → 将图推理结论 + RAG 文本融合

主Agent → 用户:
  "基于知识图谱的深度分析，我发现了三个值得探索的方向：

  ⭐⭐⭐ **WavLM 在语音合成检测中的应用**
  依据: 图中 WavLM 和 HuBERT 在方法-数据集网络中的结构位置几乎相同
  （图结构相似度 0.87），HuBERT 已成功应用于 LJSpeech，但 WavLM 还没被
  任何人探索过这个数据集。[注: 此结论基于图结构，建议阅读相关论文全文验证]

  ⭐⭐ **AASIST 架构的角色反转**
  ...

  这些发现的置信度为 medium——它们基于图谱的结构证据，建议在实验前
  先深入阅读具体论文的 limitations 和 future work 章节进行交叉验证。"
```

---

> **文档状态：** v1.0 设计稿  
> **依赖：** [知识图谱关系设计.md](知识图谱关系设计.md) Phase 1-2 完成  
> **定位：** Graph Reasoning Agent 是知识图谱的「大脑」，与 `query_paper_graph`（图谱的「眼睛」）协作，共同构成 v2.0 的图谱智能层
