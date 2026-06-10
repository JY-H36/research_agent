"""
论文信息提取 Agent（v2 精简版）
- 聚焦论文提出的方法概述（非子组件罗列）
- 特征提取器 + 损失函数分别提取
- 实验数据集（仅实验章节实际使用的，非文中提及的全部）
- 实验表格完整数据（含 baseline）
"""
import json
import re
from typing import Dict, List, Optional

from utils.logger import get_logger
from config import KG_PAPER_EXTRACTION_MAX_CHARS

logger = get_logger(__name__)

# ============================================================
# 论文提取 Prompt（v2 精简版）
# ============================================================

PAPER_EXTRACTION_SYSTEM = """你是一个学术论文信息提取专家。你的任务是从论文的 Markdown 文本中精确提取结构化信息。

【核心原则】
1. 只提取论文中明确提到的信息，不要编造
2. 实体名称使用论文中的原始表述
3. 聚焦论文自身提出的方法，不要罗列无关子组件
4. 实验数据表需要原样提取所有数值（含 baseline）
5. 返回合法 JSON"""

PAPER_EXTRACTION_USER_TEMPLATE = """请从以下学术论文中提取结构化信息。

【论文 Markdown 内容】
{paper_markdown}

请提取以下信息，严格按 JSON 格式返回：

{{
  "title": "论文完整标题",
  "keywords": ["关键词1", "关键词2"],
  "authors": ["作者1", "作者2"],
  "year": 2024,
  "venue": "会议/期刊名（如 ICASSP 2024）",

  "proposed_method": {{
    "name": "论文提出的方法/框架的名称",
    "summary": "对该方法的概述（3-5句话），总结方法章节的核心思路和整体架构。参考论文 Methodology 章节的 overview 或 introduction 部分。不要逐一列举内部子模块名称（如 OGD、GRL、RSDM 等细粒度组件），而是用连贯的文字描述整体方案",
    "category": "framework|classifier|feature_extractor|loss_function|data_augmentation"
  }},

  "feature_extractors": [
    {{
      "name": "前端特征提取方法名称（如 wav2vec 2.0、LFCC、MFCC、CQCC、1D-CNN）",
      "description": "简要描述：如何从原始数据（如音频）中提取特征",
      "variant": "使用的变体（如 Base、Large），没有则留空"
    }}
  ],

  "network_architectures": [
    {{
      "name": "神经网络架构名称（如 Transformer、Conformer、BiLSTM、Mamba、TCN、ResNet）",
      "description": "该架构在论文中扮演的角色（1句话），如 '作为 backbone 序列建模'、'用于时序特征提取'"
    }}
  ],

  "loss_functions": [
    {{
      "name": "损失函数名称（如 AAM-Softmax、OC-Softmax、BCE、MSE）",
      "description": "该损失函数的作用（1句话）"
    }}
  ],

  "experiment_datasets": [
    {{
      "name": "数据集标准名称（如 ASVspoof 2019 LA、VCTK）",
      "description": "数据集简要描述（1-2句话）",
      "role": "eval|train|both",
      "task": "在该数据集上执行的任务"
    }}
  ],

  "experiment_results": [
    {{
      "dataset": "数据集名称（与 experiment_datasets 中的 name 对应）",
      "condition": "评测条件（如 clean、noisy、seen、unseen），没有则留空",
      "results": [
        {{
          "method_name": "方法名称（论文自身的方法或对比的 baseline）",
          "is_baseline": false,
          "metrics": {{
            "EER": 2.15,
            "min t-DCF": 0.0258
          }}
        }}
      ]
    }}
  ]
}}

【提取要求】
1. **proposed_method**：只提取论文自身提出的方法。summary 是核心，参考 Method 章节的 overview，用 3-5 句话连贯描述整体方案。不要罗列内部子组件（如某个具体的特征融合模块、GRL、注意力头等）
2. **feature_extractors**：提取用于从原始数据（音频/图像/文本）中提取特征的预处理方法。这些通常是已有的预训练模型或经典信号处理方法，不是论文原创的。如果论文没有使用额外的特征提取器（直接端到端），返回空数组 []
3. **network_architectures**：提取论文中使用的经典神经网络架构名称（如 Transformer、Conformer、BiLSTM、Mamba、TCN、ResNet、CNN 等）。只提取被用作整体架构骨架的知名网络结构，不提取论文自己命名的小组件。如果论文没有使用这些知名架构，返回 []
4. **loss_functions**：提取论文使用的损失函数。只在论文明确指出时提取。如果没有特殊损失函数，返回 []
5. **experiment_datasets**：只提取在 Experiment/Results 章节实际用于训练或评测的数据集。论文仅在 Related Work 或 Introduction 中提到但不使用的数据集不要提取
6. **experiment_results**：提取 Results 章节的数据表中的所有数值（含 baseline）。把表格中每一行方法的数据都提取出来。指标名称用简写（EER、min t-DCF、accuracy、F1 等）。如果同一数据集有多个条件（如 seen/unseen），分成多个对象
7. 如果论文没有某类信息，对应字段返回空数组 [] 或空对象 {{}}
8. 返回合法 JSON，不要注释或额外文字"""


# ============================================================
# 提取主函数
# ============================================================

def extract_paper_info(md_text: str) -> Dict:
    """
    用 LLM Agent 从论文 MD 中提取完整结构化信息（v2 精简版）。

    返回:
        {
            "success": bool,
            "data": {
                "title", "keywords", "authors", "year", "venue",
                "proposed_method": {name, summary, category},
                "feature_extractors": [{name, description, variant}],
                "loss_functions": [{name, description}],
                "experiment_datasets": [{name, description, role, task}],
                "experiment_results": [{dataset, condition, results: [{method_name, is_baseline, metrics}]}],
            },
            "error": str|None,
        }
    """
    if not md_text or len(md_text.strip()) < 100:
        logger.warning("论文提取跳过: 文本太短 (%d 字符)", len(md_text) if md_text else 0)
        return {"success": False, "data": None, "error": "文本太短，无法提取"}

    # 截断过长的文本：保留头60%（标题+摘要+方法）和尾40%（实验+结果）
    max_chars = KG_PAPER_EXTRACTION_MAX_CHARS
    if len(md_text) > max_chars:
        head_len = int(max_chars * 0.6)
        tail_len = int(max_chars * 0.4)
        truncated = md_text[:head_len] + "\n\n... [中间内容省略] ...\n\n" + md_text[-tail_len:]
        logger.info("论文文本截断: %d → %d 字符 (头%d + 尾%d)", len(md_text), max_chars, head_len, tail_len)
    else:
        truncated = md_text

    try:
        from agent.llm_service import chat_completion, extract_content

        user_prompt = PAPER_EXTRACTION_USER_TEMPLATE.format(paper_markdown=truncated)
        messages = [
            {"role": "system", "content": PAPER_EXTRACTION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("论文提取 Agent v2 开始: 文本长度 %d 字符", len(truncated))

        response = chat_completion(
            messages=messages,
            tools=None,
            temperature=0.1,
            max_tokens=4096,
        )
        content = extract_content(response)

        result = _parse_extraction_result(content)

        logger.info("论文提取完成: title='%s', proposed_method='%s', fes=%d, nets=%d, "
                    "losses=%d, exp_ds=%d, exp_tables=%d",
                    result.get("title", "")[:60],
                    result.get("proposed_method", {}).get("name", "")[:40],
                    len(result.get("feature_extractors", [])),
                    len(result.get("network_architectures", [])),
                    len(result.get("loss_functions", [])),
                    len(result.get("experiment_datasets", [])),
                    len(result.get("experiment_results", [])))

        return {"success": True, "data": result, "error": None}

    except json.JSONDecodeError as e:
        logger.warning("论文提取 JSON 解析失败: %s", e)
        return {"success": False, "data": None, "error": f"JSON 解析失败: {e}"}
    except Exception as e:
        logger.error("论文提取失败: %s", e, exc_info=True)
        return {"success": False, "data": None, "error": str(e)}


def _parse_extraction_result(raw: str) -> Dict:
    """解析 LLM 返回的 JSON，校验并补全缺失字段"""
    # 清理 markdown 代码块
    content = raw.strip()
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = '\n'.join(lines)

    # 容错：找 JSON 起止位置
    start = content.find('{')
    end = content.rfind('}')
    if start != -1 and end != -1 and end > start:
        content = content[start:end + 1]

    result = json.loads(content)

    # 顶层默认值
    defaults = {
        "title": "", "keywords": [], "authors": [],
        "year": None, "venue": "",
        "proposed_method": None,
        "feature_extractors": [],
        "network_architectures": [],
        "loss_functions": [],
        "experiment_datasets": [],
        "experiment_results": [],
    }
    for key, default in defaults.items():
        if key not in result:
            result[key] = default

    # 年份清理
    if isinstance(result.get("year"), str):
        try:
            result["year"] = int(result["year"])
        except ValueError:
            result["year"] = None

    # proposed_method 默认值
    if not result.get("proposed_method"):
        result["proposed_method"] = {"name": "", "summary": "", "category": ""}
    pm = result["proposed_method"]
    pm.setdefault("name", "")
    pm.setdefault("summary", "")
    pm.setdefault("category", "")
    valid_pm_categories = {"framework", "classifier", "feature_extractor",
                           "loss_function", "data_augmentation"}
    if pm.get("category") not in valid_pm_categories:
        pm["category"] = "framework"

    # feature_extractors 清理
    for fe in result.get("feature_extractors", []):
        fe.setdefault("description", "")
        fe.setdefault("variant", "")

    # network_architectures 清理
    for na in result.get("network_architectures", []):
        na.setdefault("description", "")

    # loss_functions 清理
    for lf in result.get("loss_functions", []):
        lf.setdefault("description", "")

    # experiment_datasets 清理
    for ds in result.get("experiment_datasets", []):
        if ds.get("role") not in ("eval", "train", "both"):
            ds["role"] = "eval"
        ds.setdefault("description", "")
        ds.setdefault("task", "")

    # experiment_results 清理
    for er in result.get("experiment_results", []):
        er.setdefault("condition", "")
        er.setdefault("dataset", "")
        for r in er.get("results", []):
            r.setdefault("is_baseline", False)
            if not r.get("metrics"):
                r["metrics"] = {}

    return result


# ============================================================
# 前端展示格式化（v2 精简版）
# ============================================================

def format_extraction_for_display(result: Dict) -> str:
    """将提取结果格式化为前端展示用的 Markdown"""
    if not result or not result.get("success"):
        error_msg = result.get("error", "提取失败") if result else "无提取结果"
        return f"⚠️ 论文信息提取失败: {error_msg}"

    data = result["data"]
    lines = []

    # ---- 标题 ----
    title = data.get("title", "未知标题")
    year = data.get("year", "")
    venue = data.get("venue", "")
    year_venue = f" ({year})" if year else ""
    year_venue += f" — *{venue}*" if venue else ""
    lines.append(f"### 📄 {title}{year_venue}")

    # ---- 关键词 ----
    keywords = data.get("keywords", [])
    if keywords:
        kw_str = " · ".join([f"`{kw}`" for kw in keywords[:15]])
        lines.append(f"🏷️ {kw_str}")

    # ---- 作者 ----
    authors = data.get("authors", [])
    if authors:
        lines.append(f"👤 {', '.join(authors[:10])}")
    lines.append("")

    # ---- 提出的方法 ----
    pm = data.get("proposed_method") or {}
    if pm.get("name") or pm.get("summary"):
        lines.append("#### 🏗️ 提出的方法")
        pm_name = pm.get("name", "未命名方法")
        pm_cat = pm.get("category", "")
        cat_labels = {
            "framework": "整体框架", "classifier": "分类器",
            "feature_extractor": "特征提取器", "loss_function": "损失函数",
            "data_augmentation": "数据增强",
        }
        cat_label = cat_labels.get(pm_cat, pm_cat)
        lines.append(f"**{pm_name}** `{cat_label}`")
        if pm.get("summary"):
            lines.append(f"> {pm['summary']}")
        lines.append("")

    # ---- 特征提取器 ----
    fes = data.get("feature_extractors", [])
    if fes:
        lines.append("#### 🎛️ 特征提取器")
        for fe in fes:
            name = fe.get("name", "?")
            variant = fe.get("variant", "")
            desc = fe.get("description", "")
            v_str = f" *({variant})*" if variant else ""
            lines.append(f"- **{name}**{v_str}")
            if desc:
                lines.append(f"  > {desc}")
        lines.append("")

    # ---- 网络框架 ----
    nas = data.get("network_architectures", [])
    if nas:
        lines.append("#### 🧠 网络框架")
        for na in nas:
            name = na.get("name", "?")
            desc = na.get("description", "")
            lines.append(f"- **{name}**")
            if desc:
                lines.append(f"  > {desc}")
        lines.append("")

    # ---- 损失函数 ----
    lfs = data.get("loss_functions", [])
    if lfs:
        lines.append("#### 📉 损失函数")
        for lf in lfs:
            name = lf.get("name", "?")
            desc = lf.get("description", "")
            lines.append(f"- **{name}**")
            if desc:
                lines.append(f"  > {desc}")
        lines.append("")

    # ---- 实验数据集 ----
    datasets = data.get("experiment_datasets", [])
    if datasets:
        lines.append("#### 📊 实验数据集")
        for ds in datasets:
            name = ds.get("name", "?")
            role = ds.get("role", "eval")
            task = ds.get("task", "")
            desc = ds.get("description", "")
            role_label = {"eval": "评测", "train": "训练", "both": "训练+评测"}.get(role, role)
            meta = [f"`{role_label}`"]
            if task:
                meta.append(task)
            lines.append(f"- **{name}** {' | '.join(meta)}")
            if desc:
                lines.append(f"  > {desc}")
        lines.append("")

    # ---- 实验结果 ----
    exp_results = data.get("experiment_results", [])
    if exp_results:
        lines.append("#### 📏 实验结果")
        for er in exp_results:
            ds_name = er.get("dataset", "未知数据集")
            condition = er.get("condition", "")
            cond_str = f" ({condition})" if condition else ""
            lines.append(f"**{ds_name}{cond_str}**")

            # 构建表格
            results_list = er.get("results", [])
            if results_list:
                # 收集所有指标名称
                all_metrics = set()
                for r in results_list:
                    all_metrics.update(r.get("metrics", {}).keys())
                sorted_metrics = sorted(all_metrics)

                # 表头
                header_cols = ["方法"] + sorted_metrics
                header = "| " + " | ".join(header_cols) + " |"
                separator = "|" + "|".join([" --- " for _ in header_cols]) + "|"
                lines.append(header)
                lines.append(separator)

                # 表行
                for r in results_list:
                    method_name = r.get("method_name", "?")
                    is_baseline = r.get("is_baseline", False)
                    display_name = f"{method_name} Ⓑ" if is_baseline else method_name
                    row = [display_name]
                    for m_name in sorted_metrics:
                        val = r.get("metrics", {}).get(m_name)
                        if val is not None:
                            row.append(str(val))
                        else:
                            row.append("—")
                    lines.append("| " + " | ".join(row) + " |")
            lines.append("")

    lines.append("---")
    return '\n'.join(lines)


def format_extraction_summary(result: Dict) -> str:
    """生成提取结果的简要摘要"""
    if not result or not result.get("success"):
        return "提取失败"
    data = result["data"]
    parts = []
    pm = data.get("proposed_method") or {}
    if pm.get("name"):
        parts.append(f"方法: {pm['name'][:40]}")
    if data.get("network_architectures"):
        parts.append(f"{len(data['network_architectures'])}个网络框架")
    if data.get("experiment_datasets"):
        parts.append(f"{len(data['experiment_datasets'])}个数据集")
    if data.get("experiment_results"):
        parts.append(f"{len(data['experiment_results'])}个实验结果表")
    return f"✅ 提取完成: {', '.join(parts)}" if parts else "✅ 提取完成"
