"""
实体归一化 Agent
- 使用 LLM 的理解能力判断知识图谱中名称不同的实体是否等价
- 替代纯 SequenceMatcher 模糊匹配，特别是数据集归一化
- 结合描述文本 + 引用文献做综合判断

归一化策略：
  Phase 1: find_similar_groups() 模糊匹配 → 候选组
  Phase 2: LLM Agent 确认合并决策（本模块）
  Phase 3: merge_*() 执行实际合并
"""
import json
from typing import Dict, List, Optional

from utils.logger import get_logger
from config import KG_NORMALIZATION_USE_LLM

logger = get_logger(__name__)

# ============================================================
# 归一化 Prompt 模板
# ============================================================

NORMALIZATION_SYSTEM = """你是一个学术知识图谱的实体消歧专家。你的任务是判断知识图谱中哪些实体实际上是同一个概念，只是名称表述不同。

【判断依据】
1. **名称相似度**：名称是否显而易见是同一事物的不同写法（如 "wav2vec2" vs "wav2vec 2.0"）
2. **描述一致性**：描述是否指向同一概念/事物
3. **引用一致性**（数据集专属）：是否引用了相同的原始论文、在同一任务中使用
4. **上下文重叠**：关联的论文/方法是否重叠

【输出格式】
严格返回 JSON，不要额外文字：
{
  "merge_groups": [
    {
      "canonical_name": "规范名称（选择最标准/最完整的表述）",
      "duplicate_ids": ["应合并的实体ID1", "应合并的实体ID2"],
      "reason": "合并理由（1句话）"
    }
  ]
}
如果不需要合并任何实体，返回 {"merge_groups": []}

【注意】
- 只合并你确信是同一实体的，不确定的不要合并
- canonical_name 选择最完整、最标准的表述
- 如果两个实体只是相关但不同（如子集/变体），不应合并"""


NORMALIZATION_USER_METHOD = """请判断以下 **方法/模型** 实体中，哪些应该合并为同一个方法。

{entity_list}

请分析并返回合并建议。记住：只有确信是同一方法（不同写法/表述）时才合并。
例如 "wav2vec 2.0 Base" 和 "wav2vec2" 是同一方法；但 "wav2vec 2.0" 和 "wavlm" 是不同的方法。"""


NORMALIZATION_USER_DATASET = """请判断以下 **数据集** 实体中，哪些应该合并为同一个数据集。

{entity_list}

【特别提示】
- 数据集的 track/子集 如果论文中当作不同评测场景使用，不要合并（如 "ADD 2023 Track 1" vs "ADD 2023 Track 2" 可能是不同的子赛道）
- 但如果多个实体指向完全相同的评测集（只是名称省略了细节），应该合并
- 引用相同原始论文是强烈信号
- 描述指向同一数据内容也是强烈信号

请分析并返回合并建议。"""


NORMALIZATION_USER_TASK = """请判断以下 **研究任务** 实体中，哪些应该合并为同一个任务。

{entity_list}

请分析并返回合并建议。记住：意思相同的不同表述应该合并（如 "audio deepfake detection" 和 "audio deepfake detection task"）。"""


# ============================================================
# 主函数
# ============================================================

def normalize_with_llm(
    entity_type: str,
    candidate_groups: List[List[Dict]],
    dry_run: bool = False,
) -> Dict:
    """
    用 LLM 判断候选组内的实体是否应该合并。

    参数:
        entity_type: "method" | "dataset" | "task"
        candidate_groups: 候选组列表，每组格式:
            [
                {"id": "xxx", "name": "实体名", "description": "描述", ...},
                {"id": "yyy", "name": "实体名", "description": "描述", ...},
            ]
        dry_run: 如果 True，只返回 LLM 决策，不执行合并

    返回:
        {
            "merge_decisions": [
                {
                    "canonical_name": str,
                    "canonical_id": str,
                    "duplicate_ids": [str, ...],
                    "reason": str,
                }
            ],
            "llm_calls": int,  # LLM 调用次数
        }
    """
    if not KG_NORMALIZATION_USE_LLM:
        return {"merge_decisions": [], "llm_calls": 0}

    # 过滤：每组至少 2 个实体才需要归一化
    valid_groups = [g for g in candidate_groups if len(g) >= 2]
    if not valid_groups:
        return {"merge_decisions": [], "llm_calls": 0}

    # 选择 Prompt 模板
    user_templates = {
        "method": NORMALIZATION_USER_METHOD,
        "dataset": NORMALIZATION_USER_DATASET,
        "task": NORMALIZATION_USER_TASK,
    }
    user_template = user_templates.get(entity_type, NORMALIZATION_USER_METHOD)

    from agent.llm_service import chat_completion, extract_content

    all_decisions = []
    llm_calls = 0

    # 对每组候选实体调用 LLM（如果组太多，可以合并多组到一次调用）
    # 为减少 LLM 调用，将最多 5 组合并到一次调用
    batch_size = 5
    for batch_start in range(0, len(valid_groups), batch_size):
        batch_groups = valid_groups[batch_start:batch_start + batch_size]

        # 构建实体列表文本
        entity_parts = []
        group_offset = 0
        for gi, group in enumerate(batch_groups):
            entity_parts.append(f"\n## 候选组 {gi + 1}")
            for ei, entity in enumerate(group):
                info_parts = [f"  ID: {entity['id']}"]
                info_parts.append(f"  名称: {entity['name']}")
                if entity.get("description"):
                    info_parts.append(f"  描述: {entity['description'][:200]}")
                if entity.get("aliases"):
                    aliases = entity["aliases"]
                    if isinstance(aliases, str):
                        try:
                            aliases = json.loads(aliases)
                        except json.JSONDecodeError:
                            aliases = [aliases]
                    if aliases:
                        info_parts.append(f"  别名: {', '.join(aliases[:5])}")
                if entity.get("references"):
                    info_parts.append(f"  关联引用: {', '.join(entity['references'][:5])}")
                if entity.get("paper_count"):
                    info_parts.append(f"  关联论文数: {entity['paper_count']}")
                if entity.get("domain"):
                    info_parts.append(f"  领域: {entity['domain']}")
                entity_parts.append('\n'.join(info_parts))

        entity_text = '\n'.join(entity_parts)
        user_prompt = user_template.format(entity_list=entity_text)

        messages = [
            {"role": "system", "content": NORMALIZATION_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        try:
            logger.info("归一化 Agent 调用: entity_type=%s, groups=%d", entity_type, len(batch_groups))
            response = chat_completion(
                messages=messages,
                tools=None,
                temperature=0.1,
                max_tokens=2048,
            )
            content = extract_content(response)
            llm_calls += 1

            # 解析 LLM 决策
            decisions = _parse_normalization_result(content)
            if decisions:
                # 将 LLM 返回的 duplicate_ids 映射回实际 entity ID
                all_decisions.extend(decisions)
                logger.info("归一化 Agent 决策: %d 组合并建议", len(decisions))

        except json.JSONDecodeError as e:
            logger.warning("归一化 LLM 返回的 JSON 无法解析: %s", e)
            continue
        except Exception as e:
            logger.error("归一化 Agent 调用失败: %s", e, exc_info=True)
            continue

    return {
        "merge_decisions": all_decisions,
        "llm_calls": llm_calls,
    }


def _parse_normalization_result(raw: str) -> List[Dict]:
    """解析 LLM 返回的归一化决策"""
    content = raw.strip()
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = '\n'.join(lines)

    # 找 JSON 起止
    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1:
        logger.warning("归一化结果中未找到 JSON: %.200s", content)
        return []
    content = content[start:end + 1]

    result = json.loads(content)
    merge_groups = result.get("merge_groups", [])

    # 校验格式
    valid = []
    for mg in merge_groups:
        if mg.get("canonical_name") and mg.get("duplicate_ids") and len(mg["duplicate_ids"]) >= 1:
            valid.append({
                "canonical_name": mg["canonical_name"],
                "duplicate_ids": mg["duplicate_ids"],
                "reason": mg.get("reason", ""),
            })

    return valid


# ============================================================
# 辅助函数：构建归一化上下文
# ============================================================

def build_method_context(method_ids: List[str]) -> List[Dict]:
    """为方法实体构建丰富的归一化上下文"""
    from database.connection import SessionLocal
    from knowledge_graph.models import KgMethod, KgPaperUsesMethod

    db = SessionLocal()
    try:
        contexts = []
        for mid in method_ids:
            method = db.query(KgMethod).filter(KgMethod.method_id == mid).first()
            if not method:
                continue

            # 统计关联论文数
            paper_count = db.query(KgPaperUsesMethod).filter(
                KgPaperUsesMethod.method_id == mid
            ).count()

            ctx = {
                "id": mid,
                "name": method.name,
                "description": method.description or "",
                "aliases": method.aliases if isinstance(method.aliases, list) else [],
                "paper_count": paper_count,
            }
            contexts.append(ctx)

        return contexts
    finally:
        db.close()


def build_dataset_context(dataset_ids: List[str]) -> List[Dict]:
    """为数据集实体构建丰富的归一化上下文（含引用信息）"""
    from database.connection import SessionLocal
    from knowledge_graph.models import KgDataset, KgPaperEvaluatesOn

    db = SessionLocal()
    try:
        contexts = []
        for did in dataset_ids:
            ds = db.query(KgDataset).filter(KgDataset.dataset_id == did).first()
            if not ds:
                continue

            # 统计关联论文
            paper_count = db.query(KgPaperEvaluatesOn).filter(
                KgPaperEvaluatesOn.dataset_id == did
            ).count()

            # 获取引用指纹（来自 entity_normalizer）
            from knowledge_graph.entity_normalizer import _get_dataset_citation_fingerprint
            fingerprint = _get_dataset_citation_fingerprint(did)

            ctx = {
                "id": did,
                "name": ds.name,
                "description": ds.description or "",
                "domain": ds.domain or "",
                "task": ds.task or "",
                "paper_count": paper_count,
                "references": fingerprint.split('|')[:5] if fingerprint else [],
            }
            contexts.append(ctx)

        return contexts
    finally:
        db.close()


def build_task_context(task_ids: List[str]) -> List[Dict]:
    """为任务实体构建归一化上下文"""
    from database.connection import SessionLocal
    from knowledge_graph.models import KgTask, KgPaperBelongsToTask

    db = SessionLocal()
    try:
        contexts = []
        for tid in task_ids:
            task = db.query(KgTask).filter(KgTask.task_id == tid).first()
            if not task:
                continue

            paper_count = db.query(KgPaperBelongsToTask).filter(
                KgPaperBelongsToTask.task_id == tid
            ).count()

            ctx = {
                "id": tid,
                "name": task.name,
                "description": task.description or "",
                "paper_count": paper_count,
            }
            contexts.append(ctx)

        return contexts
    finally:
        db.close()
