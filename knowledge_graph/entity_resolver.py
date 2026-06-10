"""
实体消歧器 (Entity Resolver)
- 入库时实时消歧，尽量避免事后批量归一化
- 三级级联策略:
    L1: 精确名称匹配 (DB 查询, O(1))
    L2: 模糊规则匹配 (子串包含、token重叠、别名匹配)
    L3: LLM 语义判断 (结合 description 做最终判定)
"""
import re
import json
from difflib import SequenceMatcher
from typing import Dict, List, Tuple, Optional

from database.connection import SessionLocal
from knowledge_graph.models import KgMethod, KgDataset, KgTask, KgMetric
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 通用工具函数
# ============================================================

def _normalize_key(s: str) -> str:
    """生成归一化 key：小写、去标点、去多余空格、排序词"""
    s = s.lower().strip()
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    tokens = sorted(s.split())
    return ' '.join(tokens)


def _token_overlap(a: str, b: str) -> float:
    """计算 token 重叠度"""
    ta = set(_normalize_key(a).split())
    tb = set(_normalize_key(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def _is_substring_or_contained(new_name: str, existing_name: str) -> bool:
    """
    检查是否一个是另一个的子串/变体。
    如 "wav2vec 2.0 Base" 包含 "wav2vec 2.0"
    """
    nn = new_name.lower().strip()
    en = existing_name.lower().strip()
    # 短名包含在长名中
    if len(nn) < len(en):
        return nn in en
    else:
        return en in nn


def _normalized_equals(a: str, b: str) -> bool:
    """去掉空格、标点后比较"""
    na = re.sub(r'[^a-z0-9]', '', a.lower())
    nb = re.sub(r'[^a-z0-9]', '', b.lower())
    return na == nb


# ============================================================
# L1: 精确名称匹配
# ============================================================

def _exact_match(entity_type: str, name: str) -> Optional[str]:
    """
    Level 1: 精确名称匹配。
    返回 entity_id 如果精确匹配成功，否则 None。
    """
    db = SessionLocal()
    try:
        model_map = {
            "method": (KgMethod, KgMethod.method_id, KgMethod.name),
            "dataset": (KgDataset, KgDataset.dataset_id, KgDataset.name),
            "task": (KgTask, KgTask.task_id, KgTask.name),
        }
        entry = model_map.get(entity_type)
        if not entry:
            return None

        model, id_col, name_col = entry
        result = db.query(id_col).filter(name_col == name).first()
        if result:
            logger.debug("L1 精确匹配 '%s': %s", name, result[0][:12])
            return result[0]
        return None
    finally:
        db.close()


# ============================================================
# L2: 模糊规则匹配
# ============================================================

def _fuzzy_match(entity_type: str, name: str, description: str = "") -> List[Dict]:
    """
    Level 2: 模糊规则匹配。
    用子串包含、归一化相等、别名匹配、高token重叠来找候选。

    返回候选列表: [{"id": ..., "name": ..., "description": ..., "match_type": ...}, ...]
    match_type: "substring" | "normalized_equal" | "alias_match" | "token_overlap"
    """
    db = SessionLocal()
    try:
        model_map = {
            "method": (KgMethod, KgMethod.method_id, KgMethod.name,
                       KgMethod.aliases, KgMethod.description),
            "dataset": (KgDataset, KgDataset.dataset_id, KgDataset.name,
                        None, KgDataset.description),
            "task": (KgTask, KgTask.task_id, KgTask.name,
                     None, KgTask.description),
        }
        entry = model_map.get(entity_type)
        if not entry:
            return []

        model, id_col, name_col, alias_col, desc_col = entry
        all_entities = db.query(id_col, name_col, desc_col).all()
        if alias_col is not None:
            alias_rows = db.query(id_col, alias_col).all()
            alias_map = {}
            for row in alias_rows:
                eid, aliases_raw = row
                if aliases_raw:
                    aliases = aliases_raw if isinstance(aliases_raw, list) else json.loads(str(aliases_raw))
                    alias_map[row[0]] = aliases
        else:
            alias_map = {}

        candidates = []
        for row in all_entities:
            eid, ename, edesc = row
            if ename == name:
                continue  # L1 已处理

            # 子串包含
            if _is_substring_or_contained(name, ename):
                candidates.append({
                    "id": eid, "name": ename, "description": edesc or "",
                    "match_type": "substring",
                })
                continue

            # 归一化相等 (去空格标点后相同)
            if _normalized_equals(name, ename):
                candidates.append({
                    "id": eid, "name": ename, "description": edesc or "",
                    "match_type": "normalized_equal",
                })
                continue

            # 别名匹配
            if eid in alias_map:
                for alias in alias_map[eid]:
                    if alias.lower().strip() == name.lower().strip():
                        candidates.append({
                            "id": eid, "name": ename, "description": edesc or "",
                            "match_type": "alias_match",
                        })
                        break
                    if _is_substring_or_contained(name, alias):
                        candidates.append({
                            "id": eid, "name": ename, "description": edesc or "",
                            "match_type": "alias_match",
                        })
                        break

            # 高 token 重叠 (>75%)
            overlap = _token_overlap(name, ename)
            if overlap >= 0.75:
                candidates.append({
                    "id": eid, "name": ename, "description": edesc or "",
                    "match_type": f"token_overlap({overlap:.0%})",
                })

        return candidates
    finally:
        db.close()


# ============================================================
# L3: LLM 语义判断
# ============================================================

RESOLVER_SYSTEM = """你是一个学术知识图谱的实体消歧专家。你的任务是判断一个新提取的实体名称是否与知识库中已有的某个实体是同一个概念（只是名称写法不同）。

【判断准则】
1. **名称差异**：缩写 vs 全称（"PS" vs "Physical Access"）、子集标注（"Track 2"）、空格/连字符差异 → 可能是同一实体
2. **描述一致性**：如果两个实体的描述指向同一事物（同一数据集内容、同一方法原理、同一任务目标）→ 强烈信号合并
3. **引用一致性**（数据集）：如果引用了相同的原始论文 → 强烈信号合并
4. **不确定不合并**：如果你不能确定两者相同，选择 different

【输出格式】
严格返回 JSON，不要额外文字：
{
  "decisions": [
    {
      "new_name": "新实体名称",
      "match": "same|different|child",
      "canonical_name": "规范名称（如果match=same/child）",
      "canonical_id": "对应候选实体的ID（如果match=same/child）",
      "reason": "1句话理由"
    }
  ]
}

【match 含义】
- "same": 同一实体，应该合并
- "child": 新实体是已有实体的子集/变体，但不应合并（如 ADD 2023 Track 2 是 ADD 2023 的子track）
- "different": 不同的实体，不应合并"""


def _llm_confirm(
    entity_type: str,
    new_name: str,
    new_description: str,
    new_context: Dict = None,
    candidates: List[Dict] = None,
) -> Tuple[Optional[str], str]:
    """
    Level 3: 用 LLM 语义判断新实体是否与候选实体是同一个。

    参数:
        entity_type: "method" | "dataset" | "task"
        new_name: 新提取的实体名称
        new_description: 新实体的描述
        new_context: 额外上下文 (如 references_cited, is_track_of, category)
        candidates: L2 找到的候选实体列表

    返回:
        (canonical_id 或 None, 决策描述)
    """
    if not candidates:
        return None, "无候选实体"

    # 构建候选实体列表文本
    candidate_parts = []
    for i, c in enumerate(candidates):
        parts = [
            f"候选 {i+1}:",
            f"  ID: {c['id']}",
            f"  名称: {c['name']}",
        ]
        if c.get("description"):
            parts.append(f"  描述: {c['description'][:300]}")
        if c.get("match_type"):
            parts.append(f"  匹配方式: {c['match_type']}")
        candidate_parts.append('\n'.join(parts))

    # 构建新实体信息
    new_parts = [
        f"类型: {entity_type}",
        f"名称: {new_name}",
    ]
    if new_description:
        new_parts.append(f"描述: {new_description[:300]}")
    if new_context:
        refs = new_context.get("references_cited", [])
        if refs:
            if isinstance(refs[0], dict):
                ref_str = "; ".join(
                    f"{r.get('authors','')} ({r.get('year','')}) {r.get('title','')}"
                    for r in refs[:3]
                )
            else:
                ref_str = "; ".join(str(r) for r in refs[:3])
            new_parts.append(f"引用文献: {ref_str}")
        track_of = new_context.get("is_track_of", "")
        if track_of:
            new_parts.append(f"所属主数据集: {track_of}")
        category = new_context.get("category", "")
        if category:
            new_parts.append(f"方法类别: {category}")

    user_prompt = f"""请判断以下新提取的实体是否与候选实体是同一个概念。

【新实体】
{chr(10).join(new_parts)}

【候选已有实体】
{chr(10).join(candidate_parts)}

请逐一判断新实体与每个候选实体的关系，返回 JSON。"""

    try:
        from agent.llm_service import chat_completion, extract_content

        messages = [
            {"role": "system", "content": RESOLVER_SYSTEM},
            {"role": "user", "content": user_prompt},
        ]

        logger.info("消歧 LLM 调用: type=%s, name='%s', candidates=%d",
                    entity_type, new_name[:60], len(candidates))

        response = chat_completion(
            messages=messages,
            tools=None,
            temperature=0.1,
            max_tokens=1024,
        )
        content = extract_content(response)

        # 解析
        decisions = _parse_resolver_result(content)
        if not decisions:
            return None, "LLM 解析失败"

        # 找第一个 match="same" 的决策
        for d in decisions:
            if d.get("match") == "same":
                canonical_id = d.get("canonical_id", "")
                reason = d.get("reason", "")
                logger.info("LLM 消歧: '%s' → '%s' (ID: %s) — %s",
                            new_name[:40], d.get("canonical_name", "")[:40],
                            canonical_id[:12], reason)
                return canonical_id, reason

        # 如果有 child 关系，标记但不合并
        for d in decisions:
            if d.get("match") == "child":
                logger.info("LLM 消歧: '%s' 是 '%s' 的子集，不合并",
                            new_name[:40], d.get("canonical_name", "")[:40])
                return None, f"子集关系: {d.get('reason', '')}"

        return None, "LLM 判定不同实体"

    except Exception as e:
        logger.warning("LLM 消歧调用失败: %s", e)
        return None, f"LLM 调用异常: {e}"


def _parse_resolver_result(raw: str) -> List[Dict]:
    """解析 LLM 消歧决策"""
    content = raw.strip()
    if content.startswith("```"):
        lines = content.split('\n')
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        content = '\n'.join(lines)

    start = content.find('{')
    end = content.rfind('}')
    if start == -1 or end == -1:
        logger.warning("消歧 LLM 返回无 JSON: %.200s", content)
        return []

    try:
        result = json.loads(content[start:end + 1])
        decisions = result.get("decisions", [])
        return decisions
    except json.JSONDecodeError as e:
        logger.warning("消歧 LLM JSON 解析失败: %s", e)
        return []


# ============================================================
# 主消歧函数
# ============================================================

def resolve_entity(
    entity_type: str,
    name: str,
    description: str = "",
    context: Dict = None,
    use_llm: bool = True,
) -> Tuple[str, bool]:
    """
    对单个实体做三级消歧，返回 canonical entity_id 和是否新建。

    参数:
        entity_type: "method" | "dataset" | "task"
        name: 实体名称（从论文中提取的原始名称）
        description: 实体描述（1-2句话）
        context: 额外上下文 (category, references_cited, is_track_of, etc.)
        use_llm: 是否启用 L3 LLM 判断

    返回:
        (entity_id, is_new) — entity_id 是 canonical id，is_new 表示是否创建了新实体

    消歧流程:
        L1: 精确名称匹配 → 直接复用
        L2: 模糊规则 → 找到候选
            ├── 仅1个候选且匹配类型强(substring/normalized_equal) → 直接复用
            └── 多个候选或弱匹配 → 进入 L3
        L3: LLM 语义判断 → 决定复用或新建
    """
    context = context or {}

    # === Level 1: 精确名称匹配 ===
    exact_id = _exact_match(entity_type, name)
    if exact_id:
        logger.info("消歧 L1 命中: type=%s, '%s' → %s",
                    entity_type, name[:60], exact_id[:12])
        return exact_id, False

    # === Level 2: 模糊规则匹配 ===
    candidates = _fuzzy_match(entity_type, name, description)
    if not candidates:
        logger.info("消歧 L2 无候选: type=%s, '%s' → 创建新实体",
                    entity_type, name[:60])
        return None, True  # None 表示需要新建

    # 如果只有一个候选且匹配类型强，直接复用
    strong_match_types = {"substring", "normalized_equal", "alias_match"}
    if len(candidates) == 1 and candidates[0]["match_type"] in strong_match_types:
        cid = candidates[0]["id"]
        logger.info("消歧 L2 直接复用: type=%s, '%s' → %s (match=%s)",
                    entity_type, name[:60], cid[:12], candidates[0]["match_type"])
        return cid, False

    # === Level 3: LLM 语义判断 ===
    if use_llm:
        canonical_id, reason = _llm_confirm(
            entity_type, name, description, context, candidates,
        )
        if canonical_id:
            return canonical_id, False
        else:
            logger.info("消歧 L3 判定: type=%s, '%s' → 新建 (%s)",
                        entity_type, name[:60], reason)
            return None, True
    else:
        # 不启用 LLM，L2 有多个候选时保守处理：不合并，创建新实体
        logger.info("消歧 L2 多候选(LLM disabled): type=%s, '%s' → 新建",
                    entity_type, name[:60])
        return None, True


# ============================================================
# 批量消歧（对 L2 未解决的候选统一做一次 LLM 调用）
# ============================================================

def resolve_entities_batch(
    entity_type: str,
    entities: List[Dict],
    use_llm: bool = True,
) -> List[Tuple[str, bool]]:
    """
    批量消歧：对多个实体同时做 L1→L2→L3。

    参数:
        entity_type: "method" | "dataset" | "task"
        entities: [{"name": ..., "description": ..., "context": {...}}, ...]
        use_llm: 是否启用 L3

    返回:
        [(entity_id, is_new), ...] 与输入顺序一一对应
    """
    results = []
    llm_batch = []  # 需要 L3 判断的 (index, new_name, new_desc, new_ctx, candidates)

    for i, ent in enumerate(entities):
        name = ent.get("name", "")
        description = ent.get("description", "")
        context = ent.get("context", {})

        # L1
        exact_id = _exact_match(entity_type, name)
        if exact_id:
            results.append((exact_id, False))
            continue

        # L2
        candidates = _fuzzy_match(entity_type, name, description)
        if not candidates:
            results.append((None, True))
            continue

        strong_match_types = {"substring", "normalized_equal", "alias_match"}
        if len(candidates) == 1 and candidates[0]["match_type"] in strong_match_types:
            results.append((candidates[0]["id"], False))
            continue

        # 需要 L3
        llm_batch.append((i, name, description, context, candidates))
        results.append(None)  # 占位

    # L3: 批量调用 LLM
    if llm_batch and use_llm:
        # 合并所有需要消歧的实体到一次 LLM 调用
        batch_parts = []
        batch_map = []  # (result_index, entity_name, candidate_ids)

        for idx, name, desc, ctx, candidates in llm_batch:
            entity_index = len(batch_parts)  # 在 batch 中的位置
            batch_map.append((idx, name, [c["id"] for c in candidates]))

            parts = [
                f"## 新实体 {entity_index + 1}",
                f"类型: {entity_type}",
                f"名称: {name}",
            ]
            if desc:
                parts.append(f"描述: {desc[:300]}")
            if ctx:
                refs = ctx.get("references_cited", [])
                if refs:
                    if isinstance(refs[0], dict):
                        ref_str = "; ".join(
                            f"{r.get('authors','')} ({r.get('year','')})"
                            for r in refs[:3]
                        )
                    else:
                        ref_str = "; ".join(str(r) for r in refs[:3])
                    parts.append(f"引用: {ref_str}")
                track_of = ctx.get("is_track_of", "")
                if track_of:
                    parts.append(f"所属主数据集: {track_of}")

            parts.append("候选已有实体:")
            for j, c in enumerate(candidates):
                parts.append(f"  [{chr(65+j)}] ID: {c['id']}, 名称: {c['name']}")
                if c.get("description"):
                    parts.append(f"      描述: {c['description'][:200]}")

            batch_parts.append('\n'.join(parts))

        user_prompt = f"""请判断以下新提取的实体是否与各自的候选实体是同一个概念。

{chr(10).join(batch_parts)}

请返回 JSON:
{{"decisions": [
  {{"entity_index": 实体编号(1开始), "match": "same|different|child", "canonical_id": "候选实体ID(如A、B..对应的实际ID)", "reason": "理由"}}
]}}"""

        try:
            from agent.llm_service import chat_completion, extract_content

            messages = [
                {"role": "system", "content": RESOLVER_SYSTEM},
                {"role": "user", "content": user_prompt},
            ]

            logger.info("批量消歧 LLM: type=%s, entities=%d",
                        entity_type, len(llm_batch))

            response = chat_completion(
                messages=messages,
                tools=None,
                temperature=0.1,
                max_tokens=2048,
            )
            content = extract_content(response)
            decisions = _parse_resolver_result(content)

            # 应用决策
            for d in decisions:
                entity_idx = d.get("entity_index", 0) - 1  # 1-based → 0-based
                if entity_idx < 0 or entity_idx >= len(batch_map):
                    continue
                result_idx, orig_name, candidate_ids = batch_map[entity_idx]
                if d.get("match") == "same":
                    canonical_id = d.get("canonical_id", "")
                    if canonical_id and canonical_id in candidate_ids:
                        results[result_idx] = (canonical_id, False)
                        logger.info("批量消歧: '%s' → %s", orig_name[:40], canonical_id[:12])
                        continue
                # 默认新建
                results[result_idx] = (None, True)

        except Exception as e:
            logger.warning("批量消歧 LLM 失败: %s", e)
            # 失败时，L2 无强匹配的实体全部新建
            for idx, name, _, _, _ in llm_batch:
                if results[idx] is None:
                    results[idx] = (None, True)

    # 补全未处理的结果
    for i, r in enumerate(results):
        if r is None:
            results[i] = (None, True)

    return results
