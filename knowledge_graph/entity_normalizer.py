"""
实体名称归一化模块
- 将同一实体（任务/数据集/方法）的不同表达合并到 canonical name
- 数据集归一化额外依据引用文献一致性
- 提供 Agent 工具和 UI 触发入口
"""
import re
import json
from difflib import SequenceMatcher
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

from database.connection import SessionLocal
from knowledge_graph.models import (
    KgPaper, KgMethod, KgDataset, KgTask,
    KgPaperUsesMethod, KgPaperEvaluatesOn, KgPaperTrainsOn,
    KgPaperBelongsToTask, KgPaperCites, KgPaperAuthor,
    KgPaperReportsMetric,
)
from knowledge_graph.graph_store import _mark_graph_dirty
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 归一化预处理
# ============================================================

def _normalize_key(s: str) -> str:
    """生成归一化 key：小写、去标点、去多余空格、排序词"""
    s = s.lower().strip()
    # 去掉特殊字符，保留字母数字
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    # 去掉多余空格
    s = re.sub(r'\s+', ' ', s).strip()
    # 排序 token（处理词序不同的情况）
    tokens = sorted(s.split())
    return ' '.join(tokens)


def _similarity(a: str, b: str) -> float:
    """计算两个字符串的相似度 (0-1)"""
    return SequenceMatcher(None, _normalize_key(a), _normalize_key(b)).ratio()


def _token_overlap(a: str, b: str) -> float:
    """计算 token 重叠度"""
    ta = set(_normalize_key(a).split())
    tb = set(_normalize_key(b).split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


# ============================================================
# 数据集归一化（基于引用文献）
# ============================================================

def _get_dataset_citation_fingerprint(dataset_id: str) -> str:
    """
    生成数据集的"引用指纹"：
    找出所有使用了该数据集的论文，提取它们共同引用的关键参考文献
    """
    db = SessionLocal()
    try:
        # 找到使用该数据集的所有论文
        rows = db.query(KgPaperEvaluatesOn.paper_id).filter(
            KgPaperEvaluatesOn.dataset_id == dataset_id
        ).all()
        paper_ids = [r[0] for r in rows]

        if len(paper_ids) < 1:
            return ""

        # 获取这些论文的所有引用
        citations = (
            db.query(KgPaperCites.cited_paper_id)
            .filter(KgPaperCites.citing_paper_id.in_(paper_ids))
            .all()
        )
        cited_ids = [r[0] for r in citations]

        if not cited_ids:
            return ""

        # 找高频引用（被多数论文引用的文献）
        from collections import Counter
        cite_counts = Counter(cited_ids)
        # 取被 >= 50% 论文引用的文献
        threshold = max(1, len(paper_ids) // 2)
        common_cites = sorted(
            [cid for cid, cnt in cite_counts.items() if cnt >= threshold]
        )
        return '|'.join(sorted(common_cites))
    finally:
        db.close()


# ============================================================
# 模糊分组
# ============================================================

def find_similar_groups(items: List[Tuple[str, str]], threshold: float = 0.75) -> List[List[str]]:
    """
    用模糊匹配 + token 重叠将相似 entity 分组。
    items: [(entity_id, entity_name), ...]
    返回: [[id1, id2, ...], ...] 每组内的 ID 应该合并
    """
    if len(items) <= 1:
        return []

    n = len(items)
    # 并查集
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    for i in range(n):
        for j in range(i + 1, n):
            id_i, name_i = items[i]
            id_j, name_j = items[j]
            sim = _similarity(name_i, name_j)
            overlap = _token_overlap(name_i, name_j)

            # 高相似度 OR (中等相似度 + 高 token 重叠)
            if sim >= threshold or (sim >= 0.55 and overlap >= 0.7):
                union(i, j)

    # 收集分组
    groups = defaultdict(list)
    for i in range(n):
        root = find(i)
        groups[root].append(items[i][0])

    # 只返回有 > 1 个成员的组
    return [sorted(ids) for ids in groups.values() if len(ids) > 1]


# ============================================================
# 合并实体
# ============================================================

def merge_methods(canonical_id: str, duplicate_ids: List[str]) -> int:
    """合并方法实体：把 duplicate 的关系全部迁移到 canonical，删除 duplicate"""
    db = SessionLocal()
    migrated = 0
    try:
        for dup_id in duplicate_ids:
            if dup_id == canonical_id:
                continue
            # 迁移 USES_METHOD 关系
            rels = db.query(KgPaperUsesMethod).filter(
                KgPaperUsesMethod.method_id == dup_id
            ).all()
            for r in rels:
                # 检查是否已有相同关系
                existing = db.query(KgPaperUsesMethod).filter(
                    KgPaperUsesMethod.paper_id == r.paper_id,
                    KgPaperUsesMethod.method_id == canonical_id,
                ).first()
                if not existing:
                    db.add(KgPaperUsesMethod(
                        paper_id=r.paper_id, method_id=canonical_id,
                        role=r.role, variant=r.variant,
                        context=r.context, performance_contribution=r.performance_contribution,
                    ))
                    db.flush()  # 即时刷入，避免同一事务内重复插入
                    migrated += 1
                db.delete(r)

            # 迁移 IMPROVES_UPON 关系
            from knowledge_graph.models import KgMethodImprovesMethod as KMI
            for r in db.query(KMI).filter(KMI.method_a_id == dup_id).all():
                r.method_a_id = canonical_id
            for r in db.query(KMI).filter(KMI.method_b_id == dup_id).all():
                r.method_b_id = canonical_id

            # 合并 aliases
            dup = db.query(KgMethod).filter(KgMethod.method_id == dup_id).first()
            canon = db.query(KgMethod).filter(KgMethod.method_id == canonical_id).first()
            if dup and canon:
                dup_aliases = dup.aliases or []
                canon_aliases = canon.aliases or []
                if isinstance(dup_aliases, str):
                    dup_aliases = json.loads(dup_aliases)
                if isinstance(canon_aliases, str):
                    canon_aliases = json.loads(canon_aliases)
                canon.aliases = list(set(canon_aliases + dup_aliases + [dup.name]))

            # 删除重复实体
            db.delete(dup)

        db.commit()
        logger.info("Methods 合并: canonical=%s, merged=%d, migrated=%d relations",
                    canonical_id[:12], len(duplicate_ids), migrated)
    except Exception as e:
        db.rollback()
        logger.error("Method 合并失败: %s", e)
        raise
    finally:
        db.close()
    _mark_graph_dirty()
    return migrated


def merge_datasets(canonical_id: str, duplicate_ids: List[str]) -> int:
    """合并数据集实体"""
    db = SessionLocal()
    migrated = 0
    try:
        for dup_id in duplicate_ids:
            if dup_id == canonical_id:
                continue
            # 迁移 EVALUATES_ON
            for r in db.query(KgPaperEvaluatesOn).filter(
                KgPaperEvaluatesOn.dataset_id == dup_id
            ).all():
                existing = db.query(KgPaperEvaluatesOn).filter(
                    KgPaperEvaluatesOn.paper_id == r.paper_id,
                    KgPaperEvaluatesOn.dataset_id == canonical_id,
                ).first()
                if not existing:
                    db.add(KgPaperEvaluatesOn(
                        paper_id=r.paper_id, dataset_id=canonical_id,
                        task=r.task, split=r.split,
                        metrics=r.metrics, protocol=r.protocol,
                    ))
                    db.flush()
                    migrated += 1
                db.delete(r)

            # 迁移 TRAINS_ON
            for r in db.query(KgPaperTrainsOn).filter(
                KgPaperTrainsOn.dataset_id == dup_id
            ).all():
                existing = db.query(KgPaperTrainsOn).filter(
                    KgPaperTrainsOn.paper_id == r.paper_id,
                    KgPaperTrainsOn.dataset_id == canonical_id,
                ).first()
                if not existing:
                    db.add(KgPaperTrainsOn(
                        paper_id=r.paper_id, dataset_id=canonical_id,
                        split=r.split,
                    ))
                    db.flush()
                    migrated += 1
                db.delete(r)

            # 迁移 REPORTS_METRIC
            for r in db.query(KgPaperReportsMetric).filter(
                KgPaperReportsMetric.dataset_id == dup_id
            ).all():
                r.dataset_id = canonical_id

            db.query(KgDataset).filter(KgDataset.dataset_id == dup_id).delete()

        db.commit()
        logger.info("Datasets 合并: canonical=%s, merged=%d, migrated=%d relations",
                    canonical_id[:12], len(duplicate_ids), migrated)
    except Exception as e:
        db.rollback()
        logger.error("Dataset 合并失败: %s", e)
        raise
    finally:
        db.close()
    _mark_graph_dirty()
    return migrated


def merge_tasks(canonical_id: str, duplicate_ids: List[str]) -> int:
    """合并任务实体"""
    db = SessionLocal()
    migrated = 0
    try:
        for dup_id in duplicate_ids:
            if dup_id == canonical_id:
                continue
            for r in db.query(KgPaperBelongsToTask).filter(
                KgPaperBelongsToTask.task_id == dup_id
            ).all():
                existing = db.query(KgPaperBelongsToTask).filter(
                    KgPaperBelongsToTask.paper_id == r.paper_id,
                    KgPaperBelongsToTask.task_id == canonical_id,
                ).first()
                if not existing:
                    db.add(KgPaperBelongsToTask(
                        paper_id=r.paper_id, task_id=canonical_id,
                        is_primary=r.is_primary,
                    ))
                    db.flush()
                    migrated += 1
                db.delete(r)

            db.query(KgTask).filter(KgTask.task_id == dup_id).delete()

        db.commit()
        logger.info("Tasks 合并: canonical=%s, merged=%d, migrated=%d relations",
                    canonical_id[:12], len(duplicate_ids), migrated)
    except Exception as e:
        db.rollback()
        logger.error("Task 合并失败: %s", e)
        raise
    finally:
        db.close()
    _mark_graph_dirty()
    return migrated


# ============================================================
# 主归一化流程
# ============================================================

def normalize_all_entities(dry_run: bool = False, use_llm: bool = None) -> Dict:
    """
    扫描所有实体，找出可合并的组并执行合并。

    三阶段:
      1. 模糊匹配候选召回 (find_similar_groups)
      2. LLM Agent 确认合并决策 (可选，由 KG_NORMALIZATION_USE_LLM 控制)
      3. 执行实际合并 (merge_*)

    dry_run=True 时只返回分析结果，不实际修改。
    use_llm 为 None 时使用 config.KG_NORMALIZATION_USE_LLM 的默认值。
    """
    if use_llm is None:
        from config import KG_NORMALIZATION_USE_LLM
        use_llm = KG_NORMALIZATION_USE_LLM

    db = SessionLocal()
    result = {
        "methods": {"groups": 0, "merged": 0, "llm_verified": False},
        "datasets": {"groups": 0, "merged": 0, "llm_verified": False},
        "tasks": {"groups": 0, "merged": 0, "llm_verified": False},
        "details": [],
    }

    try:
        # ================================================================
        # Phase 1: 模糊匹配候选召回
        # ================================================================
        logger.info("归一化 Phase 1: 模糊匹配候选召回")

        # --- 方法 ---
        methods = db.query(KgMethod.method_id, KgMethod.name).all()
        method_groups = find_similar_groups(
            [(m[0], m[1]) for m in methods], threshold=0.70
        )
        result["methods"]["groups"] = len(method_groups)

        # --- 数据集 ---
        datasets = db.query(KgDataset.dataset_id, KgDataset.name).all()
        ds_groups = find_similar_groups(
            [(d[0], d[1]) for d in datasets], threshold=0.65
        )
        ds_groups = _refine_dataset_groups_with_citations(ds_groups, datasets, db)
        result["datasets"]["groups"] = len(ds_groups)

        # --- 任务 ---
        tasks = db.query(KgTask.task_id, KgTask.name).all()
        task_groups = find_similar_groups(
            [(t[0], t[1]) for t in tasks], threshold=0.70
        )
        result["tasks"]["groups"] = len(task_groups)

        # ================================================================
        # Phase 2: LLM Agent 确认合并决策（可选）
        # ================================================================
        if use_llm and (method_groups or ds_groups or task_groups):
            logger.info("归一化 Phase 2: LLM Agent 确认合并")
            from knowledge_graph.entity_normalization_agent import (
                normalize_with_llm,
                build_method_context,
                build_dataset_context,
                build_task_context,
            )

            # --- 方法 LLM 归一化 ---
            if method_groups:
                try:
                    rich_groups = []
                    for group in method_groups:
                        ctx = build_method_context(group)
                        if ctx:
                            rich_groups.append(ctx)
                    if rich_groups:
                        llm_result = normalize_with_llm("method", rich_groups, dry_run=dry_run)
                        if llm_result.get("merge_decisions"):
                            _apply_llm_decisions(
                                "method", llm_result["merge_decisions"],
                                method_groups, methods, db,
                                result, dry_run,
                            )
                            result["methods"]["llm_verified"] = True
                            logger.info("LLM 方法归一化: %d 个合并决策", len(llm_result["merge_decisions"]))
                except Exception as e:
                    logger.warning("LLM 方法归一化失败，回退到模糊匹配: %s", e)

            # --- 数据集 LLM 归一化 ---
            if ds_groups:
                try:
                    rich_groups = []
                    for group in ds_groups:
                        ctx = build_dataset_context(group)
                        if ctx:
                            rich_groups.append(ctx)
                    if rich_groups:
                        llm_result = normalize_with_llm("dataset", rich_groups, dry_run=dry_run)
                        if llm_result.get("merge_decisions"):
                            _apply_llm_decisions(
                                "dataset", llm_result["merge_decisions"],
                                ds_groups, datasets, db,
                                result, dry_run,
                            )
                            result["datasets"]["llm_verified"] = True
                            logger.info("LLM 数据集归一化: %d 个合并决策", len(llm_result["merge_decisions"]))
                except Exception as e:
                    logger.warning("LLM 数据集归一化失败，回退到模糊匹配: %s", e)

            # --- 任务 LLM 归一化 ---
            if task_groups:
                try:
                    rich_groups = []
                    for group in task_groups:
                        ctx = build_task_context(group)
                        if ctx:
                            rich_groups.append(ctx)
                    if rich_groups:
                        llm_result = normalize_with_llm("task", rich_groups, dry_run=dry_run)
                        if llm_result.get("merge_decisions"):
                            _apply_llm_decisions(
                                "task", llm_result["merge_decisions"],
                                task_groups, tasks, db,
                                result, dry_run,
                            )
                            result["tasks"]["llm_verified"] = True
                            logger.info("LLM 任务归一化: %d 个合并决策", len(llm_result["merge_decisions"]))
                except Exception as e:
                    logger.warning("LLM 任务归一化失败，回退到模糊匹配: %s", e)

        # ================================================================
        # Phase 3: 对 LLM 未处理的组，回退到模糊匹配直接合并
        # ================================================================
        merge_funcs = {
            "method": merge_methods,
            "dataset": merge_datasets,
            "task": merge_tasks,
        }
        # entity_type (单数) → result key (复数)
        _plural_map = {"method": "methods", "dataset": "datasets", "task": "tasks"}

        for entity_type, groups, all_entities in [
            ("method", method_groups, methods),
            ("dataset", ds_groups, datasets),
            ("task", task_groups, tasks),
        ]:
            plural_key = _plural_map[entity_type]
            llm_verified = result[plural_key]["llm_verified"]
            if llm_verified:
                # LLM 已处理，跳过（合并已在 _apply_llm_decisions 中执行）
                continue

            # 回退: 模糊匹配直接合并（兼容旧逻辑）
            merge_fn = merge_funcs[entity_type]
            id_to_name = {e[0]: e[1] for e in all_entities}
            for group in groups:
                canonical = group[0]
                duplicates = group[1:]
                names = [id_to_name.get(mid, "?") for mid in group]
                result["details"].append({
                    "type": entity_type,
                    "canonical": names[0],
                    "aliases": names[1:],
                    "ids": group,
                })
                if not dry_run:
                    merge_fn(canonical, duplicates)
                    result[plural_key]["merged"] += len(duplicates)

        if not dry_run and not use_llm:
            db.commit()
            _mark_graph_dirty()

    except Exception as e:
        db.rollback()
        logger.error("归一化失败: %s", e, exc_info=True)
        result["error"] = str(e)
    finally:
        db.close()

    total_merged = (result["methods"]["merged"] +
                    result["datasets"]["merged"] +
                    result["tasks"]["merged"])
    logger.info("实体归一化完成: methods=%d groups, datasets=%d groups, tasks=%d groups, total_merged=%d",
                result["methods"]["groups"], result["datasets"]["groups"],
                result["tasks"]["groups"], total_merged)
    return result


def _apply_llm_decisions(
    entity_type: str,
    llm_decisions: List[Dict],
    fuzzy_groups: List[List[str]],
    all_entities: List[Tuple[str, str]],
    db,
    result: Dict,
    dry_run: bool,
):
    """
    将 LLM 的归一化决策应用到实际合并。

    LLM 决策格式: [{"canonical_name": "...", "duplicate_ids": [...], "reason": "..."}]
    """
    id_to_name = {e[0]: e[1] for e in all_entities}

    # 构建 fuzzy group 到 entity ID 集合的映射，用于查找 canonical_id
    id_to_group_idx = {}
    for gi, group in enumerate(fuzzy_groups):
        for eid in group:
            id_to_group_idx[eid] = gi

    _plural_map = {"method": "methods", "dataset": "datasets", "task": "tasks"}
    merge_fn = {"method": merge_methods, "dataset": merge_datasets, "task": merge_tasks}[entity_type]
    plural_key = _plural_map[entity_type]

    processed_ids = set()  # 避免重复合并

    for decision in llm_decisions:
        dup_ids = decision.get("duplicate_ids", [])
        canonical_name = decision.get("canonical_name", "")

        if len(dup_ids) < 1:
            continue

        # 过滤已处理的
        pending = [did for did in dup_ids if did not in processed_ids]
        if len(pending) < 1:
            continue

        # 找到 canonical_id：优先精确匹配 canonical_name，否则取第一个
        canonical_id = None
        for eid, ename in all_entities:
            if ename == canonical_name and eid in pending:
                canonical_id = eid
                break
        if canonical_id is None:
            # 取 pending 中名字最短的作为 canonical（简洁原则）
            pending_sorted = sorted(pending, key=lambda x: len(id_to_name.get(x, "")))
            canonical_id = pending_sorted[0]

        duplicates = [did for did in pending if did != canonical_id]

        if not duplicates:
            processed_ids.add(canonical_id)
            continue

        # 记录详情
        names_in_group = [id_to_name.get(eid, "?") for eid in ([canonical_id] + duplicates)]
        result["details"].append({
            "type": entity_type,
            "canonical": id_to_name.get(canonical_id, "?"),
            "aliases": names_in_group[1:],
            "ids": [canonical_id] + duplicates,
            "reason": decision.get("reason", ""),
            "llm_verified": True,
        })

        # 执行合并
        if not dry_run:
            merge_fn(canonical_id, duplicates)
            result[plural_key]["merged"] += len(duplicates)

        # 标记已处理
        for eid in ([canonical_id] + duplicates):
            processed_ids.add(eid)

    if not dry_run:
        db.commit()
        _mark_graph_dirty()


def _refine_dataset_groups_with_citations(
    groups: List[List[str]],
    all_datasets: List[Tuple[str, str]],
    db,
) -> List[List[str]]:
    """
    用引用指纹对数据集分组做二次合并：
    如果两个数据集组的引用指纹高度重叠，则认为它们是同一个数据集。
    """
    if len(groups) <= 1:
        return groups

    # 计算每组的引用指纹（合并组内所有成员的指纹）
    group_fingerprints = {}
    for i, group in enumerate(groups):
        fps = []
        for did in group:
            fp = _get_dataset_citation_fingerprint(did)
            if fp:
                fps.append(set(fp.split('|')))
        if fps:
            # 取交集（组内共同引用）
            common = fps[0]
            for s in fps[1:]:
                common = common & s
            group_fingerprints[i] = common

    # 检查组间引用重叠
    merged_indices = set()
    new_groups = []
    group_list = list(enumerate(groups))

    for i, group_i in group_list:
        if i in merged_indices:
            continue
        merged_group = list(group_i)
        fp_i = group_fingerprints.get(i, set())

        for j, group_j in group_list:
            if j <= i or j in merged_indices:
                continue
            fp_j = group_fingerprints.get(j, set())
            if fp_i and fp_j:
                overlap = len(fp_i & fp_j) / max(1, min(len(fp_i), len(fp_j)))
                if overlap >= 0.6:  # 60% 引用重叠 → 同一个数据集
                    merged_group.extend(group_j)
                    merged_indices.add(j)
                    fp_i = fp_i | fp_j  # 合并指纹

        new_groups.append(merged_group)

    return new_groups


# ============================================================
# 单实体查询（给 Agent Tool 用）
# ============================================================

def find_canonical_name(entity_type: str, name: str) -> Dict:
    """查找某实体名称的 canonical form（如果存在）"""
    db = SessionLocal()
    try:
        model_map = {
            "method": (KgMethod, KgMethod.method_id, KgMethod.name),
            "dataset": (KgDataset, KgDataset.dataset_id, KgDataset.name),
            "task": (KgTask, KgTask.task_id, KgTask.name),
        }
        entry = model_map.get(entity_type)
        if not entry:
            return {"found": False, "reason": f"unsupported type: {entity_type}"}

        model, id_col, name_col = entry
        all_entities = db.query(id_col, name_col).all()
        items = [(e[0], e[1]) for e in all_entities]

        # 精确匹配
        for eid, ename in items:
            if ename.lower().strip() == name.lower().strip():
                return {"found": True, "canonical_name": ename, "entity_id": eid, "exact_match": True}

        # 模糊匹配
        groups = find_similar_groups([(name, name)] + items, threshold=0.70)
        for group in groups:
            # 检查 name 是否在组内
            for eid, ename in items:
                if eid in group:
                    # 找组内名字最短/最规范的
                    canonical = min(
                        [(items[[e[0] for e in items].index(gid)][1], gid) for gid in group],
                        key=lambda x: len(x[0])
                    )
                    return {
                        "found": True,
                        "canonical_name": canonical[0],
                        "entity_id": canonical[1],
                        "exact_match": False,
                        "similar_to": ename,
                    }

        return {"found": False, "reason": "no similar entity found"}
    finally:
        db.close()


def get_entity_summary(entity_type: str) -> List[Dict]:
    """获取某类实体的摘要（含别名组）"""
    db = SessionLocal()
    try:
        model_map = {
            "method": (KgMethod, KgMethod.method_id, KgMethod.name, KgMethod.aliases),
            "dataset": (KgDataset, KgDataset.dataset_id, KgDataset.name, None),
            "task": (KgTask, KgTask.task_id, KgTask.name, None),
        }
        entry = model_map.get(entity_type)
        if not entry:
            return []

        model, id_col, name_col, alias_col = entry
        rows = db.query(id_col, name_col).all()

        # 找相似组
        items = [(r[0], r[1]) for r in rows]
        groups = find_similar_groups(items, threshold=0.70)

        # 标记已归入某个组的 ID
        grouped_ids = set()
        for g in groups:
            grouped_ids.update(g)

        result = []
        # 有变体的实体
        for group in groups:
            names = []
            for gid in group:
                for rid, rname in items:
                    if rid == gid:
                        names.append(rname)
                        break
            result.append({
                "canonical": names[0] if names else "?",
                "aliases": names[1:] if len(names) > 1 else [],
                "count": len(group),
            })

        # 独立实体
        for rid, rname in items:
            if rid not in grouped_ids:
                result.append({
                    "canonical": rname,
                    "aliases": [],
                    "count": 1,
                })

        result.sort(key=lambda x: -x["count"])
        return result
    finally:
        db.close()
