"""
图谱查询模块
提供 10 种结构化查询类型，结合 SQL JOIN 和 NetworkX 图遍历
"""
import json
from typing import Dict, List, Optional

from database.connection import SessionLocal
from knowledge_graph.models import (
    KgPaper, KgAuthor, KgDataset, KgMethod, KgMetric, KgTask, KgVenue,
    KgPaperAuthor, KgPaperCites, KgPaperUsesMethod,
    KgPaperEvaluatesOn, KgPaperTrainsOn, KgPaperBelongsToTask,
    KgMethodImprovesMethod, KgPaperPublishedIn, KgPaperReportsMetric,
)
from knowledge_graph.graph_store import get_nx_graph
from utils.logger import get_logger

logger = get_logger(__name__)


def _success(data, result_type: str = "") -> Dict:
    return {"success": True, "data": data, "result_type": result_type}


def _error(msg: str) -> Dict:
    return {"success": False, "data": None, "error": msg}


# ============================================================
# Q1: 哪些论文用了某方法
# ============================================================
def query_papers_by_method(method_name: str, limit: int = 20) -> Dict:
    """查找使用了指定方法的所有论文"""
    db = SessionLocal()
    try:
        results = (
            db.query(KgPaper, KgPaperUsesMethod)
            .join(KgPaperUsesMethod, KgPaperUsesMethod.paper_id == KgPaper.paper_id)
            .join(KgMethod, KgMethod.method_id == KgPaperUsesMethod.method_id)
            .filter(KgMethod.name.contains(method_name))
            .order_by(KgPaper.year.desc())
            .limit(limit)
            .all()
        )
        data = []
        for paper, rel in results:
            data.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "method_role": rel.role,
                "method_variant": rel.variant,
                "citation_count": paper.citation_count or 0,
            })
        logger.info("papers_by_method('%s'): %d 结果", method_name[:60], len(data))
        return _success(data, "papers_by_method")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# Q2: 哪些论文在某数据集上评测
# ============================================================
def query_papers_by_dataset(dataset_name: str, limit: int = 20) -> Dict:
    """查找在指定数据集上评测的所有论文"""
    db = SessionLocal()
    try:
        results = (
            db.query(KgPaper, KgPaperEvaluatesOn)
            .join(KgPaperEvaluatesOn, KgPaperEvaluatesOn.paper_id == KgPaper.paper_id)
            .join(KgDataset, KgDataset.dataset_id == KgPaperEvaluatesOn.dataset_id)
            .filter(KgDataset.name.contains(dataset_name))
            .order_by(KgPaper.year.desc())
            .limit(limit)
            .all()
        )
        data = []
        for paper, rel in results:
            metrics_display = rel.metrics or {}
            if isinstance(metrics_display, str):
                metrics_display = json.loads(metrics_display)
            data.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "task": rel.task,
                "split": rel.split,
                "metrics": metrics_display,
            })
        logger.info("papers_by_dataset('%s'): %d 结果", dataset_name[:60], len(data))
        return _success(data, "papers_by_dataset")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# Q3: 某作者的论文
# ============================================================
def query_papers_by_author(author_name: str, limit: int = 20) -> Dict:
    """查找某作者发表的所有论文"""
    db = SessionLocal()
    try:
        results = (
            db.query(KgPaper, KgPaperAuthor)
            .join(KgPaperAuthor, KgPaperAuthor.paper_id == KgPaper.paper_id)
            .join(KgAuthor, KgAuthor.author_id == KgPaperAuthor.author_id)
            .filter(KgAuthor.name.contains(author_name))
            .order_by(KgPaper.year.desc())
            .limit(limit)
            .all()
        )
        data = []
        for paper, rel in results:
            data.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "author_order": rel.author_order,
                "is_corresponding": rel.is_corresponding,
            })
        logger.info("papers_by_author('%s'): %d 结果", author_name[:60], len(data))
        return _success(data, "papers_by_author")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# Q4: 某研究任务的论文
# ============================================================
def query_papers_by_task(task_name: str, limit: int = 20) -> Dict:
    """查找属于某研究任务的所有论文"""
    db = SessionLocal()
    try:
        results = (
            db.query(KgPaper, KgPaperBelongsToTask)
            .join(KgPaperBelongsToTask, KgPaperBelongsToTask.paper_id == KgPaper.paper_id)
            .join(KgTask, KgTask.task_id == KgPaperBelongsToTask.task_id)
            .filter(KgTask.name.contains(task_name))
            .order_by(KgPaper.year.desc())
            .limit(limit)
            .all()
        )
        data = []
        for paper, rel in results:
            data.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "is_primary": rel.is_primary,
            })
        logger.info("papers_by_task('%s'): %d 结果", task_name[:60], len(data))
        return _success(data, "papers_by_task")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# Q5: 方法演进链
# ============================================================
def query_method_evolution(method_name: str, max_depth: int = 4) -> Dict:
    """沿 IMPROVES_UPON 边追溯方法演进链（BFS）"""
    G = get_nx_graph()

    # 找到起始 method 节点
    start_nodes = []
    for node_id, attr in G.nodes(data=True):
        if attr.get("type") == "method" and method_name.lower() in attr.get("label", "").lower():
            start_nodes.append(node_id)

    if not start_nodes:
        return _success([], "method_evolution")

    # BFS 收集 IMPROVES_UPON 链
    chains = []
    visited = set()

    for start in start_nodes:
        if start in visited:
            continue
        # 从 start 出发，沿 IMPROVES_UPON 边向前追溯
        chain = []
        current_layer = {start}
        for depth in range(max_depth):
            next_layer = set()
            for node in current_layer:
                if node not in visited:
                    visited.add(node)
                    label = G.nodes[node].get("label", node)
                    chain.append({"node_id": node, "name": label, "depth": depth})
                    # 找这个节点改进的方法（出边）
                    for _, target, data in G.out_edges(node, data=True):
                        if data.get("relation") == "IMPROVES_UPON":
                            next_layer.add(target)
                    # 也找谁改进了这个节点（入边）
                    for source, _, data in G.in_edges(node, data=True):
                        if data.get("relation") == "IMPROVES_UPON":
                            next_layer.add(source)
            if not next_layer:
                break
            current_layer = next_layer
        if chain:
            chains.append(chain)

    logger.info("method_evolution('%s'): %d 链", method_name[:60], len(chains))
    return _success(chains, "method_evolution")


# ============================================================
# Q6: 数据集上性能排名
# ============================================================
def query_performance_ranking(dataset_name: str, metric_name: str = "EER",
                              limit: int = 20) -> Dict:
    """在指定数据集上按指定指标排名论文"""
    db = SessionLocal()
    try:
        # 先找到 dataset
        datasets = db.query(KgDataset).filter(KgDataset.name.contains(dataset_name)).all()
        if not datasets:
            return _success([], "performance_ranking")

        dataset_ids = [d.dataset_id for d in datasets]

        # 找在这个数据集上有指标值的论文
        results = (
            db.query(KgPaper, KgPaperReportsMetric, KgMetric)
            .join(KgPaperReportsMetric, KgPaperReportsMetric.paper_id == KgPaper.paper_id)
            .join(KgMetric, KgMetric.metric_id == KgPaperReportsMetric.metric_id)
            .filter(
                KgPaperReportsMetric.dataset_id.in_(dataset_ids),
                KgMetric.name.contains(metric_name),
            )
            .order_by(KgPaperReportsMetric.value.asc())
            .limit(limit)
            .all()
        )

        data = []
        for paper, rel, metric in results:
            data.append({
                "paper_id": paper.paper_id,
                "title": paper.title,
                "year": paper.year,
                "metric_name": metric.name,
                "value": rel.value,
                "condition": rel.condition,
            })

        # 按值排序（EER 越低越好）
        metric_dir = "lower_better"
        if data:
            data.sort(key=lambda x: x.get("value") or float('inf'),
                      reverse=(metric_dir == "higher_better"))

        logger.info("performance_ranking('%s', '%s'): %d 结果", dataset_name[:40], metric_name, len(data))
        return _success(data, "performance_ranking")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# Q7: 找相关论文（通过图谱路径）
# ============================================================
def query_related_papers(paper_title_or_id: str, max_hops: int = 3, limit: int = 20) -> Dict:
    """通过图谱多跳路径找与指定论文最相关的其他论文"""
    G = get_nx_graph()

    # 找到目标论文节点
    target_nodes = []
    for node_id, attr in G.nodes(data=True):
        if attr.get("type") != "paper":
            continue
        if paper_title_or_id in node_id or paper_title_or_id.lower() in attr.get("label", "").lower():
            target_nodes.append(node_id)

    if not target_nodes:
        return _success([], "related_papers")

    target = target_nodes[0]
    related = {}

    # 从 target 出发，BFS 收集可达论文节点
    from collections import deque
    queue = deque([(target, 0)])
    visited = {target: 0}

    while queue:
        current, dist = queue.popleft()
        if dist >= max_hops:
            continue

        for neighbor in list(G.predecessors(current)) + list(G.successors(current)):
            if neighbor not in visited:
                visited[neighbor] = dist + 1
                queue.append((neighbor, dist + 1))

                attr = G.nodes.get(neighbor, {})
                if attr.get("type") == "paper" and neighbor != target:
                    # 收集边信息
                    edges = []
                    if G.has_edge(current, neighbor):
                        for k, edata in G[current][neighbor].items():
                            edges.append(edata.get("relation", ""))
                    if G.has_edge(neighbor, current):
                        for k, edata in G[neighbor][current].items():
                            edges.append(edata.get("relation", ""))

                    related[neighbor] = {
                        "paper_id": neighbor.split(":", 1)[1],
                        "title": attr.get("label", ""),
                        "year": attr.get("year"),
                        "distance": dist + 1,
                        "relations": list(set(edges)),
                    }

    # 按距离排序，距离小于等于2优先
    sorted_related = sorted(related.values(), key=lambda x: x["distance"])[:limit]
    logger.info("related_papers('%s'): %d 相关论文", paper_title_or_id[:60], len(sorted_related))
    return _success(sorted_related, "related_papers")


# ============================================================
# Q8: 方法共现分析
# ============================================================
def query_method_co_occurrence(min_co_occurrence: int = 1, limit: int = 20) -> Dict:
    """找出经常在同一篇论文中一起使用的方法对"""
    db = SessionLocal()
    try:
        # 获取所有 paper-method 关系
        rows = db.query(
            KgPaperUsesMethod.paper_id, KgPaperUsesMethod.method_id
        ).all()

        # 按 paper 分组
        paper_methods = {}
        for paper_id, method_id in rows:
            paper_methods.setdefault(paper_id, set()).add(method_id)

        # 统计方法对共现
        from collections import Counter
        pairs = Counter()
        method_name_map = {}
        for methods in paper_methods.values():
            if len(methods) < 2:
                continue
            m_list = list(methods)
            for i in range(len(m_list)):
                for j in range(i + 1, len(m_list)):
                    pair = tuple(sorted([m_list[i], m_list[j]]))
                    pairs[pair] += 1

        # 获取方法名称
        all_method_ids = set()
        for a, b in pairs:
            all_method_ids.add(a)
            all_method_ids.add(b)
        methods_map = {}
        if all_method_ids:
            methods = db.query(KgMethod).filter(KgMethod.method_id.in_(all_method_ids)).all()
            methods_map = {m.method_id: m.name for m in methods}

        data = []
        for (a, b), count in pairs.most_common(limit):
            if count >= min_co_occurrence:
                data.append({
                    "method_a": methods_map.get(a, a),
                    "method_b": methods_map.get(b, b),
                    "co_occurrence_count": count,
                })

        logger.info("method_co_occurrence: %d 对 (min=%d)", len(data), min_co_occurrence)
        return _success(data, "method_co_occurrence")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# Q9: 数据集共现分析
# ============================================================
def query_dataset_co_occurrence(min_co_occurrence: int = 1, limit: int = 20) -> Dict:
    """找出经常在同一篇论文中一起出现的数据集对"""
    db = SessionLocal()
    try:
        rows = db.query(
            KgPaperEvaluatesOn.paper_id, KgPaperEvaluatesOn.dataset_id
        ).all()

        paper_datasets = {}
        for paper_id, dataset_id in rows:
            paper_datasets.setdefault(paper_id, set()).add(dataset_id)

        from collections import Counter
        pairs = Counter()
        for datasets in paper_datasets.values():
            if len(datasets) < 2:
                continue
            d_list = list(datasets)
            for i in range(len(d_list)):
                for j in range(i + 1, len(d_list)):
                    pair = tuple(sorted([d_list[i], d_list[j]]))
                    pairs[pair] += 1

        all_ds_ids = set()
        for a, b in pairs:
            all_ds_ids.add(a)
            all_ds_ids.add(b)
        ds_map = {}
        if all_ds_ids:
            datasets = db.query(KgDataset).filter(KgDataset.dataset_id.in_(all_ds_ids)).all()
            ds_map = {d.dataset_id: d.name for d in datasets}

        data = []
        for (a, b), count in pairs.most_common(limit):
            if count >= min_co_occurrence:
                data.append({
                    "dataset_a": ds_map.get(a, a),
                    "dataset_b": ds_map.get(b, b),
                    "co_occurrence_count": count,
                })

        logger.info("dataset_co_occurrence: %d 对", len(data))
        return _success(data, "dataset_co_occurrence")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# Q10: 研究空白发现（结构洞检测）
# ============================================================
def query_research_gap(scope: str = "", limit: int = 10) -> Dict:
    """
    发现可能的研究空白：找出高频方法-数据集对中缺失的边。
    即：某方法在多篇论文中被使用，某数据集也在多篇论文中被评测，
    但没有任何一篇论文同时使用该方法并在该数据集上评测。
    """
    db = SessionLocal()
    try:
        # 获取所有 USES_METHOD
        methods_usage = db.query(KgPaperUsesMethod).all()
        # 获取所有 EVALUATES_ON
        evals = db.query(KgPaperEvaluatesOn).all()

        # 统计每种方法被多少篇论文使用
        from collections import Counter, defaultdict
        method_paper_count = Counter()
        dataset_paper_count = Counter()
        method_papers = defaultdict(set)
        dataset_papers = defaultdict(set)

        for r in methods_usage:
            method_paper_count[r.method_id] += 1
            method_papers[r.method_id].add(r.paper_id)

        for r in evals:
            dataset_paper_count[r.dataset_id] += 1
            dataset_papers[r.dataset_id].add(r.paper_id)

        # 找高频方法和高频数据集
        top_methods = [mid for mid, cnt in method_paper_count.most_common(30) if cnt >= 2]
        top_datasets = [did for did, cnt in dataset_paper_count.most_common(30) if cnt >= 2]

        # 对于每个 (method, dataset) 组合，检查是否有共同论文
        gaps = []
        for mid in top_methods:
            m_papers = method_papers.get(mid, set())
            for did in top_datasets:
                d_papers = dataset_papers.get(did, set())
                common = m_papers & d_papers
                if not common:
                    # 这是结构洞：高频方法+高频数据集但没有共同论文
                    gaps.append({
                        "method_id": mid,
                        "dataset_id": did,
                        "method_paper_count": method_paper_count[mid],
                        "dataset_paper_count": dataset_paper_count[did],
                        "gap_score": method_paper_count[mid] + dataset_paper_count[did],
                    })

        gaps.sort(key=lambda x: x["gap_score"], reverse=True)

        # 填充名称
        all_mids = {g["method_id"] for g in gaps[:limit]}
        all_dids = {g["dataset_id"] for g in gaps[:limit]}
        m_names = {}
        d_names = {}
        if all_mids:
            methods = db.query(KgMethod).filter(KgMethod.method_id.in_(all_mids)).all()
            m_names = {m.method_id: m.name for m in methods}
        if all_dids:
            datasets = db.query(KgDataset).filter(KgDataset.dataset_id.in_(all_dids)).all()
            d_names = {d.dataset_id: d.name for d in datasets}

        data = []
        for g in gaps[:limit]:
            data.append({
                "method_name": m_names.get(g["method_id"], g["method_id"]),
                "dataset_name": d_names.get(g["dataset_id"], g["dataset_id"]),
                "method_popularity": g["method_paper_count"],
                "dataset_popularity": g["dataset_paper_count"],
                "gap_score": g["gap_score"],
                "insight": (
                    f"方法「{m_names.get(g['method_id'], '?')}」在 {g['method_paper_count']} 篇论文中被使用，"
                    f"数据集「{d_names.get(g['dataset_id'], '?')}」在 {g['dataset_paper_count']} 篇论文中被评测，"
                    f"但尚无论文同时包含两者——这可能是一个研究空白。"
                ),
            })

        logger.info("research_gap: %d 个空白发现", len(data))
        return _success(data, "research_gap")
    except Exception as e:
        return _error(str(e))
    finally:
        db.close()


# ============================================================
# 分发函数
# ============================================================
QUERY_HANDLERS = {
    "papers_by_method": query_papers_by_method,
    "papers_by_dataset": query_papers_by_dataset,
    "papers_by_author": query_papers_by_author,
    "papers_by_task": query_papers_by_task,
    "method_evolution": query_method_evolution,
    "performance_ranking": query_performance_ranking,
    "related_papers": query_related_papers,
    "method_co_occurrence": query_method_co_occurrence,
    "dataset_co_occurrence": query_dataset_co_occurrence,
    "research_gap": query_research_gap,
}


def execute_query(query_type: str, params: Dict) -> Dict:
    """执行图谱查询的分发函数"""
    handler = QUERY_HANDLERS.get(query_type)
    if handler is None:
        return _error(f"不支持的查询类型: {query_type}。"
                      f"支持的类型: {list(QUERY_HANDLERS.keys())}")

    try:
        return handler(**params)
    except TypeError as e:
        return _error(f"查询参数错误: {e}")
    except Exception as e:
        logger.error("图谱查询 '%s' 异常: %s", query_type, e, exc_info=True)
        return _error(str(e))
