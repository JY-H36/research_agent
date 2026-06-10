"""
Agent 工具: query_paper_graph
从知识图谱中结构化查询论文、作者、方法、数据集、任务及它们之间的关系
"""
from typing import Dict

from knowledge_graph.graph_query import execute_query, QUERY_HANDLERS
from knowledge_graph.graph_store import get_graph_stats
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 工具定义（OpenAI function calling 格式）
# ============================================================
QUERY_PAPER_GRAPH_TOOL_DEF = {
    "type": "function",
    "function": {
        "name": "query_paper_graph",
        "description": (
            "从知识图谱中结构化查询论文、方法、数据集、作者、研究任务及其关系。"
            "适用于以下场景：\n"
            "- 查找使用了特定方法/模型的论文（如 wav2vec、AASIST）\n"
            "- 查找在特定数据集上评测的论文（如 ASVspoof 2019）\n"
            "- 查找某位作者的论文\n"
            "- 查看某研究任务的论文分布\n"
            "- 查看数据集上的性能排名\n"
            "- 发现哪些方法经常组合使用\n"
            "- 找与某论文最相关的其他论文\n"
            "- 发现可能的研究空白"
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query_type": {
                    "type": "string",
                    "enum": list(QUERY_HANDLERS.keys()),
                    "description": (
                        "查询类型:\n"
                        "- papers_by_method: 哪些论文用了某方法\n"
                        "- papers_by_dataset: 哪些论文在某数据集上评测\n"
                        "- papers_by_author: 某作者的论文\n"
                        "- papers_by_task: 某研究任务的论文\n"
                        "- method_evolution: 方法演进链\n"
                        "- performance_ranking: 数据集上的性能排名\n"
                        "- related_papers: 找与某论文最相关的其他论文\n"
                        "- method_co_occurrence: 经常组合使用的方法对\n"
                        "- dataset_co_occurrence: 经常一起出现的数据集对\n"
                        "- research_gap: 发现可能的研究空白"
                    ),
                },
                "entity_name": {
                    "type": "string",
                    "description": "查询的实体名称（papers_by_*, performance_ranking, method_evolution, related_papers 需要）",
                },
                "metric_name": {
                    "type": "string",
                    "description": "指标名称（仅 performance_ranking 需要，如 EER, min t-DCF, accuracy）。默认为 EER",
                    "default": "EER",
                },
                "limit": {
                    "type": "integer",
                    "description": "返回结果数量上限，默认 10",
                    "default": 10,
                },
            },
            "required": ["query_type"]
        }
    }
}


# ============================================================
# 工具执行
# ============================================================
def execute_query_paper_graph(args: Dict) -> Dict:
    """
    执行 query_paper_graph 工具调用
    """
    query_type = args.get("query_type", "")
    entity_name = args.get("entity_name", "")
    limit = args.get("limit", 10)
    metric_name = args.get("metric_name", "EER")

    logger.info("query_paper_graph: type=%s, entity='%s', limit=%d", query_type, entity_name[:80] if entity_name else "", limit)

    # 构建参数
    params = {"limit": limit}

    if query_type in ("papers_by_method", "papers_by_dataset", "papers_by_author",
                       "papers_by_task", "performance_ranking", "related_papers",
                       "method_evolution"):
        if not entity_name:
            return {
                "success": False,
                "result": f"查询类型 '{query_type}' 需要提供 entity_name 参数",
                "error": "缺少 entity_name",
            }
        if query_type == "method_evolution":
            params["method_name"] = entity_name
        elif query_type == "performance_ranking":
            params["dataset_name"] = entity_name
            params["metric_name"] = metric_name
        elif query_type == "related_papers":
            params["paper_title_or_id"] = entity_name
        elif query_type == "papers_by_method":
            params["method_name"] = entity_name
        elif query_type == "papers_by_dataset":
            params["dataset_name"] = entity_name
        elif query_type == "papers_by_author":
            params["author_name"] = entity_name
        elif query_type == "papers_by_task":
            params["task_name"] = entity_name

    # 执行查询
    result = execute_query(query_type, params)

    # 格式化结果文本
    if result.get("success"):
        data = result.get("data", [])
        formatted_text = _format_result(query_type, data, entity_name)
        return {
            "success": True,
            "result": formatted_text,
            "data": data,
            "result_type": query_type,
        }
    else:
        return {
            "success": False,
            "result": f"图谱查询失败: {result.get('error', '未知错误')}",
            "error": result.get("error"),
        }


def _format_result(query_type: str, data: list, entity_name: str = "") -> str:
    """将查询结果格式化为 LLM 可读的文本"""
    if not data:
        return f"图谱查询 '{query_type}' 未找到相关结果。"

    if query_type == "papers_by_method":
        lines = [f"知识库中使用了「{entity_name}」方法的 {len(data)} 篇论文:"]
        for d in data:
            role_str = f" (角色: {d.get('method_role')})" if d.get("method_role") else ""
            variant_str = f" (变体: {d.get('method_variant')})" if d.get("method_variant") else ""
            year_str = f" ({d.get('year')})" if d.get("year") else ""
            lines.append(f"- {d.get('title')}{year_str}{role_str}{variant_str}")
        return "\n".join(lines)

    elif query_type == "papers_by_dataset":
        lines = [f"知识库中在「{entity_name}」数据集上评测的 {len(data)} 篇论文:"]
        for d in data:
            year_str = f" ({d.get('year')})" if d.get("year") else ""
            task_str = f" [任务: {d.get('task')}]" if d.get("task") else ""
            lines.append(f"- {d.get('title')}{year_str}{task_str}")
        return "\n".join(lines)

    elif query_type == "papers_by_author":
        lines = [f"知识库中作者「{entity_name}」的 {len(data)} 篇论文:"]
        for d in data:
            year_str = f" ({d.get('year')})" if d.get("year") else ""
            order_str = f" (第{d.get('author_order')}作者)" if d.get("author_order") else ""
            lines.append(f"- {d.get('title')}{year_str}{order_str}")
        return "\n".join(lines)

    elif query_type == "papers_by_task":
        lines = [f"知识库中属于「{entity_name}」研究任务的 {len(data)} 篇论文:"]
        for d in data:
            year_str = f" ({d.get('year')})" if d.get("year") else ""
            lines.append(f"- {d.get('title')}{year_str}")
        return "\n".join(lines)

    elif query_type == "performance_ranking":
        lines = [f"数据集「{entity_name}」上按 EER 排名的论文:"]
        for i, d in enumerate(data):
            val = d.get("value", "N/A")
            cond = f" ({d.get('condition')})" if d.get("condition") else ""
            lines.append(f"{i+1}. {d.get('title')} — {d.get('metric_name')}={val}{cond}")
        return "\n".join(lines)

    elif query_type == "related_papers":
        lines = [f"与「{entity_name}」在知识图谱中相关的 {len(data)} 篇论文:"]
        for d in data:
            rel = d.get("relations", [])
            rel_str = f" [关系: {', '.join(rel)}]" if rel else ""
            lines.append(f"- {d.get('title')} (距离: {d.get('distance')} 跳){rel_str}")
        return "\n".join(lines)

    elif query_type == "method_co_occurrence":
        lines = [f"知识库中最常组合使用的方法对 (共 {len(data)} 对):"]
        for d in data:
            lines.append(f"- {d.get('method_a')} + {d.get('method_b')}: {d.get('co_occurrence_count')} 篇论文")
        return "\n".join(lines)

    elif query_type == "dataset_co_occurrence":
        lines = [f"知识库中最常同时出现的数据集对 (共 {len(data)} 对):"]
        for d in data:
            lines.append(f"- {d.get('dataset_a')} + {d.get('dataset_b')}: {d.get('co_occurrence_count')} 篇论文")
        return "\n".join(lines)

    elif query_type == "method_evolution":
        lines = [f"「{entity_name}」的方法演进链:"]
        for chain in data:
            nodes = " → ".join([n.get("name", "?") for n in chain])
            lines.append(f"- {nodes}")
        return "\n".join(lines)

    elif query_type == "research_gap":
        lines = [f"知识图谱中发现的 {len(data)} 个潜在研究空白:"]
        for i, d in enumerate(data):
            lines.append(f"{i+1}. {d.get('insight', '')}")
        return "\n".join(lines)

    else:
        return f"图谱查询结果 ({len(data)} 条记录)。"


def get_tool_result_summary(result: Dict) -> str:
    """生成工具执行结果的简要摘要"""
    if not result.get("success"):
        return f"图谱查询失败: {result.get('error', '未知错误')}"
    data = result.get("data", [])
    result_type = result.get("result_type", "?")
    return f"图谱查询完成: {result_type} → {len(data) if isinstance(data, list) else '?'} 条结果"
