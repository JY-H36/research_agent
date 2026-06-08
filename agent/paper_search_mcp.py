"""
联网论文搜索 — MCP 工具模块

演示 MCP (Model Context Protocol) 的核心概念：
  1. Tool 注册：每个 Tool 有 name / description / inputSchema (JSON Schema)
  2. Tool Handler：接收标准化参数 → 执行业务逻辑 → 返回标准化结果
  3. 分离关注点：Tool 定义与 Agent 编排解耦

如果以后要拆分为独立 MCP Server 进程：
  - 本文件的 tool 定义可直接导出给 MCP Server
  - Agent 通过 stdio/HTTP JSON-RPC 调用，而非 Python import

当前阶段：与 Agent 同进程运行，Python 直接调用（零网络开销）
"""
import json
import logging
from typing import Dict, List, Any

from config import RETRIEVAL_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)


# ═══════════════════════════════════════════════════════════
# MCP Tool Schema 定义
# 按照 MCP 规范：每个 Tool = name + description + inputSchema
# inputSchema 使用 JSON Schema 格式描述参数
# ═══════════════════════════════════════════════════════════

PAPER_SEARCH_TOOL_SCHEMA: Dict[str, Any] = {
    "name": "search_papers_online",
    "description": (
        "从互联网学术数据库（arXiv + Semantic Scholar + OpenAlex）检索论文。"
        "当用户想要查找最新论文、了解某个方向的研究现状、"
        "或知识库中缺少相关内容时使用此工具。"
        "返回论文的标题、作者、年份、摘要（中英文）、下载链接等信息。"
    ),
    "inputSchema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "搜索关键词，英文效果更好，如 'audio deepfake detection wav2vec'",
            },
            "limit": {
                "type": "integer",
                "description": "返回论文篇数。务必根据用户明确指定的数量设置，如用户说'5篇'则设为5。用户未指定时默认5篇",
                "default": 5,
            },
            "source": {
                "type": "string",
                "enum": ["all", "arxiv", "semantic_scholar", "openalex"],
                "description": "数据源: all=全部, arxiv=arXiv预印本, semantic_scholar=Semantic Scholar, openalex=OpenAlex",
                "default": "all",
            },
            "since_year": {
                "type": "integer",
                "description": "论文起始年份，如 2022。不指定则自动从 query 推断或默认近3年",
            },
        },
        "required": ["query"],
    },
}


# ═══════════════════════════════════════════════════════════
# MCP Tool Handler
# 对应 MCP 协议的 tools/call 请求
# 输入：arguments (dict) — 与 inputSchema 对应的参数
# 输出：content (list) + structuredContent — MCP 标准响应格式
# ═══════════════════════════════════════════════════════════

def handle_search_papers(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    处理 search_papers_online 的 MCP tools/call 请求

    MCP 标准返回格式:
    {
        "content": [{"type": "text", "text": "..."}],
        "structuredContent": { ... },
        "isError": false
    }
    """
    query = arguments.get("query", "")
    limit = arguments.get("limit", 5)
    source = arguments.get("source", "all")
    since_year = arguments.get("since_year")

    if not query:
        return _mcp_error("查询内容不能为空")

    try:
        from knowledge_base.paper_search import search_papers, translate_abstracts

        logger.info("[MCP] search_papers_online: query='%s', limit=%d, source=%s",
                   query[:100], limit, source)

        papers = search_papers(query=query, limit=limit, source=source,
                              since_year=since_year)

        if not papers:
            return _mcp_success(
                f"未在学术数据库中找到与「{query}」相关的论文。请尝试更换关键词。",
                {"query": query, "papers": [], "count": 0}
            )

        # 翻译摘要
        papers = translate_abstracts(papers)

        # 构建 LLM 可读的简要文本（保持简洁，不撑爆上下文）
        llm_text = _format_for_llm(papers, query)

        # 结构化数据（供前端渲染论文卡片）
        compact = _compact_papers(papers)

        return _mcp_success(llm_text, {
            "query": query,
            "papers": compact,
            "count": len(compact),
            "result_type": "paper_search",
        })

    except Exception as e:
        logger.error("[MCP] search_papers_online 异常: %s", e, exc_info=True)
        return _mcp_error(str(e))


# ═══════════════════════════════════════════════════════════
# 辅助函数
# ═══════════════════════════════════════════════════════════

def _mcp_success(text: str, structured: dict) -> dict:
    """构建 MCP 成功响应"""
    return {
        "success": True,
        "result": text,
        "content": [{"type": "text", "text": text}],
        "structuredContent": structured,
        "isError": False,
        # 向后兼容现有 agent_core.py 的字段
        "papers": structured.get("papers", []),
        "result_type": structured.get("result_type", ""),
    }


def _mcp_error(message: str) -> dict:
    """构建 MCP 错误响应"""
    return {
        "success": False,
        "result": None,
        "content": [{"type": "text", "text": message}],
        "structuredContent": {"error": message},
        "isError": True,
        "error": message,
        "papers": [],
    }


def _format_for_llm(papers: List[Dict], query: str) -> str:
    """为 LLM 生成简洁的论文摘要文本"""
    lines = [f"检索到 {len(papers)} 篇与「{query}」相关的论文。请对以下论文进行简要总结分析："]
    for i, p in enumerate(papers):
        authors_str = ", ".join(p.get("authors", [])[:3])
        if len(p.get("authors", [])) > 3:
            authors_str += " et al."
        year_str = f" ({p['year']})" if p.get("year") else ""
        venue_str = f" — {p.get('venue', '')}" if p.get("venue") else ""
        citations = f" [引用:{p['citation_count']}]" if p.get("citation_count") else ""
        first_sentence = ""
        if p.get("abstract"):
            first_sentence = p["abstract"].split(". ")[0][:200]
        lines.append(
            f"{i + 1}. {p['title']}{year_str}{venue_str}{citations}\n"
            f"   作者: {authors_str}"
            + (f"\n   摘要首句: {first_sentence}." if first_sentence else "")
        )
    return "\n".join(lines)


def _compact_papers(papers: List[Dict]) -> List[Dict]:
    """压缩论文数据（只保留前端卡片需要的字段）"""
    compact = []
    for p in papers:
        compact.append({
            "title": p.get("title", ""),
            "authors": p.get("authors", [])[:8],
            "year": p.get("year", ""),
            "venue": p.get("venue", ""),
            "abstract": (p.get("abstract", "") or "")[:600],
            "abstract_cn": (p.get("abstract_cn", "") or "")[:600],
            "pdf_url": p.get("pdf_url", ""),
            "url": p.get("url", ""),
            "arxiv_id": p.get("arxiv_id", ""),
            "citation_count": p.get("citation_count", 0) or 0,
            "source": p.get("source", ""),
            "paper_id": p.get("paper_id", ""),
        })
    return compact


# ═══════════════════════════════════════════════════════════
# MCP-style 导出接口
# 供 Agent 直接调用，也方便以后迁移到独立 MCP Server 进程
# ═══════════════════════════════════════════════════════════

def get_tool_definition() -> Dict[str, Any]:
    """
    返回 MCP 标准的 Tool 定义
    对应 MCP 协议的 tools/list 响应中的单个 tool 条目
    """
    return PAPER_SEARCH_TOOL_SCHEMA


def call_tool(tool_name: str, arguments: Dict[str, Any]) -> Dict[str, Any]:
    """
    MCP tools/call 的入口
    根据 tool_name 路由到对应的 handler
    """
    if tool_name == "search_papers_online":
        return handle_search_papers(arguments)
    else:
        return _mcp_error(f"未知工具: {tool_name}")


def list_tools() -> List[Dict[str, Any]]:
    """
    返回所有已注册的 MCP Tool 列表
    对应 MCP 协议的 tools/list 响应
    """
    return [PAPER_SEARCH_TOOL_SCHEMA]


# ═══════════════════════════════════════════════════════════
# 论文持久化（与 tools.py 共享同一个 encode/decode 逻辑）
# ═══════════════════════════════════════════════════════════

PAPERS_JSON_MARKER = "\n\n<!--PAPERS_JSON-->\n"


def encode_papers_in_message(content: str, papers: List[Dict]) -> str:
    """将论文元数据编码到消息内容末尾，实现持久化"""
    if not papers:
        return content
    compact = _compact_papers(papers)
    return content + PAPERS_JSON_MARKER + json.dumps(compact, ensure_ascii=False)


def decode_papers_from_message(content: str) -> tuple:
    """从消息内容中提取论文元数据，返回 (clean_content, papers_list)"""
    if PAPERS_JSON_MARKER not in content:
        return content, []
    clean, _, json_str = content.partition(PAPERS_JSON_MARKER)
    try:
        papers = json.loads(json_str.strip())
        return clean, papers
    except json.JSONDecodeError:
        return content, []
