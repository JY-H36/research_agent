"""
Agent 工具定义与执行
"""
import logging
from typing import Dict, List

from knowledge_base.retriever import get_retriever
from config import RETRIEVAL_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)


# ============================================================
# 工具定义（OpenAI function calling 格式）
# ============================================================
TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": (
                "从知识库中检索与用户查询相关的论文章节片段。"
                "当用户提出科研问题、询问某篇论文的内容、或需要查找特定方法/技术时使用此工具。"
                "支持中英文混合检索。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "检索查询文本，可以是中文或英文，尽量提取用户问题中的核心概念作为查询词"
                    },
                    "top_k": {
                        "type": "integer",
                        "description": f"返回的结果数量，默认为 {RETRIEVAL_TOP_K}",
                        "default": RETRIEVAL_TOP_K,
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "search_papers_online",
            "description": (
                "从互联网学术数据库（arXiv + Semantic Scholar + OpenAlex）检索论文。"
                "当用户想要查找最新论文、了解某个方向的研究现状、"
                "或知识库中缺少相关内容时使用此工具。"
                "返回论文的标题、作者、年份、摘要（中英文）、下载链接等信息。"
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "搜索关键词，英文效果更好，如 'audio deepfake detection wav2vec'"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "返回论文篇数。务必根据用户明确指定的数量设置，如用户说'5篇'则设为5。用户未指定时默认5篇",
                        "default": 5,
                    },
                    "source": {
                        "type": "string",
                        "enum": ["all", "arxiv", "semantic_scholar", "openalex"],
                        "description": "数据源: all=全部, arxiv=arXiv, semantic_scholar=Semantic Scholar, openalex=OpenAlex",
                        "default": "all",
                    },
                    "since_year": {
                        "type": "integer",
                        "description": "论文起始年份，如 2022",
                    },
                },
                "required": ["query"]
            }
        }
    }
]


# ============================================================
# 工具执行
# ============================================================
def execute_tool(tool_name: str, arguments: Dict) -> Dict:
    """
    执行指定工具并返回结果
    """
    logger.debug("执行工具: %s, args=%s", tool_name, arguments)

    if tool_name == "search_knowledge_base":
        return _search_knowledge_base(arguments)
    elif tool_name == "search_papers_online":
        return _search_papers_online(arguments)
    else:
        logger.warning("未知工具调用: %s", tool_name)
        return {"success": False, "result": None, "error": f"未知工具: {tool_name}"}


def _search_knowledge_base(args: Dict) -> Dict:
    """执行知识库检索（含查询改写 + 扩大候选池 + Cross-Encoder 重排序）"""
    query = args.get("query", "")
    top_k = args.get("top_k", RETRIEVAL_TOP_K)

    if not query:
        logger.warning("检索查询为空")
        return {"success": False, "result": None, "error": "查询内容不能为空"}

    try:
        # L1: 查询改写 — 生成多个同义/多角度的查询变体
        from knowledge_base.query_rewriter import rewrite_query
        query_variants = rewrite_query(query)
        logger.debug("查询变体: %d 个", len(query_variants))

        # L2: 多查询检索 + Cross-Encoder 重排序
        retriever = get_retriever()
        results = retriever.multi_query_search(
            queries=query_variants,
            original_question=query,  # Cross-Encoder 用原问题打分
            final_top_k=top_k,
        )

        if not results:
            logger.info("多查询检索: 0 个结果")
            return {
                "success": True,
                "result": "知识库中暂无与查询相关的内容。建议先上传相关论文到知识库。",
                "chunks": [],
            }

        # 格式化检索结果
        formatted_chunks = []
        for i, r in enumerate(results):
            md = r.get("metadata", {})
            # 优先用 Cross-Encoder 分数，其次 RRF
            score = r.get("rerank_score", r.get("rrf_score", 0))
            formatted_chunks.append({
                "index": i + 1,
                "document_id": md.get("document_id", "未知"),
                "section": md.get("section_name", "未知章节"),
                "content": r.get("content", ""),
                "relevance_score": score,
            })

        logger.info("多查询检索完成: %d 个结果, top_score=%.4f",
                    len(formatted_chunks),
                    formatted_chunks[0]["relevance_score"] if formatted_chunks else 0)

        result_text = _format_search_results(formatted_chunks, query_variants)

        return {
            "success": True,
            "result": result_text,
            "chunks": formatted_chunks,
            "query_variants": query_variants,  # 供前端展示
        }
    except Exception as e:
        logger.error("知识库检索异常: %s", e, exc_info=True)
        return {"success": False, "result": None, "error": str(e)}


def _format_search_results(chunks: List[Dict], query_variants: List[str] = None) -> str:
    """将检索到的 chunks 格式化为 LLM 可读的文本"""
    lines = ["以下是从知识库中检索到的相关论文片段：\n"]

    if query_variants and len(query_variants) > 1:
        lines.append(f"（本次检索使用了 {len(query_variants)} 种不同的查询角度，包括：{' / '.join(query_variants[:5])}）\n")

    for c in chunks:
        lines.append(
            f"---\n"
            f"【片段 {c['index']}】\n"
            f"来源文档 ID: {c['document_id']}\n"
            f"章节: {c['section']}\n"
            f"相关度: {c['relevance_score']}\n"
            f"内容:\n{c['content']}\n"
        )
    return '\n'.join(lines)


def _search_papers_online(args: Dict) -> Dict:
    """执行联网论文检索（三源并行 + 年份过滤）"""
    query = args.get("query", "")
    limit = args.get("limit", 5)  # 默认5篇，与工具定义一致
    source = args.get("source", "all")
    since_year = args.get("since_year")

    if not query:
        return {"success": False, "result": None, "error": "查询内容不能为空"}

    try:
        from knowledge_base.paper_search import search_papers, translate_abstracts
        papers = search_papers(query=query, limit=limit, source=source, since_year=since_year)

        if not papers:
            return {"success": True, "result": f"未在学术数据库中找到与「{query}」相关的论文。", "papers": []}

        papers = translate_abstracts(papers)
        formatted = _format_paper_results(papers, query)

        return {
            "success": True,
            "result": formatted["text"],
            "papers": papers,
            "result_type": "paper_search",
        }
    except Exception as e:
        logger.error("联网论文检索异常: %s", e, exc_info=True)
        return {"success": False, "result": None, "error": str(e)}


def _format_paper_results(papers: List[Dict], query: str) -> Dict:
    """格式化论文结果：llm_text 给 LLM（简洁），papers 给前端卡片"""
    llm_lines = [f"检索到 {len(papers)} 篇与「{query}」相关的论文。请对以下论文进行简要总结分析："]
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
        llm_lines.append(
            f"{i + 1}. {p['title']}{year_str}{venue_str}{citations}\n"
            f"   作者: {authors_str}"
            + (f"\n   摘要首句: {first_sentence}." if first_sentence else "")
        )
    return {"text": "\n".join(llm_lines), "papers": papers}


# 论文结果嵌入标记（用于在消息中持久化存储论文元数据）
PAPERS_JSON_MARKER = "\n\n<!--PAPERS_JSON-->\n"


def encode_papers_in_message(content: str, papers: List[Dict]) -> str:
    """将论文元数据编码到消息内容末尾，实现持久化"""
    if not papers:
        return content
    import json as _json
    # 只保留前端渲染卡片需要的字段
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
    return content + PAPERS_JSON_MARKER + _json.dumps(compact, ensure_ascii=False)


def decode_papers_from_message(content: str) -> tuple:
    """从消息内容中提取论文元数据，返回 (clean_content, papers_list)"""
    if PAPERS_JSON_MARKER not in content:
        return content, []
    import json as _json
    clean, _, json_str = content.partition(PAPERS_JSON_MARKER)
    try:
        papers = _json.loads(json_str.strip())
        return clean, papers
    except _json.JSONDecodeError:
        return content, []


def get_tool_result_summary(tool_name: str, result: Dict) -> str:
    """生成工具执行结果的简要摘要"""
    if not result.get("success"):
        return f"执行失败: {result.get('error', '未知错误')}"

    if tool_name == "search_knowledge_base":
        chunks = result.get("chunks", [])
        if chunks:
            return f"检索到 {len(chunks)} 个相关片段, top_score={chunks[0].get('relevance_score', 'N/A')}"
        return "未检索到相关内容"

    if tool_name == "search_papers_online":
        papers = result.get("papers", [])
        if papers:
            return f"检索到 {len(papers)} 篇论文"
        return "未检索到相关论文"

    return "执行成功"
