"""
论文联网检索模块（从 MCP Server JS 迁移）
- arXiv API         — 学术预印本
- Semantic Scholar  — 学术搜索引擎（引用量、DOI）
- OpenAlex          — 开放学术索引（无速率限制）
- 三源并行检索 + 年份过滤 + 去重排序 + LLM 摘要翻译
"""
import time
import re
import json
import logging
import requests
import feedparser
import concurrent.futures
from typing import List, Dict, Optional

from config import RETRIEVAL_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 配置
# ============================================================
ARXIV_API_URL = "http://export.arxiv.org/api/query"
S2_API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
OPENALEX_API_URL = "https://api.openalex.org/works"
REQUEST_TIMEOUT = 15
MAX_RETRIES = 2
RETRY_DELAY = 2
DEFAULT_LIMIT = 10
HEADERS = {"User-Agent": "ResearchAgent/1.0 (mailto:research@example.com)"}


def _request_with_retry(url: str, params: dict = None, headers: dict = None,
                        stream: bool = False, timeout: int = REQUEST_TIMEOUT) -> requests.Response:
    """带重试的 HTTP GET"""
    req_headers = {**HEADERS, **(headers or {})}
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=timeout,
                               stream=stream, headers=req_headers)
            if resp.status_code == 429:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                logger.debug("HTTP 429, 等待 %ds 后重试 (%d/%d)", wait, attempt, MAX_RETRIES)
                time.sleep(wait)
                continue
            resp.raise_for_status()
            return resp
        except requests.exceptions.HTTPError as e:
            if e.response is not None and e.response.status_code == 429:
                wait = RETRY_DELAY * (2 ** (attempt - 1))
                time.sleep(wait)
                continue
            last_error = e
            break
        except Exception as e:
            last_error = e
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    raise last_error or RuntimeError(f"Request failed after {MAX_RETRIES} retries")


# ============================================================
# arXiv 检索
# ============================================================
def search_arxiv(query: str, limit: int = DEFAULT_LIMIT) -> List[Dict]:
    """通过 arXiv API 检索论文"""
    params = {
        "search_query": f"all:{query}",
        "start": 0, "max_results": limit,
        "sortBy": "relevance", "sortOrder": "descending",
    }
    try:
        resp = _request_with_retry(ARXIV_API_URL, params=params)
        return _parse_arxiv_response(resp.text)
    except Exception as e:
        logger.warning("arXiv 检索失败: %s", e)
        return []


def _parse_arxiv_response(xml: str) -> List[Dict]:
    """解析 arXiv XML 响应"""
    feed = feedparser.parse(xml)
    papers = []
    for entry in feed.entries:
        try:
            arxiv_id = entry.get("id", "").split("/abs/")[-1].split("v")[0].strip()
            published = entry.get("published", "")
            papers.append({
                "source": "arxiv",
                "paper_id": arxiv_id,
                "title": entry.get("title", "").strip().replace("\n", " "),
                "abstract": entry.get("summary", "").strip().replace("\n", " "),
                "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "year": int(published[:4]) if published else None,
                "venue": "arXiv",
                "url": entry.get("id", ""),
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "citation_count": None,
            })
        except Exception as e:
            logger.debug("解析 arXiv 条目失败: %s", e)
    return papers


# ============================================================
# Semantic Scholar 检索
# ============================================================
def search_semantic_scholar(query: str, limit: int = DEFAULT_LIMIT) -> List[Dict]:
    """通过 Semantic Scholar API 检索论文"""
    params = {
        "query": query, "limit": limit,
        "fields": "paperId,title,abstract,year,venue,authors,url,citationCount,referenceCount,openAccessPdf",
    }
    try:
        resp = _request_with_retry(S2_API_URL, params=params)
        data = resp.json()
        papers = []
        for item in data.get("data", []):
            try:
                oa = item.get("openAccessPdf") or {}
                papers.append({
                    "source": "semantic_scholar",
                    "paper_id": item.get("paperId", ""),
                    "title": item.get("title", "").strip(),
                    "abstract": item.get("abstract", "") or "",
                    "authors": [a.get("name", "") for a in item.get("authors", [])],
                    "year": item.get("year"),
                    "venue": item.get("venue", "") or "",
                    "url": item.get("url", ""),
                    "pdf_url": oa.get("url", ""),
                    "citation_count": item.get("citationCount"),
                    "reference_count": item.get("referenceCount"),
                })
            except Exception as e:
                logger.debug("解析 S2 条目失败: %s", e)
        logger.debug("Semantic Scholar: '%s' → %d 篇", query[:80], len(papers))
        return papers
    except Exception as e:
        logger.warning("Semantic Scholar 检索失败: %s", e)
        return []


# ============================================================
# OpenAlex 检索（新增）
# ============================================================
def search_openalex(query: str, limit: int = DEFAULT_LIMIT) -> List[Dict]:
    """通过 OpenAlex API 检索论文（免费、无速率限制）"""
    params = {
        "search": query, "per-page": min(limit, 200),
        "select": "id,display_name,publication_year,abstract_inverted_index,authorships,primary_location,cited_by_count,open_access",
    }
    try:
        resp = _request_with_retry(OPENALEX_API_URL, params=params)
        data = resp.json()
        papers = []
        for item in data.get("results", []):
            try:
                paper_id = item.get("id", "").split("/")[-1] if item.get("id") else ""
                authors = [
                    a.get("author", {}).get("display_name", "")
                    for a in item.get("authorships", [])
                ]
                loc = item.get("primary_location") or {}
                oa = item.get("open_access") or {}
                papers.append({
                    "source": "openalex",
                    "paper_id": paper_id,
                    "title": item.get("display_name", "").strip(),
                    "abstract": _recover_openalex_abstract(item.get("abstract_inverted_index")),
                    "authors": authors,
                    "year": item.get("publication_year"),
                    "venue": loc.get("source", {}).get("display_name", "") if isinstance(loc.get("source"), dict) else "",
                    "url": loc.get("landing_page_url", "") or item.get("id", ""),
                    "pdf_url": oa.get("oa_url", ""),
                    "citation_count": item.get("cited_by_count"),
                })
            except Exception as e:
                logger.debug("解析 OpenAlex 条目失败: %s", e)
        logger.debug("OpenAlex: '%s' → %d 篇", query[:80], len(papers))
        return papers
    except Exception as e:
        logger.warning("OpenAlex 检索失败: %s", e)
        return []


def _recover_openalex_abstract(inverted_index) -> str:
    """从 OpenAlex 的倒排索引恢复摘要文本"""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pairs = []
    for word, positions in inverted_index.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(word for _, word in pairs)


# ============================================================
# 合并检索（并行 + 年份过滤 + 去重排序）
# ============================================================
def _extract_year_hints(query: str, current_year: int) -> tuple:
    """从查询文本中提取年份提示"""
    years = [int(m) for m in re.findall(r'\b(19|20)\d{2}\b', query)]
    min_year = current_year - 3  # 默认最近 3 年
    max_year = None
    if len(years) == 1:
        min_year = years[0]
        max_year = years[0]
    elif len(years) >= 2:
        min_year = min(years)
        max_year = max(years)
    return min_year, max_year


def search_papers(
    query: str,
    limit: int = DEFAULT_LIMIT,
    source: str = "all",
    since_year: int = None,
    until_year: int = None,
) -> List[Dict]:
    """
    并行检索 arXiv + Semantic Scholar + OpenAlex，合并结果

    参数:
        query: 搜索关键词
        limit: 返回论文数
        source: "all" | "arxiv" | "semantic_scholar" | "openalex"
        since_year: 起始年份
        until_year: 截止年份

    返回: 论文列表（按引用量降序）
    """
    logger.info("论文检索: query='%s', limit=%d, source=%s", query[:120], limit, source)

    # 扩大检索量（多捞再精选）
    expanded_limit = min(50, max(limit * 4, limit))

    # 确定年份范围
    from datetime import datetime
    current_year = datetime.now().year
    min_year, max_year = _extract_year_hints(query, current_year)
    if since_year is not None:
        min_year = since_year
    if until_year is not None:
        max_year = until_year
    logger.debug("年份过滤: %s ~ %s", min_year, max_year or "无上限")

    # 选择数据源
    sources_to_search = []
    if source == "all":
        sources_to_search = [
            ("openalex", lambda: search_openalex(query, expanded_limit)),
            ("arxiv", lambda: search_arxiv(query, expanded_limit)),
            ("semantic_scholar", lambda: search_semantic_scholar(query, expanded_limit)),
        ]
    else:
        source_map = {
            "arxiv": ("arxiv", lambda: search_arxiv(query, expanded_limit)),
            "semantic_scholar": ("semantic_scholar", lambda: search_semantic_scholar(query, expanded_limit)),
            "openalex": ("openalex", lambda: search_openalex(query, expanded_limit)),
        }
        if source in source_map:
            sources_to_search.append(source_map[source])

    # 并行检索
    warnings = []
    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {
            executor.submit(runner): name
            for name, runner in sources_to_search
        }
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                logger.debug("  %s: %d 篇", name, len(rows))
            except Exception as e:
                warnings.append({"source": name, "error": str(e)})
                logger.warning("  %s 失败: %s", name, e)

    # 年份过滤
    filtered = [
        p for p in all_rows
        if p.get("title") and (
            p.get("year") is None
            or (p["year"] >= min_year and (max_year is None or p["year"] <= max_year))
        )
    ]

    # 去重（按 title 前 100 字符）
    seen = {}
    for p in filtered:
        key = p.get("paper_id") or p.get("title", "")[:100].lower()
        if key and key not in seen:
            seen[key] = p
        elif key:
            # 合并信息（保留更丰富的摘要和引用数据）
            existing = seen[key]
            if not existing.get("abstract") and p.get("abstract"):
                existing["abstract"] = p["abstract"]
            if not existing.get("citation_count") and p.get("citation_count"):
                existing["citation_count"] = p["citation_count"]
            if not existing.get("pdf_url") and p.get("pdf_url"):
                existing["pdf_url"] = p.get("pdf_url")
            if not existing.get("venue") and p.get("venue"):
                existing["venue"] = p.get("venue")

    papers = list(seen.values())

    # 按引用量降序
    papers.sort(key=lambda x: x.get("citation_count") or 0, reverse=True)
    papers = papers[:limit]

    logger.info("论文检索完成: %d 篇 (总候选=%d, 去重后=%d, 数据源警告=%d)",
               len(papers), len(all_rows), len(seen), len(warnings))
    return papers


# ============================================================
# 摘要翻译（英文 → 中文）
# ============================================================
TRANSLATE_PROMPT = """Translate the following English academic paper abstracts into Chinese.
Keep technical terms in English (e.g., model names, metric names). Output ONLY the Chinese translation, one per line separated by "---", in the same order.

Abstracts:
{abstracts}

Translations (one per line, separated by ---):"""


def translate_abstracts(papers: List[Dict]) -> List[Dict]:
    """为论文列表批量翻译摘要"""
    need_translation = [
        (i, p) for i, p in enumerate(papers)
        if p.get("abstract") and len(p["abstract"]) > 20
    ]
    if not need_translation:
        return papers

    try:
        abstracts_text = "\n---\n".join([
            p["abstract"][:1500] for _, p in need_translation
        ])
        prompt = TRANSLATE_PROMPT.format(abstracts=abstracts_text)

        from agent.llm_service import chat_completion, extract_content
        logger.debug("批量翻译 %d 篇摘要", len(need_translation))
        response = chat_completion(
            messages=[{"role": "user", "content": prompt}],
            tools=None, temperature=0.1, max_tokens=2048,
        )
        translations = [
            t.strip().lstrip("0123456789.-) ").strip()
            for t in extract_content(response).split("---")
            if t.strip()
        ]

        for idx, (paper_idx, paper) in enumerate(need_translation):
            paper["abstract_cn"] = translations[idx] if idx < len(translations) else ""

        logger.info("摘要翻译: %d/%d 篇", min(len(translations), len(need_translation)), len(need_translation))
    except Exception as e:
        logger.warning("摘要翻译失败: %s", e)
        for _, paper in need_translation:
            paper["abstract_cn"] = ""

    return papers


# ============================================================
# PDF 下载
# ============================================================
def download_paper_pdf(pdf_url: str, save_path: str) -> bool:
    """下载论文 PDF"""
    if not pdf_url:
        return False
    try:
        resp = _request_with_retry(pdf_url, stream=True, timeout=30)
        with open(save_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)
        logger.info("PDF 下载成功: %s", save_path)
        return True
    except Exception as e:
        logger.error("PDF 下载失败: %s — %s", pdf_url[:80], e)
        return False
