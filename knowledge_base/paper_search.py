"""
论文联网检索模块
- arXiv: 直连 + 显式代理双路径（主力源）
- OpenAlex: 显式代理（主力源，无速率限制）
- Semantic Scholar: 带冷却计时，避免连续 429
"""
import time
import re
import json
import requests
import urllib.request
import urllib.parse
import urllib.error
import feedparser
import concurrent.futures
from typing import List, Dict
from datetime import datetime

from config import RETRIEVAL_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)

# ============================================================
# 配置
# ============================================================
REQUEST_TIMEOUT = 10
DEFAULT_LIMIT = 10
HEADERS = {"User-Agent": "ResearchAgent/1.0 (mailto:research@example.com)"}

# Clash Verge 代理地址
PROXY_URL = "http://127.0.0.1:7897"
PROXIES = {"http": PROXY_URL, "https": PROXY_URL}

# Semantic Scholar 冷却时间（秒）
S2_COOLDOWN_SECONDS = 120
_s2_last_request_time: float = 0.0


def _s2_is_cooled_down() -> bool:
    """检查 S2 是否已冷却"""
    return (time.time() - _s2_last_request_time) >= S2_COOLDOWN_SECONDS


def _s2_mark_request():
    """记录 S2 请求时间"""
    global _s2_last_request_time
    _s2_last_request_time = time.time()


# ============================================================
# 双路径 HTTP 请求（直连 / 显式代理）
# ============================================================
def _try_fetch(url: str, accept: str = None, timeout: int = REQUEST_TIMEOUT) -> str:
    """
    Path A: urllib 直连（绕过代理）
    Path B: requests + 显式代理 http://127.0.0.1:7897
    两个路径并行，哪个先成功用哪个
    """
    req_headers = dict(HEADERS)
    if accept:
        req_headers["Accept"] = accept

    errors = {}

    def try_direct():
        try:
            r = urllib.request.Request(url, headers=req_headers)
            opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
            with opener.open(r, timeout=timeout) as resp:
                return resp.read().decode("utf-8")
        except Exception as e:
            errors["direct"] = str(e)[:120]
            return None

    def try_proxy():
        try:
            r = requests.get(url, headers=req_headers, timeout=timeout, proxies=PROXIES)
            r.raise_for_status()
            return r.text
        except Exception as e:
            errors["proxy"] = str(e)[:120]
            return None

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        f_direct = executor.submit(try_direct)
        f_proxy = executor.submit(try_proxy)

        # 等第一个成功的
        concurrent.futures.wait([f_direct, f_proxy],
                               return_when=concurrent.futures.FIRST_COMPLETED,
                               timeout=timeout + 5)

        for f in [f_direct, f_proxy]:
            if f.done():
                try:
                    result = f.result()
                    if result is not None:
                        return result
                except Exception:
                    pass

        for f in [f for f in [f_direct, f_proxy] if not f.done()]:
            try:
                result = f.result(timeout=5)
                if result is not None:
                    return result
            except Exception:
                pass

    raise RuntimeError(f"双路径均失败: {errors}")


# ============================================================
# arXiv
# ============================================================
def search_arxiv(query: str, limit: int = DEFAULT_LIMIT) -> List[Dict]:
    params = {
        "search_query": f"all:{query}", "start": 0,
        "max_results": str(limit), "sortBy": "relevance", "sortOrder": "descending",
    }
    url = f"https://export.arxiv.org/api/query?{urllib.parse.urlencode(params)}"
    try:
        xml = _try_fetch(url, accept="application/atom+xml")
        feed = feedparser.parse(xml)
        papers = []
        for entry in feed.entries:
            arxiv_id = entry.get("id", "").split("/abs/")[-1].split("v")[0].strip()
            published = entry.get("published", "")
            papers.append({
                "source": "arxiv", "paper_id": arxiv_id,
                "title": entry.get("title", "").strip().replace("\n", " "),
                "abstract": entry.get("summary", "").strip().replace("\n", " "),
                "authors": [a.get("name", "") for a in entry.get("authors", [])],
                "year": int(published[:4]) if published else None,
                "venue": "arXiv", "url": entry.get("id", ""),
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}.pdf",
                "citation_count": 0,
            })
        logger.debug("arXiv: %d 篇", len(papers))
        return papers
    except Exception as e:
        logger.warning("arXiv 检索失败: %s", e)
        return []


# ============================================================
# Semantic Scholar（带冷却控制）
# ============================================================
def search_semantic_scholar(query: str, limit: int = DEFAULT_LIMIT) -> List[Dict]:
    if not _s2_is_cooled_down():
        remaining = int(S2_COOLDOWN_SECONDS - (time.time() - _s2_last_request_time))
        logger.info("S2 冷却中 (还需 %ds)，跳过本次检索", remaining)
        return []

    _s2_mark_request()
    params = {
        "query": query, "limit": str(limit),
        "fields": "paperId,title,abstract,year,venue,authors,url,citationCount,referenceCount,openAccessPdf",
    }
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?{urllib.parse.urlencode(params)}"
    try:
        text = _try_fetch(url, accept="application/json")
        data = json.loads(text)
        papers = []
        for item in data.get("data", []):
            oa = item.get("openAccessPdf") or {}
            papers.append({
                "source": "semantic_scholar", "paper_id": item.get("paperId", ""),
                "title": (item.get("title") or "").strip(),
                "abstract": item.get("abstract") or "",
                "authors": [a.get("name", "") for a in item.get("authors", [])],
                "year": item.get("year"), "venue": item.get("venue") or "",
                "url": item.get("url", ""), "pdf_url": oa.get("url", ""),
                "citation_count": item.get("citationCount"),
            })
        logger.debug("Semantic Scholar: %d 篇", len(papers))
        return papers
    except Exception as e:
        # 被限速时重置冷却
        if "429" in str(e):
            _s2_mark_request()
            logger.warning("S2 429，冷却 %ds", S2_COOLDOWN_SECONDS)
        else:
            logger.warning("Semantic Scholar 检索失败: %s", e)
        return []


# ============================================================
# OpenAlex
# ============================================================
def search_openalex(query: str, limit: int = DEFAULT_LIMIT) -> List[Dict]:
    params = {
        "search": query, "per-page": str(min(limit, 200)),
        "select": "id,display_name,publication_year,abstract_inverted_index,authorships,primary_location,cited_by_count,open_access",
    }
    url = f"https://api.openalex.org/works?{urllib.parse.urlencode(params)}"
    try:
        text = _try_fetch(url, accept="application/json")
        data = json.loads(text)
        papers = []
        for item in data.get("results", []):
            title = (item.get("display_name") or "").strip()
            if not title:
                continue
            paper_id = item.get("id", "").split("/")[-1] if item.get("id") else ""
            authors = [a.get("author", {}).get("display_name", "") for a in item.get("authorships", [])]
            loc = item.get("primary_location") or {}
            oa = item.get("open_access") or {}
            papers.append({
                "source": "openalex", "paper_id": paper_id, "title": title,
                "abstract": _recover_oa_abstract(item.get("abstract_inverted_index")),
                "authors": authors, "year": item.get("publication_year"),
                "venue": (loc.get("source") or {}).get("display_name", "") if isinstance(loc.get("source"), dict) else "",
                "url": loc.get("landing_page_url", "") or item.get("id", ""),
                "pdf_url": oa.get("oa_url", ""),
                "citation_count": item.get("cited_by_count"),
            })
        logger.debug("OpenAlex: %d 篇", len(papers))
        return papers
    except Exception as e:
        logger.warning("OpenAlex 检索失败: %s", e)
        return []


def _recover_oa_abstract(inverted_index) -> str:
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    pairs = []
    for word, positions in inverted_index.items():
        if isinstance(positions, list):
            for pos in positions:
                if isinstance(pos, int):
                    pairs.append((pos, word))
    pairs.sort(key=lambda x: x[0])
    return " ".join(word for _, word in pairs)


# ============================================================
# 合并检索（arXiv + OpenAlex 主力，S2 冷却后补充）
# ============================================================
def search_papers(query: str, limit: int = DEFAULT_LIMIT, source: str = "all",
                  since_year: int = None) -> List[Dict]:
    logger.info("论文检索: '%s', limit=%d, source=%s", query[:120], limit, source)

    expanded_limit = min(50, max(limit * 4, limit))

    # 年份范围
    current_year = datetime.now().year
    years_in_query = [int(m) for m in re.findall(r'\b(19|20)\d{2}\b', query)]
    min_year = since_year if since_year else (years_in_query[0] if len(years_in_query) == 1 else current_year - 3)
    max_year = years_in_query[0] if len(years_in_query) == 1 else None
    if len(years_in_query) >= 2:
        min_year, max_year = min(years_in_query), max(years_in_query)

    # 数据源：arxiv + openalex 主力，S2 冷却后加入
    sources: List[tuple] = []
    if source == "all":
        sources = [("arxiv", lambda: search_arxiv(query, expanded_limit)),
                    ("openalex", lambda: search_openalex(query, expanded_limit))]
        if _s2_is_cooled_down():
            sources.append(("semantic_scholar", lambda: search_semantic_scholar(query, expanded_limit)))
        else:
            logger.info("S2 冷却中，跳过")
    else:
        m = {"arxiv": search_arxiv, "semantic_scholar": search_semantic_scholar, "openalex": search_openalex}
        if source in m:
            sources = [(source, lambda: m[source](query, expanded_limit))]

    # 并行检索
    all_rows = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = {executor.submit(runner): name for name, runner in sources}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                rows = future.result()
                all_rows.extend(rows)
                logger.debug("  %s: %d 篇", name, len(rows))
            except Exception as e:
                logger.warning("  %s 失败: %s", name, e)

    # 年份过滤
    filtered = [p for p in all_rows if p.get("title") and (
        p.get("year") is None or (p["year"] >= min_year and (max_year is None or p["year"] <= max_year)))]

    # 去重 + 信息合并
    seen = {}
    for p in filtered:
        key = p.get("paper_id") or p.get("title", "")[:100].lower()
        if key and key not in seen:
            seen[key] = p
        elif key:
            e = seen[key]
            for fld in ["abstract", "citation_count", "pdf_url", "venue"]:
                if not e.get(fld) and p.get(fld):
                    e[fld] = p[fld]

    papers = sorted(seen.values(), key=lambda x: x.get("citation_count") or 0, reverse=True)[:limit]
    logger.info("检索完成: %d 篇", len(papers))
    return papers


# ============================================================
# 摘要翻译
# ============================================================
TRANSLATE_PROMPT = """Translate the following English academic paper abstracts into Chinese.
Keep technical terms in English. Output translations separated by "---", same order.

Abstracts:
{abstracts}

Translations (separated by ---):"""


def translate_abstracts(papers: List[Dict]) -> List[Dict]:
    need = [(i, p) for i, p in enumerate(papers) if p.get("abstract") and len(p["abstract"]) > 20]
    if not need:
        return papers
    try:
        text = "\n---\n".join([p["abstract"][:1500] for _, p in need])
        from agent.llm_service import chat_completion, extract_content
        resp = chat_completion(messages=[{"role": "user", "content": TRANSLATE_PROMPT.format(abstracts=text)}],
                              tools=None, temperature=0.1, max_tokens=2048)
        translations = [t.strip().lstrip("0123456789.-) ") for t in extract_content(resp).split("---") if t.strip()]
        for idx, (pi, p) in enumerate(need):
            p["abstract_cn"] = translations[idx] if idx < len(translations) else ""
    except Exception as e:
        logger.warning("摘要翻译失败: %s", e)
    return papers


# ============================================================
# PDF 下载
# ============================================================
def download_paper_pdf(pdf_url: str, save_path: str) -> bool:
    if not pdf_url:
        return False
    try:
        r = requests.get(pdf_url, headers=HEADERS, timeout=30, stream=True, proxies=PROXIES)
        r.raise_for_status()
        with open(save_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)
        return True
    except Exception as e:
        logger.error("PDF 下载失败: %s", e)
        return False
