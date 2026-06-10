"""
实体提取模块
- Level 1: 正则快速提取 (title, authors, year, keywords, abstract, venue)
- Level 2: LLM Agent 深度提取 (完整 MD → 结构化信息)
- 完整提取 + 入图流水线
"""
import re
import json
from typing import Dict, List, Tuple

from database.connection import SessionLocal
from utils.logger import get_logger
from config import KG_ENABLE_LLM_EXTRACTION, KG_RESOLVER_USE_LLM

logger = get_logger(__name__)

# ============================================================
# Level 1: 正则快速提取（保留作为回退）
# ============================================================

# 常见英文人名模式: First Last, F. Last, First M. Last
_AUTHOR_PATTERN = re.compile(
    r'(?:^|\n)([A-Z][a-zà-ü]+(?:\s+[A-Z]\.)?(?:\s+[A-Z][a-zà-ü]+)+)(?:,|;|\n|$|\(|[0-9])',
    re.MULTILINE,
)

# Keywords 提取模式
_KEYWORDS_PATTERNS = [
    re.compile(r'\*\*Keywords\*\*[:\s]*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'Keywords[—–\-:]\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'Index Terms[—–\-:]\s*(.+?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'KEYWORDS[:\s]*(.+?)(?:\n|$)', re.IGNORECASE),
]

# 年份模式
_YEAR_PATTERN = re.compile(r'\b(19|20)\d{2}\b')

# Venue 版权信息模式
_VENUE_PATTERNS = [
    re.compile(r'(?:©|\(c\))\s*(?:20\d{2})\s*(.*?)(?:\n|$)', re.IGNORECASE),
    re.compile(r'(IEEE|ACM|ISCA|INTERSPEECH|ICASSP|AAAI|NeurIPS|ICML|CVPR|ACL|EMNLP|NAACL|COLING|ECCV|ICCV|ICLR)\s*(?:20\d{2})?', re.IGNORECASE),
]


def extract_metadata_from_md(md_text: str) -> Dict:
    """
    从 Docling 转换的 Markdown 中用正则提取基本元数据（回退方案）。
    返回: {
        "title": str, "authors": [str, ...], "year": int|None,
        "keywords": [str, ...], "abstract": str, "venue": str,
    }
    """
    result = {
        "title": "",
        "authors": [],
        "year": None,
        "keywords": [],
        "abstract": "",
        "venue": "",
    }

    if not md_text or not md_text.strip():
        return result

    lines = md_text.split('\n')

    # --- 提取 title ---
    title = ""
    for line in lines[:30]:
        stripped = line.strip()
        if stripped.startswith('# '):
            title = stripped.lstrip('#').strip()
            break

    if not title:
        candidates = [l.strip() for l in lines[:15] if l.strip() and not l.strip().startswith('#')]
        if candidates:
            def _looks_like_author_line(line: str) -> bool:
                if re.search(r'[∗†‡§¶‖]', line):
                    return True
                if line.count(',') >= 3 and len(line) < 200:
                    return True
                if re.match(r'^\d+\s+[A-Z]', line) and ',' in line:
                    return True
                return False

            for c in candidates:
                if len(c) > 20 and len(c) < 300 and not c.lower().startswith(('abstract', 'keywords')):
                    if _looks_like_author_line(c):
                        continue
                    title = c
                    break
            if not title and candidates:
                title = candidates[0][:300]

    result["title"] = title[:500]

    # --- 提取年份 ---
    for line in lines[:50]:
        years = _YEAR_PATTERN.findall(line)
        if years:
            for y in years:
                y_int = int(y)
                if 1990 <= y_int <= 2030:
                    result["year"] = y_int
                    break
        if result["year"]:
            break

    # --- 提取 abstract ---
    abstract_lines = []
    in_abstract = False
    abstract_start_markers = [
        'abstract', 'abstract—', 'abstract:', 'a b s t r a c t',
        '摘要', 'abstract ', 'abstract\n',
    ]

    for i, line in enumerate(lines):
        stripped = line.strip().lower()
        if not in_abstract:
            for marker in abstract_start_markers:
                if stripped.startswith(marker):
                    in_abstract = True
                    after_marker = line.strip()
                    for m in abstract_start_markers:
                        if after_marker.lower().startswith(m):
                            after_marker = after_marker[len(m):].strip('—–-: ')
                            break
                    if after_marker and len(after_marker) > 10:
                        abstract_lines.append(after_marker)
                    break
        else:
            if (stripped.startswith('#') or
                stripped.lower().startswith(('introduction', '1.', 'i.', 'related work',
                                              'keywords', 'index terms', '1 introduction'))):
                break
            if stripped:
                abstract_lines.append(line.strip())
        if i > 100:
            break

    result["abstract"] = ' '.join(abstract_lines)[:3000]

    # --- 提取 keywords ---
    for pattern in _KEYWORDS_PATTERNS:
        for line in lines[:80]:
            m = pattern.search(line)
            if m:
                kw_text = m.group(1).strip()
                kws = re.split(r'[,;，；]', kw_text)
                result["keywords"] = [k.strip().rstrip('.') for k in kws if k.strip() and len(k.strip()) > 1]
                break
        if result["keywords"]:
            break

    # --- 提取 authors ---
    authors = []
    author_section_found = False
    for i, line in enumerate(lines[:30]):
        stripped = line.strip()
        if stripped.startswith('#') or not stripped:
            continue
        names = _AUTHOR_PATTERN.findall(stripped)
        if names:
            authors.extend([n.strip() for n in names if len(n.strip()) > 3])
            author_section_found = True
        elif author_section_found:
            break

    seen = set()
    result["authors"] = [a for a in authors if not (a in seen or seen.add(a))][:30]

    # --- 提取 venue ---
    for line in lines[:100]:
        for pattern in _VENUE_PATTERNS:
            m = pattern.search(line)
            if m:
                v = m.group(1) if m.lastindex and m.lastindex >= 1 else m.group(0)
                v = v.strip()
                if v and len(v) < 200:
                    result["venue"] = v
                    break
        if result["venue"]:
            break

    if not result["venue"]:
        for line in lines[:80]:
            l = line.strip().lower()
            if l.startswith(('proceedings of', 'submitted to', 'published in',
                             'conference on', 'workshop on')):
                result["venue"] = line.strip()[:200]
                break

    logger.info("正则提取: title='%s', authors=%d, year=%s, keywords=%d, venue='%s'",
                result["title"][:60], len(result["authors"]),
                result["year"], len(result["keywords"]), result["venue"][:40])
    return result


# ============================================================
# 完整提取 + 入图（一键调用）
# ============================================================

def extract_and_ingest(md_text: str, document_id: int = None,
                       filename: str = "") -> Tuple[str, Dict]:
    """
    从论文 Markdown 提取所有实体并写入图谱（v3 精简版）。

    知识图谱实体: Paper, Method(feature_extractor/network_architecture/loss_function), Dataset
    不再创建 Author, Venue, Metric 实体（存为 Paper 的属性）

    返回:
        (paper_id, extraction_result)
    """
    from knowledge_graph.graph_store import (
        add_paper, add_method, add_dataset,
        link_paper_method, link_paper_dataset_eval, link_paper_dataset_train,
    )

    # ==== Step 1: 正则提取 abstract ====
    meta = extract_metadata_from_md(md_text)

    # ==== Step 2: LLM Agent v2 深度提取 ====
    agent_result = {"success": False, "data": None, "error": "LLM 提取未启用"}
    if KG_ENABLE_LLM_EXTRACTION:
        try:
            from knowledge_graph.paper_extraction_agent import extract_paper_info
            agent_result = extract_paper_info(md_text)
            logger.info("LLM Agent v2 提取: success=%s", agent_result.get("success"))
        except Exception as e:
            logger.error("LLM Agent 提取失败: %s", e, exc_info=True)
            agent_result = {"success": False, "data": None, "error": str(e)}

    agent_data = agent_result.get("data") or {}

    # ==== Step 3: 确定 Paper 属性（Agent 为主，正则为 fallback）====
    title = agent_data.get("title") or meta.get("title") or filename or "Untitled"
    year = agent_data.get("year") or meta.get("year")
    keywords = agent_data.get("keywords") or meta.get("keywords", [])
    authors = agent_data.get("authors") or meta.get("authors", [])
    venue_name = agent_data.get("venue") or meta.get("venue", "")
    abstract = meta.get("abstract", "")

    # ==== Step 4: 创建 Paper（proposed_method 信息直接存 Paper 属性）====
    pm = agent_data.get("proposed_method") or {}
    paper_id = add_paper({
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "author_names": authors,
        "method_name": pm.get("name", ""),
        "method_summary": pm.get("summary", ""),
        "year": year,
        "venue_name": venue_name,
        "source": "upload",
        "document_id": document_id,
        "docling_md": md_text[:20000],
    })

    # ==== Step 5: 入库实体 ====
    if agent_result.get("success") and agent_data:
        from knowledge_graph.entity_resolver import resolve_entity

        method_id_map = {}
        dataset_id_map = {}

        # --- 5a. 入库 feature_extractors ---
        for fe in agent_data.get("feature_extractors", []):
            try:
                fe_name = fe.get("name", "")
                if not fe_name:
                    continue
                fe_desc = fe.get("description", "")
                fe_variant = fe.get("variant", "")

                resolved_id, is_new = resolve_entity(
                    "method", fe_name, fe_desc,
                    {"category": "feature_extractor"},
                    use_llm=KG_RESOLVER_USE_LLM,
                )
                if is_new:
                    fe_id = add_method(
                        name=fe_name, category="feature_extractor",
                        description=fe_desc, aliases=[],
                    )
                else:
                    fe_id = resolved_id
                    if fe_desc:
                        _enrich_method_description(fe_id, fe_desc, "feature_extractor")

                method_id_map[fe_name] = fe_id
                link_paper_method(
                    paper_id, fe_id,
                    role="feature_extractor",
                    variant=fe_variant[:200],
                    context=fe_desc[:500],
                )
            except Exception as e:
                logger.warning("feature_extractor 入库失败 '%s': %s", fe.get("name", ""), e)

        # --- 5c. 入库 network_architectures ---
        for na in agent_data.get("network_architectures", []):
            try:
                na_name = na.get("name", "")
                if not na_name:
                    continue
                na_desc = na.get("description", "")

                resolved_id, is_new = resolve_entity(
                    "method", na_name, na_desc,
                    {"category": "network_architecture"},
                    use_llm=KG_RESOLVER_USE_LLM,
                )
                if is_new:
                    na_id = add_method(
                        name=na_name, category="network_architecture",
                        description=na_desc, aliases=[],
                    )
                else:
                    na_id = resolved_id
                    if na_desc:
                        _enrich_method_description(na_id, na_desc, "network_architecture")

                method_id_map[na_name] = na_id
                link_paper_method(
                    paper_id, na_id,
                    role="network_architecture",
                    variant="",
                    context=na_desc[:500],
                )
            except Exception as e:
                logger.warning("network_architecture 入库失败 '%s': %s", na.get("name", ""), e)

        # --- 5d. 入库 loss_functions ---
        for lf in agent_data.get("loss_functions", []):
            try:
                lf_name = lf.get("name", "")
                if not lf_name:
                    continue
                lf_desc = lf.get("description", "")

                resolved_id, is_new = resolve_entity(
                    "method", lf_name, lf_desc,
                    {"category": "loss_function"},
                    use_llm=KG_RESOLVER_USE_LLM,
                )
                if is_new:
                    lf_id = add_method(
                        name=lf_name, category="loss_function",
                        description=lf_desc, aliases=[],
                    )
                else:
                    lf_id = resolved_id
                    if lf_desc:
                        _enrich_method_description(lf_id, lf_desc, "loss_function")

                method_id_map[lf_name] = lf_id
                link_paper_method(
                    paper_id, lf_id,
                    role="loss_function",
                    variant="",
                    context=lf_desc[:500],
                )
            except Exception as e:
                logger.warning("loss_function 入库失败 '%s': %s", lf.get("name", ""), e)

        # --- 5e. 入库实验数据集 ---
        for ds in agent_data.get("experiment_datasets", []):
            try:
                ds_name = ds.get("name", "")
                if not ds_name:
                    continue
                ds_desc = ds.get("description", "")
                ds_task = ds.get("task", "")
                ds_role = ds.get("role", "eval")

                resolved_id, is_new = resolve_entity(
                    "dataset", ds_name, ds_desc,
                    use_llm=KG_RESOLVER_USE_LLM,
                )
                if is_new:
                    ds_id = add_dataset(
                        name=ds_name,
                        description=ds_desc,
                        task=ds_task,
                    )
                else:
                    ds_id = resolved_id
                    if ds_desc:
                        _enrich_dataset_description(ds_id, ds_desc)

                dataset_id_map[ds_name] = ds_id

                if ds_role in ("eval", "both"):
                    link_paper_dataset_eval(
                        paper_id, ds_id, task=ds_task, split="eval",
                    )
                if ds_role in ("train", "both"):
                    link_paper_dataset_train(
                        paper_id, ds_id, split="train+dev",
                    )
            except Exception as e:
                logger.warning("实验数据集入库失败 '%s': %s", ds.get("name", ""), e)

        # --- 5f. 入库实验结果（存储表数据到 JSON，不创建 Metric 实体） ---
        for er in agent_data.get("experiment_results", []):
            ds_name = er.get("dataset", "")
            condition = er.get("condition", "")
            results_list = er.get("results", [])

            if not results_list:
                continue

            # 找到 dataset_id
            dataset_id = dataset_id_map.get(ds_name)
            if not dataset_id:
                dataset_id = add_dataset(name=ds_name)
                dataset_id_map[ds_name] = dataset_id

            # 存储完整表数据到 KgPaperEvaluatesOn.metrics JSON 字段
            _store_experiment_table(paper_id, dataset_id, condition, results_list)

    # ==== Step 6: 自动归一化 ====
    from knowledge_graph.graph_store import _mark_graph_dirty
    _mark_graph_dirty()

    try:
        from knowledge_graph.entity_normalizer import normalize_all_entities
        norm_result = normalize_all_entities(dry_run=False)
        total_merged = (norm_result.get("methods", {}).get("merged", 0) +
                        norm_result.get("datasets", {}).get("merged", 0) +
                        norm_result.get("tasks", {}).get("merged", 0))
        if total_merged > 0:
            logger.info("自动归一化: 合并 %d 个重复实体", total_merged)
    except Exception as e:
        logger.warning("自动归一化失败: %s", e)

    return paper_id, agent_result


# ============================================================
# 辅助函数
# ============================================================

def _store_experiment_table(paper_id: str, dataset_id: str,
                            condition: str, results_list: List[Dict]):
    """将实验结果表存储到 KgPaperEvaluatesOn.metrics + 更新/创建关系"""
    from database.connection import SessionLocal as SL
    from knowledge_graph.models import KgPaperEvaluatesOn
    import json as _json

    db = SL()
    try:
        # 查找或创建 EVALUATES_ON 关系
        existing = db.query(KgPaperEvaluatesOn).filter(
            KgPaperEvaluatesOn.paper_id == paper_id,
            KgPaperEvaluatesOn.dataset_id == dataset_id,
        ).first()

        if existing:
            # 合并 metrics JSON
            cur = existing.metrics or {}
            if isinstance(cur, str):
                cur = _json.loads(cur)
            # 用 condition 做 key 存储
            cur[condition or "default"] = results_list
            existing.metrics = cur
        else:
            db.add(KgPaperEvaluatesOn(
                paper_id=paper_id, dataset_id=dataset_id,
                task="", split="eval",
                metrics={condition or "default": results_list},
                protocol="",
            ))

        db.commit()
        logger.info("实验表数据已存储: paper=%s, dataset=%s, condition='%s', rows=%d",
                    paper_id[:12], dataset_id[:12], condition, len(results_list))
    except Exception as e:
        db.rollback()
        logger.warning("实验表存储失败: %s", e)
    finally:
        db.close()


def _enrich_method_description(method_id: str, new_desc: str, category: str = ""):
    """补充已有 Method 的描述和类别"""
    from knowledge_graph.models import KgMethod
    db = SessionLocal()
    try:
        method = db.query(KgMethod).filter(KgMethod.method_id == method_id).first()
        if not method:
            return
        if new_desc and (not method.description or len(method.description) < len(new_desc)):
            method.description = new_desc
        if category and not method.category:
            method.category = category
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("补充 Method 描述失败: %s", e)
    finally:
        db.close()


def _enrich_dataset_description(dataset_id: str, new_desc: str):
    """补充已有 Dataset 的描述"""
    from knowledge_graph.models import KgDataset
    db = SessionLocal()
    try:
        dataset = db.query(KgDataset).filter(KgDataset.dataset_id == dataset_id).first()
        if not dataset:
            return
        if new_desc and (not dataset.description or len(dataset.description) < len(new_desc)):
            dataset.description = new_desc
            db.commit()
    except Exception as e:
        db.rollback()
        logger.warning("补充 Dataset 描述失败: %s", e)
    finally:
        db.close()
