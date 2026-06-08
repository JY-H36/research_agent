"""
Cross-Encoder 重排序模块（可选增强）
用 BAAI/bge-reranker-large 对候选 chunk 精排。若模型不可用则自动回退到 RRF 排序。
"""
import os
import logging
from typing import List, Dict, Optional

from config import RERANKER_MODEL, RETRIEVAL_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局状态
_reranker = None           # CrossEncoder 实例 或 None
_reranker_failed = False   # True = 已尝试加载且失败，不再重试


def get_reranker() -> Optional[object]:
    """获取 Cross-Encoder。若加载失败返回 None"""
    global _reranker, _reranker_failed

    if _reranker is not None:
        return _reranker
    if _reranker_failed:
        return None

    try:
        # 确保 HF 镜像在加载前已设置
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

        logger.info("加载 Cross-Encoder 模型: %s", RERANKER_MODEL)
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL, max_length=512)
        logger.info("Cross-Encoder 模型加载完成 ✅")

    except Exception as e:
        _reranker_failed = True
        _reranker = None
        logger.warning(
            "Cross-Encoder 模型加载失败 (%s)。网络不可达或模型未缓存。"
            "此后将使用 RRF 排序，不影响检索功能。"
            "如需启用重排序，请在有网络时手动运行: "
            "python -c \"from sentence_transformers import CrossEncoder; "
            "CrossEncoder('%s', max_length=512)\"",
            e, RERANKER_MODEL
        )

    return _reranker


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = RETRIEVAL_TOP_K,
) -> List[Dict]:
    """
    用 Cross-Encoder 重排序候选 chunk。模型不可用时自动回退 RRF。

    返回: 重排后的 top_k chunks
    """
    if not candidates:
        return []

    # 去重
    seen = set()
    unique_candidates = []
    for c in candidates:
        cid = c.get("chunk_id", "")
        if cid and cid not in seen:
            seen.add(cid)
            unique_candidates.append(c)
        elif not cid:
            unique_candidates.append(c)

    if not unique_candidates:
        return []

    logger.debug("重排序: query='%s', 候选数=%d", query[:100], len(unique_candidates))

    # 尝试 Cross-Encoder 重排
    model = get_reranker()
    if model is not None:
        try:
            pairs = [(query, c.get("content", "")) for c in unique_candidates]
            scores = model.predict(pairs, show_progress_bar=False)
            for i, c in enumerate(unique_candidates):
                c["rerank_score"] = round(float(scores[i]), 4)
            unique_candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)
            top = unique_candidates[:top_k]
            if top:
                logger.info("重排序完成 (Cross-Encoder): top_score=%.4f", top[0].get("rerank_score", 0))
            return top
        except Exception as e:
            logger.error("Cross-Encoder 打分失败: %s，回退 RRF", e)

    # 回退: RRF 排序
    unique_candidates.sort(
        key=lambda x: x.get("rrf_score", 0), reverse=True
    )
    top = unique_candidates[:top_k]
    if top:
        logger.info("重排序完成 (RRF 回退): top_score=%.4f", top[0].get("rrf_score", 0))
    return top
