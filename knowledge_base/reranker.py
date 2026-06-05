"""
Cross-Encoder 重排序模块
用 BAAI/bge-reranker-large 对候选 chunk 精排，解决概念匹配问题
"""
import logging
from typing import List, Dict, Optional
from sentence_transformers import CrossEncoder

from config import RERANKER_MODEL, RETRIEVAL_TOP_K
from utils.logger import get_logger

logger = get_logger(__name__)

# 全局单例（延迟加载，首次使用时下载模型）
_reranker: Optional[CrossEncoder] = None


def get_reranker() -> CrossEncoder:
    """获取 Cross-Encoder 单例"""
    global _reranker
    if _reranker is None:
        logger.info("加载 Cross-Encoder 模型: %s", RERANKER_MODEL)
        _reranker = CrossEncoder(
            RERANKER_MODEL,
            max_length=512,  # query+chunk 最大 token 数
        )
        logger.info("Cross-Encoder 模型加载完成")
    return _reranker


def rerank(
    query: str,
    candidates: List[Dict],
    top_k: int = RETRIEVAL_TOP_K,
) -> List[Dict]:
    """
    用 Cross-Encoder 对候选 chunk 重排序

    参数:
        query: 原始用户问题（非变体，用原问题打分最准确）
        candidates: 候选列表，每项需含 "content" 字段
        top_k: 返回数量

    返回:
        重排后的 top_k chunks，每项额外包含 "rerank_score"
    """
    if not candidates:
        logger.debug("重排序: 候选为空")
        return []

    # 去重（按 chunk_id）
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

    logger.debug("重排序: query='%s', 候选数=%d (去重后)", query[:100], len(unique_candidates))

    try:
        model = get_reranker()

        # 构建 (query, document) 对
        pairs = [(query, c.get("content", "")) for c in unique_candidates]

        # Cross-Encoder 打分（一批完成）
        scores = model.predict(pairs, show_progress_bar=False)

        # 绑定分数
        for i, c in enumerate(unique_candidates):
            c["rerank_score"] = round(float(scores[i]), 4)

        # 按分数降序排序
        unique_candidates.sort(key=lambda x: x.get("rerank_score", 0), reverse=True)

        top = unique_candidates[:top_k]

        logger.info("重排序完成: top_score=%.4f, bottom_score=%.4f",
                   top[0].get("rerank_score", 0) if top else 0,
                   top[-1].get("rerank_score", 0) if top else 0)

        return top

    except Exception as e:
        logger.error("重排序失败: %s，回退到 RRF 排序", e, exc_info=True)
        # 失败时回退到 RRF 分数排序
        unique_candidates.sort(
            key=lambda x: x.get("rrf_score", x.get("rerank_score", 0)),
            reverse=True,
        )
        return unique_candidates[:top_k]
