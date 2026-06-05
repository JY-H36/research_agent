"""
Embedding 服务模块
使用 DashScope text-embedding-v4 进行文本向量化
"""
import time
import logging
from typing import List
import dashscope
from dashscope import TextEmbedding

from config import DASHSCOPE_API_KEY, EMBEDDING_MODEL
from utils.logger import get_logger

logger = get_logger(__name__)

dashscope.api_key = DASHSCOPE_API_KEY


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    批量文本向量化
    """
    if not texts:
        return []

    logger.debug("向量化开始: %d 个文本, model=%s", len(texts), EMBEDDING_MODEL)
    t_start = time.time()

    embeddings = []
    batch_size = 10  # text-embedding-v4 限制单批最多 10 条

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        try:
            resp = TextEmbedding.call(model=EMBEDDING_MODEL, input=batch)
            if resp.status_code == 200:
                batch_embeddings = [item['embedding'] for item in resp.output['embeddings']]
                embeddings.extend(batch_embeddings)
                logger.debug("向量化批次 %d/%d: %d 个, OK",
                           i // batch_size + 1,
                           (len(texts) + batch_size - 1) // batch_size,
                           len(batch))
            else:
                logger.error("Embedding API 调用失败: code=%s, msg=%s", resp.code, resp.message)
                for _ in batch:
                    embeddings.append([0.0] * 1024)
        except Exception as e:
            logger.error("Embedding API 异常 (批次 %d): %s", i // batch_size + 1, e, exc_info=True)
            for _ in batch:
                embeddings.append([0.0] * 1024)

    elapsed = time.time() - t_start
    logger.info("向量化完成: %d 个向量, 耗时 %.1fs", len(embeddings), elapsed)
    return embeddings


def embed_query(query: str) -> List[float]:
    """单条查询向量化"""
    try:
        resp = TextEmbedding.call(model=EMBEDDING_MODEL, input=query)
        if resp.status_code == 200:
            return resp.output['embeddings'][0]['embedding']
        else:
            logger.error("Embedding API 调用失败: code=%s, msg=%s", resp.code, resp.message)
            return [0.0] * 1024
    except Exception as e:
        logger.error("Embedding API 异常: %s", e, exc_info=True)
        return [0.0] * 1024
