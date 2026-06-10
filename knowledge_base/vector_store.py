"""
向量存储模块
使用 Chroma 进行向量持久化存储和语义检索
"""
import chromadb
from chromadb.config import Settings
from typing import List, Dict

from config import CHROMA_PERSIST_DIR, CHROMA_COLLECTION_NAME
from utils.logger import get_logger

logger = get_logger(__name__)

_client = chromadb.PersistentClient(
    path=CHROMA_PERSIST_DIR,
    settings=Settings(anonymized_telemetry=False),
)


def get_collection():
    return _client.get_or_create_collection(
        name=CHROMA_COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def add_chunks(
    chunk_ids: List[str],
    chunks_text: List[str],
    embeddings: List[List[float]],
    metadatas: List[Dict],
):
    if not chunk_ids:
        return

    logger.debug("Chroma 写入: %d 条记录", len(chunk_ids))
    collection = get_collection()
    try:
        collection.add(
            ids=chunk_ids,
            documents=chunks_text,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Chroma 写入完成: %d 条, collection 总数=%d", len(chunk_ids), collection.count())
    except Exception as e:
        logger.error("Chroma 添加 chunks 失败: %s", e, exc_info=True)


def semantic_search(query_embedding: List[float], top_k: int = 5) -> List[Dict]:
    collection = get_collection()

    if collection.count() == 0:
        logger.debug("Chroma 语义检索: collection 为空")
        return []

    try:
        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k, collection.count()),
            include=["documents", "metadatas", "distances"],
        )

        formatted = []
        if results and results['ids'] and results['ids'][0]:
            for i, chunk_id in enumerate(results['ids'][0]):
                distance = results['distances'][0][i] if results.get('distances') else 0
                similarity = 1.0 - distance
                formatted.append({
                    "chunk_id": chunk_id,
                    "content": results['documents'][0][i] if results.get('documents') else "",
                    "metadata": results['metadatas'][0][i] if results.get('metadatas') else {},
                    "score": round(similarity, 4),
                })

        logger.debug("Chroma 语义检索: %d 结果, top_score=%.4f",
                    len(formatted), formatted[0]['score'] if formatted else 0)
        return formatted
    except Exception as e:
        logger.error("Chroma 语义检索失败: %s", e, exc_info=True)
        return []


def delete_document_chunks(doc_id: int):
    logger.debug("Chroma 删除: doc_id=%d", doc_id)
    collection = get_collection()
    try:
        results = collection.get(where={"document_id": doc_id}, include=[])
        if results and results['ids']:
            collection.delete(ids=results['ids'])
            logger.info("Chroma 删除完成: doc_id=%d, 删除 %d 条", doc_id, len(results['ids']))
    except Exception as e:
        logger.error("Chroma 删除 chunks 失败: doc_id=%d, %s", doc_id, e, exc_info=True)


def get_chunk_count() -> int:
    try:
        collection = get_collection()
        return collection.count()
    except Exception:
        return 0


def clear_collection():
    """清空 collection 中的所有数据（保留 collection 本身）"""
    try:
        collection = get_collection()
        total = collection.count()
        if total > 0:
            # get() 默认只返回少量结果，必须指定 limit 才能拿到全部 ID
            all_ids = collection.get(limit=total, include=[])["ids"]
            if all_ids:
                collection.delete(ids=all_ids)
            logger.info("Chroma 已清空: 删除 %d 条记录", len(all_ids))
    except Exception as e:
        logger.warning("Chroma 清空失败: %s", e)
