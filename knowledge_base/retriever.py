"""
混合检索模块
结合 BM25（关键词检索）和语义检索，通过 RRF（Reciprocal Rank Fusion）融合
"""
import jieba
import logging
from typing import List, Dict
from rank_bm25 import BM25Okapi

from config import RETRIEVAL_TOP_K, RRF_K, CANDIDATE_POOL_SIZE
from knowledge_base.embedding_service import embed_query
from knowledge_base.vector_store import semantic_search as vs_semantic_search
from database.connection import SessionLocal
from database.models import Chunk
from utils.logger import get_logger

logger = get_logger(__name__)


class HybridRetriever:
    """混合检索器：BM25 + 语义检索 + RRF 融合"""

    def __init__(self):
        self._bm25_index: BM25Okapi = None
        self._chunk_texts: List[str] = []
        self._chunk_metadatas: List[Dict] = []

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        tokens = jieba.lcut(text)
        return [t.strip() for t in tokens if t.strip() and not all(c in '，。！？、；：""''（）…—·\t\n\r ' for c in t)]

    def build_bm25_index(self):
        db = SessionLocal()
        try:
            chunks = db.query(Chunk).order_by(Chunk.id).all()
            self._chunk_texts = []
            self._chunk_metadatas = []

            for chunk in chunks:
                self._chunk_texts.append(chunk.content)
                self._chunk_metadatas.append({
                    "chunk_id": f"doc{chunk.document_id}_chunk{chunk.chunk_index}",
                    "document_id": chunk.document_id,
                    "section_name": chunk.section_name,
                    "chunk_index": chunk.chunk_index,
                })

            if self._chunk_texts:
                tokenized = [self._tokenize(t) for t in self._chunk_texts]
                self._bm25_index = BM25Okapi(tokenized)
            else:
                self._bm25_index = None

            logger.info("BM25 索引构建完成: %d 个 chunk", len(self._chunk_texts))
        except Exception as e:
            logger.error("BM25 索引构建失败: %s", e, exc_info=True)
            self._bm25_index = None
        finally:
            db.close()

    def bm25_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        if self._bm25_index is None or not self._chunk_texts:
            logger.debug("BM25 检索: 索引为空")
            return []

        try:
            tokenized_query = self._tokenize(query)
            scores = self._bm25_index.get_scores(tokenized_query)

            indexed_scores = list(enumerate(scores))
            indexed_scores.sort(key=lambda x: x[1], reverse=True)
            top_indices = indexed_scores[:top_k]

            results = []
            for idx, score in top_indices:
                if score > 0:
                    results.append({
                        "chunk_id": self._chunk_metadatas[idx]["chunk_id"],
                        "content": self._chunk_texts[idx],
                        "metadata": self._chunk_metadatas[idx],
                        "score": round(float(score), 4),
                    })

            logger.debug("BM25 检索: query='%s', %d 结果, top_score=%.2f",
                        query[:60], len(results), results[0]['score'] if results else 0)
            return results
        except Exception as e:
            logger.error("BM25 检索失败: %s", e, exc_info=True)
            return []

    def semantic_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        try:
            query_emb = embed_query(query)
            return vs_semantic_search(query_emb, top_k)
        except Exception as e:
            logger.error("语义检索失败: %s", e, exc_info=True)
            return []

    def hybrid_search(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[Dict]:
        logger.debug("混合检索开始: query='%s', top_k=%d", query[:100], top_k)

        bm25_results = self.bm25_search(query, top_k=top_k * 2)
        semantic_results = self.semantic_search(query, top_k=top_k * 2)

        if not bm25_results and not semantic_results:
            logger.info("混合检索: 双路均为空")
            return []

        # RRF 融合
        chunk_scores = {}

        for rank, r in enumerate(bm25_results):
            cid = r["chunk_id"]
            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "bm25_rank": None, "semantic_rank": None,
                    "content": r["content"], "metadata": r["metadata"],
                    "bm25_score": r["score"], "semantic_score": None
                }
            chunk_scores[cid]["bm25_rank"] = rank + 1
            chunk_scores[cid]["bm25_score"] = r["score"]

        for rank, r in enumerate(semantic_results):
            cid = r["chunk_id"]
            if cid not in chunk_scores:
                chunk_scores[cid] = {
                    "bm25_rank": None, "semantic_rank": None,
                    "content": r["content"], "metadata": r["metadata"],
                    "bm25_score": None, "semantic_score": r["score"]
                }
            chunk_scores[cid]["semantic_rank"] = rank + 1
            chunk_scores[cid]["semantic_score"] = r["score"]

        for cid, info in chunk_scores.items():
            rrf = 0.0
            if info["bm25_rank"] is not None:
                rrf += 1.0 / (RRF_K + info["bm25_rank"])
            if info["semantic_rank"] is not None:
                rrf += 1.0 / (RRF_K + info["semantic_rank"])
            info["rrf_score"] = round(rrf, 6)

        sorted_results = sorted(chunk_scores.items(), key=lambda x: x[1]["rrf_score"], reverse=True)

        final_results = []
        for cid, info in sorted_results[:top_k]:
            final_results.append({
                "chunk_id": cid,
                "content": info["content"],
                "metadata": info["metadata"],
                "bm25_score": info["bm25_score"],
                "semantic_score": info["semantic_score"],
                "rrf_score": info["rrf_score"],
            })

        logger.info("混合检索完成: BM25=%d, Semantic=%d, 融合后=%d",
                   len(bm25_results), len(semantic_results), len(final_results))
        return final_results

    def multi_query_search(
        self,
        queries: List[str],
        original_question: str,
        top_k_per_query: int = CANDIDATE_POOL_SIZE,
        final_top_k: int = RETRIEVAL_TOP_K,
    ) -> List[Dict]:
        """
        多查询变体检索 + Cross-Encoder 重排序

        参数:
            queries: 查询变体列表
            original_question: 原始用户问题（用于 Cross-Encoder 打分）
            top_k_per_query: 每个变体检索的候选数（扩大候选池）
            final_top_k: 最终返回数

        返回: reranker 精选后的 top-k 结果
        """
        logger.info("多查询检索: %d 个变体, 每路 top-%d, 最终 top-%d",
                   len(queries), top_k_per_query, final_top_k)

        # Layer 2a: 每个变体独立检索，扩大候选池
        all_candidates = []
        seen_ids = set()

        for i, q in enumerate(queries):
            results = self.hybrid_search(q, top_k=top_k_per_query)
            logger.debug("  变体 %d/%d: '%s' → %d 结果",
                        i + 1, len(queries), q[:60], len(results))
            for r in results:
                cid = r.get("chunk_id", "")
                if cid and cid not in seen_ids:
                    seen_ids.add(cid)
                    all_candidates.append(r)

        logger.info("候选池: %d 个去重候选 (来自 %d 个变体)",
                   len(all_candidates), len(queries))

        # Layer 2b: Cross-Encoder 重排序
        if len(all_candidates) <= final_top_k:
            logger.debug("候选数 (%d) ≤ top_k (%d)，跳过重排序",
                        len(all_candidates), final_top_k)
            return all_candidates

        from knowledge_base.reranker import rerank
        return rerank(original_question, all_candidates, top_k=final_top_k)


_retriever_instance: HybridRetriever = None


def get_retriever() -> HybridRetriever:
    global _retriever_instance
    if _retriever_instance is None:
        _retriever_instance = HybridRetriever()
        _retriever_instance.build_bm25_index()
    return _retriever_instance


def rebuild_retriever():
    global _retriever_instance
    _retriever_instance = HybridRetriever()
    _retriever_instance.build_bm25_index()
    return _retriever_instance
