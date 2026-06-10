"""
清空知识库脚本
删除所有 PDF/TXT 文件、MySQL 数据、Chroma 向量、知识图谱数据、BM25 索引
（保留表结构，只清空数据）
"""
import os
import sys
import shutil

# 确保能找到项目模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import database.models  # noqa: F401 — 注册所有表
from database.connection import init_db, SessionLocal
from database.models import Document, Chunk
from knowledge_base.vector_store import clear_collection
from knowledge_base.retriever import rebuild_retriever
from knowledge_graph.graph_store import clear_graph
from config import UPLOAD_DIR
from utils.logger import setup_logging, get_logger

setup_logging()
logger = get_logger("clear_kb")


def main():
    print("⚠️  即将清空全部知识库数据（表结构保留）")
    print(f"   - uploads/ 目录: {UPLOAD_DIR}")
    print(f"   - MySQL: documents + chunks 表")
    print(f"   - Chroma: 向量集合")
    print(f"   - 知识图谱: 全部 17 张 kg_* 表")
    print(f"   - BM25: 重建为空索引")
    print()
    confirm = input("确认清空？输入 'yes' 继续: ")
    if confirm.strip().lower() != "yes":
        print("已取消")
        return

    init_db()

    # 1. Chroma
    try:
        clear_collection()
        print("✅ Chroma 向量库已清空")
    except Exception as e:
        print(f"❌ Chroma 清空失败: {e}")

    # 2. MySQL documents + chunks
    db = SessionLocal()
    try:
        chunk_count = db.query(Chunk).count()
        doc_count = db.query(Document).count()
        db.query(Chunk).delete()
        db.query(Document).delete()
        db.commit()
        print(f"✅ MySQL 已清空: {doc_count} 个文档, {chunk_count} 个分块")
    except Exception as e:
        db.rollback()
        print(f"❌ MySQL 清空失败: {e}")
    finally:
        db.close()

    # 3. 知识图谱
    try:
        clear_graph()
        print("✅ 知识图谱已清空")
    except Exception as e:
        print(f"❌ 知识图谱清空失败: {e}")

    # 4. 上传文件
    if os.path.exists(UPLOAD_DIR):
        file_count = 0
        for f in os.listdir(UPLOAD_DIR):
            fpath = os.path.join(UPLOAD_DIR, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
                file_count += 1
        print(f"✅ uploads/ 已清空: {file_count} 个文件")
    else:
        print("✅ uploads/ 目录不存在，跳过")

    # 5. 重建空 BM25
    try:
        rebuild_retriever()
        print("✅ BM25 索引已重建（空）")
    except Exception as e:
        print(f"❌ BM25 重建失败: {e}")

    print()
    print("🎉 知识库已清空，可以重新上传论文了")


if __name__ == "__main__":
    main()
