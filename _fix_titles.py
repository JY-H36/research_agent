"""修复已入库论文的标题：从 docling_md 重新提取正确的标题"""
import sys; sys.path.insert(0, r"f:\科研助手agent")
import database.models
from database.connection import SessionLocal
from knowledge_graph.models import KgPaper
from knowledge_graph.entity_extractor import extract_metadata_from_md

db = SessionLocal()
papers = db.query(KgPaper).all()
fixed = 0
for p in papers:
    old_title = p.title
    if p.docling_md:
        meta = extract_metadata_from_md(p.docling_md)
        new_title = meta.get("title", "") or old_title
        # 只更新确实不同的
        if new_title and new_title != old_title and len(new_title) > 10:
            # 判断 old_title 是否像作者行
            import re
            is_author_line = bool(re.search(r'[,\d][,\d]', old_title) and len(old_title) < 200)
            if is_author_line or len(new_title) > len(old_title):
                p.title = new_title
                fixed += 1
                print(f"FIXED: [{len(old_title)}->{len(new_title)} chars]")
db.commit()
print(f"\nTotal fixed: {fixed}/{len(papers)}")
db.close()
