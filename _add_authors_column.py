"""给 kg_papers 表添加 authors JSON 列"""
import sys; sys.path.insert(0, '.')
from database.connection import engine
from sqlalchemy import text

with engine.connect() as conn:
    result = conn.execute(text("""
        SELECT COLUMN_NAME FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'kg_papers'
        AND COLUMN_NAME = 'authors'
    """))
    if result.fetchone():
        print("[OK] kg_papers.authors column already exists")
    else:
        conn.execute(text("ALTER TABLE kg_papers ADD COLUMN authors JSON"))
        conn.commit()
        print("[OK] kg_papers.authors column added")

print("Migration done")
