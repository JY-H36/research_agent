"""给 kg_papers 表添加 method_name 和 method_summary 列"""
import sys; sys.path.insert(0, '.')
from database.connection import engine
from sqlalchemy import text

columns_to_add = [
    ("method_name", "VARCHAR(500) DEFAULT ''"),
    ("method_summary", "TEXT"),
]

with engine.connect() as conn:
    for col_name, col_type in columns_to_add:
        result = conn.execute(text("""
            SELECT COLUMN_NAME FROM information_schema.COLUMNS
            WHERE TABLE_SCHEMA = DATABASE()
            AND TABLE_NAME = 'kg_papers'
            AND COLUMN_NAME = :col
        """), {"col": col_name})
        if result.fetchone():
            print(f"[OK] kg_papers.{col_name} already exists")
        else:
            conn.execute(text(f"ALTER TABLE kg_papers ADD COLUMN `{col_name}` {col_type}"))
            conn.commit()
            print(f"[OK] kg_papers.{col_name} added")

print("Migration done")
