"""修复 kg_paper_uses_method 表的 role 列类型：VARCHAR → TEXT"""
import sys; sys.path.insert(0, '.')
from database.connection import engine
from sqlalchemy import text

with engine.connect() as conn:
    # 检查当前列类型
    result = conn.execute(text("""
        SELECT COLUMN_TYPE FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'kg_paper_uses_method'
        AND COLUMN_NAME = 'role'
    """))
    row = result.fetchone()
    if row:
        current_type = row[0]
        print(f"当前 role 列类型: {current_type}")
        if 'varchar' in current_type.lower():
            print("正在 ALTER ...")
            conn.execute(text("ALTER TABLE kg_paper_uses_method MODIFY COLUMN `role` TEXT"))
            conn.commit()
            print("✅ role 列已改为 TEXT")
        else:
            print("role 列已是 TEXT 类型，无需修改")
    else:
        print("表 kg_paper_uses_method 不存在")

print("迁移完成")
