"""迁移 kg_methods.category 从 ENUM 到 VARCHAR + 迁移 FRAMEWORK 数据到 Paper"""
import sys; sys.path.insert(0, '.')
from database.connection import engine
from sqlalchemy import text

with engine.connect() as conn:
    # 1. 将 category 列从 ENUM 改为 VARCHAR(50)
    result = conn.execute(text("""
        SELECT COLUMN_TYPE FROM information_schema.COLUMNS
        WHERE TABLE_SCHEMA = DATABASE()
        AND TABLE_NAME = 'kg_methods'
        AND COLUMN_NAME = 'category'
    """))
    row = result.fetchone()
    if row and 'enum' in row[0].lower():
        print(f"Current type: {row[0]}")
        conn.execute(text(
            "ALTER TABLE kg_methods MODIFY COLUMN category VARCHAR(50) DEFAULT NULL"
        ))
        conn.commit()
        print("[OK] category column changed to VARCHAR(50)")
    else:
        print(f"[OK] category already VARCHAR: {row[0] if row else 'N/A'}")

    # 2. 将 FRAMEWORK 类型的方法数据迁移到对应 Paper
    result = conn.execute(text("""
        SELECT m.method_id, m.name, m.description, m.paper_id
        FROM kg_methods m
        WHERE m.category = 'FRAMEWORK'
    """))
    framework_methods = result.fetchall()
    print(f"\nFound {len(framework_methods)} FRAMEWORK methods to migrate")

    for mid, name, desc, paper_id in framework_methods:
        if paper_id:
            # 更新对应 Paper 的 method_name 和 method_summary
            conn.execute(text("""
                UPDATE kg_papers
                SET method_name = :name, method_summary = :desc
                WHERE paper_id = :paper_id
            """), {"name": name, "desc": desc or "", "paper_id": paper_id})
            print(f"  Paper {paper_id[:12]}: method_name='{name[:40]}'")

    # 3. 删除不再需要的 FRAMEWORK 方法关系+实体
    for mid, name, desc, paper_id in framework_methods:
        # 删除 USES_METHOD 关系
        conn.execute(text("DELETE FROM kg_paper_uses_method WHERE method_id = :mid"), {"mid": mid})
        # 删除 IMPROVES_UPON 引用
        conn.execute(text("DELETE FROM kg_method_improves_method WHERE method_a_id = :mid OR method_b_id = :mid"), {"mid": mid})
        # 删除方法实体
        conn.execute(text("DELETE FROM kg_methods WHERE method_id = :mid"), {"mid": mid})
    if framework_methods:
        print(f"\n[OK] Deleted {len(framework_methods)} FRAMEWORK methods")

    # 4. 将 CLASSIFIER/其他旧类型改为 feature_extractor（如果存在）
    conn.execute(text("""
        UPDATE kg_methods SET category = 'feature_extractor'
        WHERE category NOT IN ('feature_extractor', 'network_architecture', 'loss_function')
    """))

    conn.commit()
    print("\n[DONE] Migration complete")
