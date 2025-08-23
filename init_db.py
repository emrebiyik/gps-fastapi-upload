# init_db.py
"""
Run this once at startup to create ALL tables.
Works with the engine & Base defined in database.py.
"""

from sqlalchemy import text
from sqlalchemy.exc import SQLAlchemyError

# IMPORTANT: these imports must succeed so SQLAlchemy "sees" all models
from database import engine, Base
import models  # noqa: F401  # just to register model classes with Base

def main():
    print("🔧 [init_db] Connecting to DB…")
    try:
        with engine.connect() as conn:
            db_name = conn.exec_driver_sql("SELECT current_database()").scalar() \
                if engine.dialect.name == "postgresql" else "sqlite/memory/other"
            print(f"🔌 [init_db] Dialect: {engine.dialect.name}, DB: {db_name}")

            # (Optional) Enable FK constraints on SQLite
            if engine.dialect.name == "sqlite":
                conn.execute(text("PRAGMA foreign_keys=ON;"))
                print("✅ [init_db] SQLite foreign_keys pragma enabled")

    except SQLAlchemyError as e:
        print(f"❌ [init_db] Connection failed: {type(e).__name__}: {e}")
        raise

    # Create all tables
    try:
        print("🧱 [init_db] Creating tables if not exist…")
        Base.metadata.create_all(bind=engine)
        print("✅ [init_db] All tables are up to date.")
    except SQLAlchemyError as e:
        print(f"❌ [init_db] create_all failed: {type(e).__name__}: {e}")
        raise

    # Quick sanity check: list created tables (optional)
    try:
        from sqlalchemy import inspect
        insp = inspect(engine)
        tables = insp.get_table_names()
        print(f"📋 [init_db] Tables: {', '.join(tables) if tables else '(none)'}")
    except Exception as e:
        print(f"⚠️ [init_db] Could not list tables: {type(e).__name__}: {e}")

if __name__ == "__main__":
    main()