# init_db.py — ensure tables exist (first-run helper)
"""
Run this once at startup to create ALL tables defined on SQLAlchemy Base metadata.

It imports `models` so SQLAlchemy discovers User, GPSData, CreditScore classes.
If you need migrations (adding columns/constraints to an existing table),
use Alembic in production. This script is intentionally conservative.
"""

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy import inspect

from database import engine, Base
import models  # noqa: F401  (ensure models are registered on Base)


def main():
    print("🔧 [init_db] Creating tables if not present...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ [init_db] create_all completed.")
    except SQLAlchemyError as e:
        print(f"❌ [init_db] create_all failed: {type(e).__name__}: {e}")
        raise

    try:
        insp = inspect(engine)
        tables = insp.get_table_names()
        print(f"📋 [init_db] Tables: {', '.join(tables) if tables else '(none)'}")
    except Exception as e:
        print(f"⚠️ [init_db] Could not list tables: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()