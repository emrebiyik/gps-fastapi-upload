# database.py — SQLAlchemy session/engine bootstrap (Postgres or SQLite fallback)
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ------------------------------------------------------------------------------
# DATABASE_URL
#   Production (Postgres) example:
#     postgresql+psycopg2://USER:PASSWORD@HOST:5432/DBNAME
#   Local fallback (if not provided): SQLite file
# ------------------------------------------------------------------------------
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./app.db")

is_sqlite = DATABASE_URL.startswith("sqlite")

# For SQLite, need check_same_thread=False when used with FastAPI (multi-threaded)
connect_args = {"check_same_thread": False} if is_sqlite else {}

# Engine
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,                  # avoids stale connections
    connect_args=connect_args,           # only relevant for SQLite
    future=True,
    # NOTE: You may also pass pool_size/max_overflow for Postgres if desired, e.g.:
    # pool_size=5, max_overflow=10
)

# Session factory
SessionLocal = sessionmaker(
    bind=engine,
    autocommit=False,
    autoflush=False,
    future=True,
)

# Base class for ORM models
Base = declarative_base()

# FastAPI dependency for DB sessions
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()