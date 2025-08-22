# database.py — PostgreSQL via SQLAlchemy
import os
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# Read connection string from environment
# Example value:
# postgresql://USER:PASSWORD@HOST:5432/DBNAME
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    raise RuntimeError(
        "DATABASE_URL is not set. Example: "
        "postgresql://USER:PASSWORD@HOST:5432/DBNAME"
    )

# Create engine (safe for Render; pre-ping avoids stale connections)
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,
    pool_size=5,
    max_overflow=10,
    future=True,
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