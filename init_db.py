# init_db.py
from database import Base, engine
import models

def init():
    print("📦 Creating tables if not exist...")
    Base.metadata.create_all(bind=engine)
    print("✅ DB initialized at:", engine.url)

if __name__ == "__main__":
    init()