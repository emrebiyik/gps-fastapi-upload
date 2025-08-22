# init_db.py
from database import Base, engine
import models

def init():
    print("📌 Ensuring tables exist...")
    try:
        Base.metadata.create_all(bind=engine)
        print("✅ Tables checked/created.")
    except Exception as e:
        print("⚠️ Skipping create_all due to:", e)

if __name__ == "__main__":
    init()