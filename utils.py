# utils.py (or inside main.py)
from sqlalchemy.orm import Session
from models import User

def get_or_create_user(db: Session, external_user_id: str) -> User:
    if not external_user_id:
        raise ValueError("external_user_id is required")

    user = db.query(User).filter(User.external_user_id == external_user_id).first()
    if user:
        return user

    user = User(external_user_id=external_user_id)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user