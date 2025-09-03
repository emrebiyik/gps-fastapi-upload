# auth.py
from datetime import datetime, timedelta
import os
from typing import Dict, Any, Optional

from fastapi import APIRouter, Depends, HTTPException, status, Form
from fastapi.security import APIKeyHeader
from jose import JWTError, jwt

# ---------------- Config ----------------
SECRET_KEY = os.getenv("SECRET_KEY", "CHANGE_ME_SUPER_SECRET")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "120"))

# Demo users (example)
FAKE_USERS_DB = {
    "USER123": "secret123",
    "emre": "secret",
}

router = APIRouter(prefix="/auth", tags=["auth"])

# -------------- Token helpers --------------
def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# -------------- Login endpoint (issues token) --------------
@router.post("/token", summary="Login For Access Token")
def login_for_access_token(
    username: str = Form(...),
    password: str = Form(...),
    grant_type: Optional[str] = Form(None),
    scope: Optional[str] = Form(None),
    client_id: Optional[str] = Form(None),
    client_secret: Optional[str] = Form(None),
):
    real_pwd = FAKE_USERS_DB.get(username)
    if real_pwd is None or real_pwd != password:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Incorrect username or password")
    access_token = create_access_token(data={"sub": username})
    return {"access_token": access_token, "token_type": "bearer"}

# -------------- Swagger API Key (Authorization header) --------------
api_key_header = APIKeyHeader(name="Authorization", auto_error=False)

def _extract_bearer_token(auth_header: Optional[str]) -> str:
    if not auth_header:
        raise HTTPException(
            status_code=401,
            detail="Missing Authorization header",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=401,
            detail='Authorization must start with "Bearer "',
            headers={"WWW-Authenticate": "Bearer"},
        )
    return auth_header.split(" ", 1)[1].strip()

def _decode_token(token: str) -> Dict[str, Any]:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(
            status_code=401,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

def get_current_user(auth_header: Optional[str] = Depends(api_key_header)) -> Dict[str, Any]:
    token = _extract_bearer_token(auth_header)
    payload = _decode_token(token)
    sub = payload.get("sub")
    if not sub:
        raise HTTPException(status_code=401, detail="Token missing 'sub'")
    if sub not in FAKE_USERS_DB:
        raise HTTPException(status_code=401, detail="User not found")
    return {"username": sub}