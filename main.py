from __future__ import annotations
from typing import Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models import User, GPSData, CreditScore
from utils import extract_gps_pillow, haversine_distance, get_or_create_user
from auth import router as auth_router, get_current_user


app = FastAPI(title="Credit Scoring API", version="1.0.0")

# ----------------------------- CORS ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Expose /auth/token for OAuth2 password flow
app.include_router(auth_router)


# ---------------------- GPS Ingest (Protected) ----------------------
@app.post("/api/v1/gps/ingest")
async def ingest_gps(
    user_id: str,
    file: UploadFile = File(...),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Extract GPS from an uploaded image (EXIF), compute hop distance from last point,
    mark suspicious jumps, and persist to DB.

    Auth: Bearer token required. The token subject (sub) must match `user_id`.
    """
    # Enforce per-user access
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this user")

    # Ensure user exists (create if needed)
    user = get_or_create_user(db, user_id)

    # Extract coordinates
    coords = extract_gps_pillow(file)
    if not coords:
        raise HTTPException(status_code=400, detail="No GPS data found in image EXIF")

    latitude, longitude = coords
    filename = file.filename

    # Get last GPS point for distance calculation
    last_gps = (
        db.query(GPSData)
        .filter(GPSData.user_id == user.id)
        .order_by(GPSData.id.desc())
        .first()
    )

    distance_km = None
    flag = None
    if last_gps and last_gps.latitude is not None and last_gps.longitude is not None:
        distance_km = haversine_distance(
            last_gps.latitude, last_gps.longitude, latitude, longitude
        )
        # Simple anomaly rule — adjust threshold as needed
        if distance_km is not None and distance_km > 500:
            flag = "abnormal"

    # Persist
    gps_row = GPSData(
        user_id=user.id,
        filename=filename,
        latitude=latitude,
        longitude=longitude,
        distance_km=distance_km,
        flag=flag,
    )
    db.add(gps_row)
    db.commit()
    db.refresh(gps_row)

    return {"status": "ok", "gps_id": gps_row.id, "flag": flag, "distance_km": distance_km}


# -------------------------- Scoring helpers -------------------------
def score_gps(feat: Dict[str, Any] | None) -> int:
    """Very simple GPS rule: penalize abnormal jumps, reward otherwise."""
    if not feat:
        return 0
    return -5 if feat.get("flag") == "abnormal" else 10


def decision_from_score(total: int) -> str:
    """Map total score to a decision label."""
    if total >= 10:
        return "approve"
    if total >= 0:
        return "review"
    return "deny"


# -------------------------- Pydantic models -------------------------
class ScoreIn(BaseModel):
    user_id: str
    loan_id: str


class ScoreOut(BaseModel):
    user_id: str
    loan_id: str
    score: int
    decision: str


# ------------------- Compute Score (Protected) ----------------------
@app.post("/api/v1/score/compute", response_model=ScoreOut)
def compute_score(
    payload: ScoreIn,
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    Compute a credit score using GPS-derived features only (Phase 3 focus).

    Auth: Bearer token required. The token subject (sub) must match `payload.user_id`.
    """
    # Enforce per-user access
    if current_user["sub"] != payload.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this user")

    # Locate the user
    user = db.query(User).filter(User.user_id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Use the latest GPS record as our feature source
    latest_gps = (
        db.query(GPSData)
        .filter(GPSData.user_id == user.id)
        .order_by(GPSData.id.desc())
        .first()
    )

    gps_s = score_gps({"flag": latest_gps.flag if latest_gps else None})
    total = gps_s
    decision = decision_from_score(total)

    # Upsert by (user_id, loan_id): idempotent writes for the same application
    existing = (
        db.query(CreditScore)
        .filter(CreditScore.user_id == user.id, CreditScore.loan_id == payload.loan_id)
        .first()
    )

    if existing:
        existing.score = total
        existing.decision = decision
        existing.explanation_json = {"gps": gps_s}
    else:
        row = CreditScore(
            user_id=user.id,
            loan_id=payload.loan_id,
            score=total,
            decision=decision,
            explanation_json={"gps": gps_s},
        )
        db.add(row)

    db.commit()

    return ScoreOut(
        user_id=payload.user_id,
        loan_id=payload.loan_id,
        score=total,
        decision=decision,
    )