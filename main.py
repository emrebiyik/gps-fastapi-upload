from __future__ import annotations
from typing import Dict, Any, List

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
    files: List[UploadFile] = File(..., description="One or more images"),
    db: Session = Depends(get_db),
    current_user: Dict[str, Any] = Depends(get_current_user),
):
    """
    MULTI-FILE VERSION
    - Accepts one or more images (multipart/form-data).
    - Extracts GPS (EXIF) for each image.
    - Uses the **first** image as the reference; distance for all other images
      is computed **relative to the first image**, not hop-to-hop.
    - If the **first** image has no GPS, raise 400.
    - Marks a record as 'abnormal' if distance_km > 1 km,
      otherwise 'normal' (if GPS present).
    - Persists all processed images to DB.

    Auth: Bearer token required. The token subject (sub) must match `user_id`.
    """
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this user")

    user = get_or_create_user(db, user_id)

    if not files or len(files) == 0:
        raise HTTPException(status_code=400, detail="No files were uploaded")

    first_file = files[0]
    first_coords = extract_gps_pillow(first_file)
    if not first_coords:
        raise HTTPException(status_code=400, detail="No GPS data in the FIRST image")

    ref_lat, ref_lon = first_coords
    results = []

    try:
        first_file.file.seek(0)
    except Exception:
        pass

    THRESHOLD_KM = 1.0

    for idx, up in enumerate(files):
        coords = extract_gps_pillow(up)
        if coords:
            lat, lon = coords
            if idx == 0:
                distance_km = 0.0
                flag = "normal"
            else:
                distance_km = haversine_distance(ref_lat, ref_lon, lat, lon)
                if distance_km is not None and distance_km > THRESHOLD_KM:
                    flag = "abnormal"
                else:
                    flag = "normal"
        else:
            lat = lon = None
            distance_km = None
            flag = "no_gps"

        row = GPSData(
            user_id=user.id,
            filename=up.filename,
            latitude=lat,
            longitude=lon,
            distance_km=distance_km,
            flag=flag,
        )
        db.add(row)
        db.flush()

        pretty_distance = round(distance_km, 2) if distance_km is not None else None

        results.append(
            {
                "index": idx,
                "gps_id": row.id,
                "filename": up.filename,
                "latitude": lat,
                "longitude": lon,
                "distance_km": pretty_distance,
                "flag": flag,
            }
        )

    db.commit()

    return {
        "status": "ok",
        "user_id": user_id,
        "reference": {
            "filename": files[0].filename,
            "latitude": ref_lat,
            "longitude": ref_lon,
        },
        "items": results,
    }


# -------------------------- Scoring helpers -------------------------
def score_gps(feat: Dict[str, Any] | None) -> int:
    if not feat:
        return 0
    return -5 if feat.get("flag") == "abnormal" else 10


def decision_from_score(total: int) -> str:
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
    if current_user["sub"] != payload.user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this user")

    user = db.query(User).filter(User.user_id == payload.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    latest_gps = (
        db.query(GPSData)
        .filter(GPSData.user_id == user.id)
        .order_by(GPSData.id.desc())
        .first()
    )

    gps_s = score_gps({"flag": latest_gps.flag if latest_gps else None})
    total = gps_s
    decision = decision_from_score(total)

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