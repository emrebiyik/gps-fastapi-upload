from __future__ import annotations
from typing import Dict, Any, List, Optional
import io

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, status, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sqlalchemy.orm import Session

from database import get_db
from models import User, GPSData, CreditScore
from utils import (
    extract_gps_pillow,
    haversine_distance,
    get_or_create_user,
    score_calllogs_from_csv,
)
from auth import router as auth_router, get_current_user

app = FastAPI(title="Credit Scoring API", version="1.1.0")

# ----------------------------- CORS ---------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          # tighten in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------- Routers --------------------------------
app.include_router(auth_router)

# --------------------------- Schemas --------------------------------
class ScoreOut(BaseModel):
    user_id: str
    loan_id: str
    score: float
    decision: str
    details: Optional[Dict[str, Any]] = None

class GPSIngestResult(BaseModel):
    processed: int
    first_lat: float | None = None
    first_lon: float | None = None
    abnormalities: int

# --------------------------- Health ---------------------------------
@app.get("/health")
def health():
    return {"status": "ok"}

# ----------------------- GPS image upload ---------------------------
@app.post("/users/{user_id}/images", response_model=GPSIngestResult, tags=["gps"])
def upload_images_for_gps(
    user_id: str,
    files: List[UploadFile] = File(..., description="One or more images with EXIF GPS"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Ingest images, extract EXIF GPS, compute distance from FIRST image (km).

    Rules:
    - Distance for each subsequent image is computed **relative to the first image**, not hop-to-hop.
    - If the **first** image has no GPS, 400 is raised.
    - Marks a record as 'abnormal' if distance_km > 1 km; otherwise None (normal). If no GPS, flag='no_gps'.
    - Persists all processed images to DB.

    Auth: Bearer token required. The token subject (sub) must match `user_id`.
    """
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this user")

    user = get_or_create_user(db, user_id)

    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    # First image sets reference GPS
    ref_coords = None
    processed = 0
    abnormalities = 0

    for i, f in enumerate(files):
        coords = extract_gps_pillow(f)  # (lat, lon) or None
        if i == 0:
            if not coords:
                raise HTTPException(status_code=400, detail="First image has no GPS data")
            ref_coords = coords
            first_lat, first_lon = coords

        distance_km = None
        flag = None
        lat = lon = None
        if coords:
            lat, lon = coords
            distance_km = round(haversine_distance(ref_coords[0], ref_coords[1], lat, lon), 2)
            if distance_km > 1.0:
                flag = "abnormal"
                abnormalities += 1
        else:
            flag = "no_gps"

        row = GPSData(
            user_id=user.id,
            filename=getattr(f, "filename", None),
            latitude=lat,
            longitude=lon,
            distance_km=distance_km,
            flag=flag,
        )
        db.add(row)
        processed += 1

    db.commit()

    return GPSIngestResult(
        processed=processed,
        first_lat=ref_coords[0] if ref_coords else None,
        first_lon=ref_coords[1] if ref_coords else None,
        abnormalities=abnormalities,
    )

# ----------------------- Call-logs CSV scoring ----------------------
@app.post("/users/{user_id}/score/calllogs", response_model=ScoreOut, tags=["scoring"])
def score_from_calllogs_csv(
    user_id: str,
    loan_id: str = Form(..., description="Loan/application id"),
    calllogs_csv: UploadFile = File(..., description="CallLogs CSV exported from phone"),
    current_user: Dict[str, Any] = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    """Compute a **rule-based credit score** using the uploaded CallLogs CSV.

    - Uses environment-tunable weights (W_*) with sensible defaults.
    - Decision thresholds tunable via APPROVE_MIN (default: 60) and REVIEW_MIN (default: 40).
    - Stores/updates a CreditScore row for (user_id, loan_id) with an explanation JSON.

    This endpoint **does not change** the GPS ingestion behavior. Existing GPS routes remain intact.
    """
    if current_user["sub"] != user_id:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden for this user")

    user = get_or_create_user(db, user_id)

    # Read the file into memory (works fine for typical CSV sizes)
    calllogs_csv.file.seek(0)
    blob = calllogs_csv.file.read()
    if isinstance(blob, str):
        blob = blob.encode("utf-8")

    result = score_calllogs_from_csv(io.BytesIO(blob))

    # Upsert CreditScore for (user, loan_id)
    existing: CreditScore | None = (
        db.query(CreditScore)
        .filter(CreditScore.user_id == user.id, CreditScore.loan_id == loan_id)
        .first()
    )
    if existing:
        existing.score = int(round(result["score"]))
        existing.decision = result["decision"]
        existing.explanation_json = {"calllogs": {"metrics": result["metrics"], "awarded": result["awarded"]}}
    else:
        row = CreditScore(
            user_id=user.id,
            loan_id=loan_id,
            score=int(round(result["score"])),
            decision=result["decision"],
            explanation_json={"calllogs": {"metrics": result["metrics"], "awarded": result["awarded"]}},
        )
        db.add(row)
    db.commit()

    return ScoreOut(
        user_id=user_id,
        loan_id=loan_id,
        score=result["score"],
        decision=result["decision"],
        details={"calllogs": result},
    )