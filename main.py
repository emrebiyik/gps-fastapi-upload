# main.py
from __future__ import annotations

import os
from typing import Any, List, Optional, Dict, Tuple

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    status,
    Form,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- DB/Models (optional – keep if you already have them) ---
# from sqlalchemy.orm import Session
# from database import get_db
# from models import User, GPSData, CreditScore

# --- Auth ---
from auth import router as auth_router, get_current_user

# --- Utils you already have (names from your screenshots) ---
from utils import (
    extract_gps_pillow,     # async or sync – we handle both
    haversine_distance,
    score_calllogs_from_csv,
    # get_or_create_user,   # keep if you persist to DB
)

# =========================
# App & Middleware
# =========================
app = FastAPI(title="Credit Scoring API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include auth router (/auth/token)
app.include_router(auth_router)

# =========================
# Schemas (Pydantic)
# =========================
class GPSReference(BaseModel):
    filename: str
    latitude: float
    longitude: float

class GPSItem(BaseModel):
    index: int
    gps_id: Optional[int] = None
    filename: str
    latitude: float
    longitude: float
    distance_km: float
    flag: Optional[str] = None  # "normal" | "abnormal" | "no_gps"

class GPSIngestResult(BaseModel):
    status: str = "ok"
    user_id: str
    reference: GPSReference
    items: List[GPSItem]

class ScoreOut(BaseModel):
    loan_id: str
    score: float
    decision: str
    awarded: Dict[str, float] | Dict[str, Any]
    details: Dict[str, Any]

# =========================
# Settings / thresholds
# =========================
ABNORMAL_THRESHOLD_KM: float = float(os.getenv("GPS_ABNORMAL_KM", "1.0"))

# For scoring (defaults – override in env if you want)
APPROVE_MIN = float(os.getenv("APPROVE_MIN", "60"))
REVIEW_MIN = float(os.getenv("REVIEW_MIN", "40"))

# Example weights; keep in sync with your utils implementation
W_CALLS_PER_DAY = float(os.getenv("W_CALLS_PER_DAY", "0.25"))
W_AVG_DURATION  = float(os.getenv("W_AVG_DURATION",  "0.25"))
W_STABLE_RATIO  = float(os.getenv("W_STABLE_RATIO",  "0.50"))


# =========================
# Helpers
# =========================
def _caller_id(current_user: dict) -> str:
    """
    Support both {"sub": "..."} and {"username": "..."} shapes.
    """
    return current_user.get("sub") or current_user.get("username") or ""


async def _extract_first_gps(file: UploadFile) -> Tuple[float, float]:
    """
    Extract EXIF GPS from the first image. Raises HTTP 400 if not found.
    Works whether `extract_gps_pillow` is async or sync.
    """
    try:
        maybe_coro = extract_gps_pillow(file)
        if hasattr(maybe_coro, "__await__"):
            latlon = await maybe_coro  # async util
        else:
            latlon = maybe_coro       # sync util
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read EXIF GPS: {e}")

    if not latlon:
        raise HTTPException(status_code=400, detail="First image has no EXIF GPS")

    lat, lon = latlon
    return float(lat), float(lon)


async def _extract_gps_optional(file: UploadFile) -> Optional[Tuple[float, float]]:
    """
    Extract EXIF GPS; return None if missing. Never raises 500.
    """
    try:
        maybe_coro = extract_gps_pillow(file)
        if hasattr(maybe_coro, "__await__"):
            latlon = await maybe_coro
        else:
            latlon = maybe_coro
        if not latlon:
            return None
        lat, lon = latlon
        return float(lat), float(lon)
    except Exception:
        return None


# =========================
# Routes
# =========================
@app.get("/health", tags=["default"])
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.post(
    "/users/{user_id}/images",
    response_model=GPSIngestResult,
    tags=["gps"],
    summary="Upload Images For GPS",
)
async def upload_images_for_gps(
    user_id: str,
    files: List[UploadFile] = File(...),
    current_user: dict = Depends(get_current_user),
):
    """
    Ingest images, extract EXIF GPS, compute distance from FIRST image (km).
    Rules:
      - Distance for each subsequent image is computed relative to the FIRST image.
      - If the first has no GPS -> 400.
      - Mark as 'abnormal' if distance_km > ABNORMAL_THRESHOLD_KM, else 'normal'.
      - If an image has no GPS, set flag 'no_gps' and distance 0.
    Auth:
      - Bearer token required.
      - Token subject (sub/username) must match path param user_id.
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")

    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # FIRST image (must have GPS)
    first = files[0]
    ref_lat, ref_lon = await _extract_first_gps(first)
    reference = GPSReference(filename=first.filename, latitude=ref_lat, longitude=ref_lon)

    items: List[GPSItem] = []
    # include first image as baseline
    items.append(
        GPSItem(
            index=0,
            gps_id=None,  # replace with saved id if you persist
            filename=first.filename,
            latitude=ref_lat,
            longitude=ref_lon,
            distance_km=0.0,
            flag="normal",
        )
    )

    # subsequent images – distances relative to FIRST image
    for idx, up in enumerate(files[1:], start=1):
        latlon = await _extract_gps_optional(up)
        if not latlon:
            items.append(
                GPSItem(
                    index=idx,
                    gps_id=None,
                    filename=up.filename,
                    latitude=0.0,
                    longitude=0.0,
                    distance_km=0.0,
                    flag="no_gps",
                )
            )
            continue

        lat, lon = latlon
        try:
            dist_km = float(haversine_distance(ref_lat, ref_lon, lat, lon))
        except Exception as e:
            # Never leak 500 because of a math error
            raise HTTPException(status_code=400, detail=f"Failed to compute distance: {e}")

        flag = "abnormal" if dist_km > ABNORMAL_THRESHOLD_KM else "normal"

        items.append(
            GPSItem(
                index=idx,
                gps_id=None,  # replace with saved id if you persist
                filename=up.filename,
                latitude=lat,
                longitude=lon,
                distance_km=dist_km,
                flag=flag,
            )
        )

    return GPSIngestResult(user_id=user_id, reference=reference, items=items)


@app.post(
    "/users/{user_id}/score/calllogs",
    response_model=ScoreOut,
    tags=["scoring"],
    summary="Score From CallLogs CSV",
)
async def score_from_calllogs_csv(
    user_id: str,
    loan_id: str = Form(..., description="Loan/application id"),
    calllogs_csv: UploadFile = File(..., description="CallLogs CSV exported from phone"),
    current_user: dict = Depends(get_current_user),
):
    """
    Compute a rule-based credit score using the uploaded CallLogs CSV.

    - Weights are environment-tunable (W_*).
    - Decision thresholds are env-tunable: APPROVE_MIN (default 60) and REVIEW_MIN (default 40).
    - Returns a detailed payload with sub-scores and observed metrics.

    Auth:
      - Bearer token required.
      - Token subject (sub/username) must match path param user_id.
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")

    if calllogs_csv is None:
        raise HTTPException(status_code=400, detail="Missing calllogs_csv")

    # Read file-like object; make sure we never explode with 500 on bad input
    try:
        # Some utils accept UploadFile directly; others need .file
        # Try both patterns safely.
        try:
            result = score_calllogs_from_csv(calllogs_csv)  # type: ignore
        except TypeError:
            result = score_calllogs_from_csv(calllogs_csv.file)  # type: ignore
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # Normalize the result to our response model
    # Accept either a dict or a tuple (score, decision, awarded, details)
    if isinstance(result, dict):
        score = float(result.get("score"))
        decision = str(result.get("decision"))
        awarded = result.get("awarded", {})
        details = result.get("details", {})
    else:
        # assume tuple-like
        try:
            score, decision, awarded, details = result  # type: ignore
            score = float(score)
            decision = str(decision)
            if awarded is None:
                awarded = {}
            if details is None:
                details = {}
        except Exception:
            raise HTTPException(status_code=400, detail="Unexpected scoring output format")

    # Fallback decision logic if your util only returns the score
    if not decision:
        if score >= APPROVE_MIN:
            decision = "APPROVE"
        elif score >= REVIEW_MIN:
            decision = "REVIEW"
        else:
            decision = "REJECT"

    return ScoreOut(
        loan_id=loan_id,
        score=score,
        decision=decision,
        awarded=awarded,
        details=details,
    )