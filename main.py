# main.py
from __future__ import annotations

import csv
import io
import os
from typing import Any, Dict, List, Optional, Tuple

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# --- Auth ---
from auth import router as auth_router, get_current_user

# --- Utils (keep these names consistent with your utils.py) ---
from utils import (
    extract_gps_pillow,     # returns (lat, lon) or None; may be sync/async
    haversine_distance,     # (lat1, lon1, lat2, lon2) -> km
    score_calllogs_from_csv # can accept UploadFile or file-like; we normalize below
)

# ============================================================
# App
# ============================================================
app = FastAPI(title="Credit Scoring API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)  # /auth/token

# ============================================================
# Models
# ============================================================
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
    awarded: Dict[str, Any]
    details: Dict[str, Any]

# ============================================================
# Settings
# ============================================================
ABNORMAL_THRESHOLD_KM = float(os.getenv("GPS_ABNORMAL_KM", "1.0"))
APPROVE_MIN = float(os.getenv("APPROVE_MIN", "60"))
REVIEW_MIN  = float(os.getenv("REVIEW_MIN",  "40"))

# Optional weights (if your scoring util uses them)
W_CALLS_PER_DAY = float(os.getenv("W_CALLS_PER_DAY", "0.25"))
W_AVG_DURATION  = float(os.getenv("W_AVG_DURATION",  "0.25"))
W_STABLE_RATIO  = float(os.getenv("W_STABLE_RATIO",  "0.50"))

# ============================================================
# Helpers
# ============================================================
def _caller_id(current_user: dict) -> str:
    """Support both {'sub': ...} and {'username': ...}."""
    return current_user.get("sub") or current_user.get("username") or ""

async def _extract_first_gps(file: UploadFile) -> Tuple[float, float]:
    """Read EXIF GPS from first image; raise 400 if missing or unreadable."""
    try:
        maybe = extract_gps_pillow(file)
        latlon = await maybe if hasattr(maybe, "__await__") else maybe
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read EXIF GPS: {e}")
    if not latlon:
        raise HTTPException(status_code=400, detail="First image has no EXIF GPS")
    lat, lon = latlon
    return float(lat), float(lon)

async def _extract_gps_optional(file: UploadFile) -> Optional[Tuple[float, float]]:
    """Read EXIF GPS; return None instead of raising on error/missing."""
    try:
        maybe = extract_gps_pillow(file)
        latlon = await maybe if hasattr(maybe, "__await__") else maybe
        if not latlon:
            return None
        lat, lon = latlon
        return float(lat), float(lon)
    except Exception:
        return None

# ============================================================
# Routes
# ============================================================
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
    - Distance is always relative to the first image.
    - If first image has no GPS → 400.
    - Flag 'abnormal' if distance_km > GPS_ABNORMAL_KM (default 1.0).
    - If an image has no GPS → flag 'no_gps', distance 0.
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Baseline
    first = files[0]
    ref_lat, ref_lon = await _extract_first_gps(first)
    reference = GPSReference(filename=first.filename, latitude=ref_lat, longitude=ref_lon)

    items: List[GPSItem] = [
        GPSItem(
            index=0,
            gps_id=None,
            filename=first.filename,
            latitude=ref_lat,
            longitude=ref_lon,
            distance_km=0.0,
            flag="normal",
        )
    ]

    # Others relative to the first
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
            raise HTTPException(status_code=400, detail=f"Failed to compute distance: {e}")

        flag = "abnormal" if dist_km > ABNORMAL_THRESHOLD_KM else "normal"
        items.append(
            GPSItem(
                index=idx,
                gps_id=None,
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
async def score_from_calllogs_csv_endpoint(
    user_id: str,
    loan_id: str = Form(..., description="Loan/application id"),
    calllogs_csv: UploadFile = File(..., description="CallLogs CSV exported from phone"),
    current_user: dict = Depends(get_current_user),
):
    """
    Compute a rule-based credit score using the uploaded CallLogs CSV.

    - Reads UploadFile asynchronously (await file.read()) and parses CSV safely.
    - Uses your `score_calllogs_from_csv` utility (dict or tuple outputs supported).
    - Decision thresholds env-tunable: APPROVE_MIN / REVIEW_MIN.
    - Returns detailed structure with 'awarded' and 'details'.
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")

    if calllogs_csv is None:
        raise HTTPException(status_code=400, detail="Missing calllogs_csv")

    # Read the uploaded file (async), normalize into a text stream for CSV parsers
    try:
        content: bytes = await calllogs_csv.read()
        if not content:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        text = content.decode("utf-8", errors="replace")
        text_stream = io.StringIO(text)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # Try your util with both UploadFile-like and file-like inputs
    try:
        try:
            result = score_calllogs_from_csv(calllogs_csv)         # some utils accept UploadFile
        except TypeError:
            result = score_calllogs_from_csv(text_stream)          # others accept io.TextIOBase
    except ValueError as e:
        raise HTTPException(status_code=400, detail=f"CSV parse error: {e}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # Normalize output
    if isinstance(result, dict):
        score = float(result.get("score"))
        decision = str(result.get("decision") or "")
        awarded = result.get("awarded", {}) or {}
        details = result.get("details", {}) or {}
    else:
        try:
            score, decision, awarded, details = result  # type: ignore
            score = float(score)
            decision = str(decision or "")
            awarded = awarded or {}
            details = details or {}
        except Exception:
            raise HTTPException(status_code=400, detail="Unexpected scoring output format")

    # Fallback decision if util returned only a score
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