# main.py
from __future__ import annotations

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

# --- Auth (matches your auth.py) ---
from auth import router as auth_router, get_current_user

# --- Utils (keep names in sync with your utils.py) ---
from utils import (
    extract_gps_pillow,     # UploadFile -> (lat, lon) | None  (may be sync/async)
    haversine_distance,     # (lat1, lon1, lat2, lon2) -> float (km)
    score_calllogs_from_csv # (io.TextIOBase) -> dict or tuple  (may be sync/async)
)

# ============================================================
# App & CORS
# ============================================================
app = FastAPI(title="Credit Scoring API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=os.getenv("CORS_ALLOW_ORIGINS", "*").split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# /auth/token
app.include_router(auth_router)

# ============================================================
# Schemas
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

# ============================================================
# Helpers
# ============================================================
def _caller_id(current_user: dict) -> str:
    """Support both {'sub': ...} and {'username': ...} from auth."""
    return current_user.get("sub") or current_user.get("username") or ""

async def maybe_await(value):
    """Await a coroutine if needed; otherwise return the value."""
    return await value if hasattr(value, "__await__") else value

async def _extract_first_gps(file: UploadFile) -> Tuple[float, float]:
    """Read EXIF GPS from first image; raise 400 if missing or unreadable."""
    try:
        latlon = await maybe_await(extract_gps_pillow(file))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read EXIF GPS: {e}")
    if not latlon:
        raise HTTPException(status_code=400, detail="First image has no EXIF GPS")
    lat, lon = latlon
    return float(lat), float(lon)

async def _extract_gps_optional(file: UploadFile) -> Optional[Tuple[float, float]]:
    """Read EXIF GPS; return None instead of raising."""
    try:
        latlon = await maybe_await(extract_gps_pillow(file))
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
    Ingest images, extract EXIF GPS, compute distance from the FIRST image.
    - Distances are relative to the first image (not hop-to-hop).
    - If first image has no GPS -> 400.
    - Flag 'abnormal' if distance_km > GPS_ABNORMAL_KM (default 1 km).
    - If any image has no GPS -> flag 'no_gps', distance 0.
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

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

        items.append(
            GPSItem(
                index=idx,
                gps_id=None,
                filename=up.filename,
                latitude=lat,
                longitude=lon,
                distance_km=dist_km,
                flag="abnormal" if dist_km > ABNORMAL_THRESHOLD_KM else "normal",
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

    IMPORTANT:
    - We ALWAYS read the UploadFile with `await` and pass a **StringIO text stream**
      into your scoring util to avoid 'coroutine was never awaited' issues.
    - The util may be sync or async; both are handled.
    - User mistakes return 400 (not 500).
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")

    if calllogs_csv is None:
        raise HTTPException(status_code=400, detail="Missing calllogs_csv")

    # 1) Read upload ASYNC, decode, wrap in StringIO
    try:
        raw: bytes = await calllogs_csv.read()  # <-- this removes the Render warning
        if not raw:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        text = raw.decode("utf-8", errors="replace")
        stream = io.StringIO(text)
        stream.seek(0)  # safe for any downstream 'seek'
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # 2) Call your util with the **sync file-like** (never pass UploadFile)
    try:
        result = await maybe_await(score_calllogs_from_csv(stream))
    except HTTPException:
        raise
    except Exception as e:
        # If your util itself expects a string and tries splitlines on a coroutine,
        # this keeps the error readable for Swagger users.
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # 3) Normalize output (dict OR tuple supported)
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

    # 4) Fallback decision thresholds if util returned only a score
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