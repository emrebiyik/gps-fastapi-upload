# main.py
from __future__ import annotations

import io
import os
import csv
from datetime import datetime, timezone  # <-- timezone üstte
from typing import Any, Dict, List, Optional, Tuple
from collections import Counter

from fastapi import (
    FastAPI,
    Depends,
    HTTPException,
    UploadFile,
    File,
    Form,
    Path,
    status,
)
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# --- Auth (matches your auth.py) ---
from auth import router as auth_router, get_current_user

# --- Utils (single source of truth) ---
from utils import (
    extract_gps_pillow,     # UploadFile -> (lat, lon) | None  (sync)
    haversine_distance,     # (lat1, lon1, lat2, lon2) -> float (km)
    score_calllogs_from_csv # (io.TextIOBase) -> dict (sync)
)

# Optional reverse geocoding (lat/lon -> city, country)
try:
    # signature: async def reverse_geocode_city(lat: float, lon: float) -> Tuple[Optional[str], Optional[str]]
    from utils import reverse_geocode_city  # type: ignore
except Exception:
    reverse_geocode_city = None  # gracefully degrade if not provided

# ============================================================
# App & CORS
# ============================================================
app = FastAPI(title="Credit Scoring API", version="1.2.0")

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
    city: Optional[str] = None
    country: Optional[str] = None

class GPSItem(BaseModel):
    index: int
    gps_id: Optional[int] = None
    filename: str
    latitude: float
    longitude: float
    distance_km: float
    flag: Optional[str] = None  # "normal" | "abnormal" | "no_gps"
    city: Optional[str] = None
    country: Optional[str] = None

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

# GPS scoring input models (city/country optional; will be enriched if missing)
class GPSPointIn(BaseModel):
    ts: datetime = Field(..., description="ISO8601 timestamp with timezone, e.g., 2025-09-01T12:34:56Z")
    lat: float
    lon: float
    city: Optional[str] = None
    country: Optional[str] = None

class GPSScoreIn(BaseModel):
    points: List[GPSPointIn] = Field(..., description="List of GPS points")

# ============================================================
# Settings
# ============================================================
ABNORMAL_THRESHOLD_KM = float(os.getenv("GPS_ABNORMAL_KM", "1.0"))
APPROVE_MIN = float(os.getenv("APPROVE_MIN", "60"))
REVIEW_MIN  = float(os.getenv("REVIEW_MIN",  "40"))

# --- GPS scoring configurable knobs ---
HOME_RADIUS_M = int(os.getenv("GPS_HOME_RADIUS_M", "250"))
WORK_RADIUS_M = int(os.getenv("GPS_WORK_RADIUS_M", "500"))
TRIP_HOP_KM   = float(os.getenv("GPS_TRIP_HOP_KM", "1.0"))
IMPOSSIBLE_SPEED_KMH = float(os.getenv("GPS_IMPOSSIBLE_SPEED_KMH", "250"))
MAX_POINTS = int(os.getenv("GPS_MAX_POINTS", "10000"))

# Reverse geocoding toggle
REVERSE_GEOCODE_ENABLED = os.getenv("REVERSE_GEOCODE_ENABLED", "true").lower() in {"1","true","yes"}

# ============================================================
# Helpers
# ============================================================
def _caller_id(current_user: dict) -> str:
    """Support both {'sub': ...} and {'username': ...} from auth."""
    return current_user.get("sub") or current_user.get("username") or ""

async def _maybe_reverse_geocode(lat: float, lon: float) -> Tuple[Optional[str], Optional[str]]:
    """Soft wrapper for reverse geocoding: returns (city, country) or (None, None)."""
    if not REVERSE_GEOCODE_ENABLED or reverse_geocode_city is None:
        return (None, None)
    try:
        return await reverse_geocode_city(lat, lon)
    except Exception:
        return (None, None)

def _mode_or_none(values: List[Optional[str]]) -> Optional[str]:
    vals = [v for v in values if v]
    if not vals:
        return None
    count = Counter(vals)
    return count.most_common(1)[0][0]

# ============================================================
# Routes
# ============================================================
@app.get("/health", tags=["default"])
def health() -> Dict[str, str]:
    return {"status": "ok"}

# ---------------- Image ingest -> GPS + optional city/country ----------------
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
    Distances are relative to the first image (not hop-to-hop).
    Adds city/country when reverse geocoder is available.
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    first = files[0]
    # extract GPS
    try:
        latlon = extract_gps_pillow(first)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to read EXIF GPS: {e}")
    if not latlon:
        raise HTTPException(status_code=400, detail="First image has no EXIF GPS")

    ref_lat, ref_lon = map(float, latlon)
    ref_city, ref_country = await _maybe_reverse_geocode(ref_lat, ref_lon)

    reference = GPSReference(
        filename=first.filename,
        latitude=ref_lat,
        longitude=ref_lon,
        city=ref_city,
        country=ref_country,
    )

    items: List[GPSItem] = [
        GPSItem(
            index=0,
            gps_id=None,
            filename=first.filename,
            latitude=ref_lat,
            longitude=ref_lon,
            distance_km=0.0,
            flag="normal",
            city=ref_city,
            country=ref_country,
        )
    ]

    for idx, up in enumerate(files[1:], start=1):
        try:
            latlon = extract_gps_pillow(up)
        except Exception:
            latlon = None

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
                    city=None,
                    country=None,
                )
            )
            continue

        lat, lon = map(float, latlon)
        try:
            dist_km = float(haversine_distance(ref_lat, ref_lon, lat, lon))
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Failed to compute distance: {e}")

        item_city, item_country = await _maybe_reverse_geocode(lat, lon)

        items.append(
            GPSItem(
                index=idx,
                gps_id=None,
                filename=up.filename,
                latitude=lat,
                longitude=lon,
                distance_km=round(dist_km, 2),
                flag="abnormal" if dist_km > ABNORMAL_THRESHOLD_KM else "normal",
                city=item_city,
                country=item_country,
            )
        )

    return GPSIngestResult(user_id=user_id, reference=reference, items=items)

# ---------------- Calllogs CSV scoring ----------------
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
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")

    if calllogs_csv is None:
        raise HTTPException(status_code=400, detail="Missing calllogs_csv")

    # Read upload ASYNC, decode, wrap in StringIO
    try:
        raw: bytes = await calllogs_csv.read()
        if not raw:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        text = raw.decode("utf-8", errors="replace")
        stream = io.StringIO(text)
        stream.seek(0)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # Call util with the sync file-like
    try:
        result = score_calllogs_from_csv(stream)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # Normalize output
    if isinstance(result, dict):
        score = float(result.get("score"))
        decision = str(result.get("decision") or "")
        awarded = result.get("awarded", {}) or {}
        details = result.get("metrics", {}) or result.get("details", {}) or {}
    else:
        raise HTTPException(status_code=400, detail="Unexpected scoring output format")

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

# ---------------- GPS CSV scoring ----------------
@app.post(
    "/users/{user_id}/score/gps_csv",
    response_model=ScoreOut,
    tags=["scoring"],
    summary="Score From GPS CSV"
)
async def score_from_gps_csv_endpoint(
    user_id: str,
    loan_id: str = Form(..., description="Loan/application id"),
    gps_csv: UploadFile = File(..., description="CSV with columns: ts, lat, lon, [city], [country]"),
    current_user: dict = Depends(get_current_user),
):
    """
    Accepts a CSV file of GPS points and computes the GPS-only score.
    Expected headers (case-insensitive):
      - ts        : ISO8601 (e.g. 2025-09-01T12:34:56Z) or epoch ms
      - lat, lon  : floats
      - city      : optional
      - country   : optional (ISO alpha-2 or name)
    """
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Forbidden: user mismatch")

    if gps_csv is None:
        raise HTTPException(status_code=400, detail="Missing gps_csv")

    # read file -> text
    try:
        raw: bytes = await gps_csv.read()
        if not raw:
            raise HTTPException(status_code=400, detail="CSV file is empty")
        text = raw.decode("utf-8", errors="replace")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid CSV content: {e}")

    # parse CSV -> List[GPSPointIn]
    def _parse_ts(s: str) -> datetime:
        s = (s or "").strip()
        # epoch ms?
        if s.isdigit():
            try:
                ms = int(s)
                return datetime.fromtimestamp(ms / 1000.0, tz=timezone.utc)
            except Exception:
                pass
        # ISO8601
        try:
            if s.endswith("Z"):
                s = s.replace("Z", "+00:00")
            return datetime.fromisoformat(s)
        except Exception:
            raise HTTPException(status_code=400, detail=f"Invalid timestamp: {s}")

    try:
        rows = list(csv.DictReader(text.splitlines()))
        if not rows:
            raise HTTPException(status_code=400, detail="CSV has no rows")
        points: List[GPSPointIn] = []
        for r in rows:
            ts_raw = r.get("ts") or r.get("timestamp") or r.get("time")
            lat_raw = r.get("lat") or r.get("latitude")
            lon_raw = r.get("lon") or r.get("longitude") or r.get("lng")
            if ts_raw is None or lat_raw is None or lon_raw is None:
                continue  # skip incomplete row
            ts = _parse_ts(str(ts_raw))
            try:
                lat = float(str(lat_raw))
                lon = float(str(lon_raw))
            except Exception:
                continue
            city = r.get("city") or None
            country = r.get("country") or None
            points.append(GPSPointIn(ts=ts, lat=lat, lon=lon, city=city, country=country))
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to parse CSV: {e}")

    # compute score (reuse existing logic)
    result = await compute_gps_score_async(points)

    # map to ScoreOut
    gps_score = float(result.get("gps_score", 0))
    if gps_score >= APPROVE_MIN:
        decision = "APPROVE"
    elif gps_score >= REVIEW_MIN:
        decision = "REVIEW"
    else:
        decision = "REJECT"

    details = result.get("details", {})
    if "reason" in result:
        details = {**details, "reason": result["reason"]}

    return ScoreOut(
        loan_id=loan_id,
        score=gps_score,
        decision=decision,
        awarded={},   # no per-rule breakdown yet
        details=details,
    )

# ============================================================
# GPS-only scoring (JSON input, no DB)
# ============================================================
from statistics import median
from math import radians, sin, cos, asin, sqrt

def _haversine_km_tuple(a: Tuple[float,float], b: Tuple[float,float]) -> float:
    lat1, lon1, lat2, lon2 = map(radians, [a[0], a[1], b[0], b[1]])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    r = 6371.0
    h = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * r * asin(sqrt(h))

def _centroid(points: List[Tuple[float,float]]) -> Tuple[float,float]:
    return (sum(p[0] for p in points)/len(points), sum(p[1] for p in points)/len(points))

def _to_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)

async def _ensure_cities(points: List[GPSPointIn]) -> List[GPSPointIn]:
    """Fill city/country for points that miss them (if reverse geocoder is available)."""
    if not REVERSE_GEOCODE_ENABLED or reverse_geocode_city is None:
        return points
    enriched: List[GPSPointIn] = []
    for p in points:
        if p.city and p.country:
            enriched.append(p)
            continue
        city, country = await _maybe_reverse_geocode(p.lat, p.lon)
        enriched.append(GPSPointIn(ts=p.ts, lat=p.lat, lon=p.lon,
                                   city=p.city or city, country=p.country or country))
    return enriched

def _dominant_city(points: List[GPSPointIn]) -> Optional[str]:
    return _mode_or_none([p.city for p in points])

def _dominant_country(points: List[GPSPointIn]) -> Optional[str]:
    return _mode_or_none([p.country for p in points])

async def compute_gps_score_async(points: List[GPSPointIn]) -> Dict[str, Any]:
    # manual validation (pydantic v1/v2 safe)
    if len(points) > MAX_POINTS:
        raise HTTPException(status_code=413, detail=f"Too many points (>{MAX_POINTS})")
    for p in points:
        if not (-90.0 <= p.lat <= 90.0) or not (-180.0 <= p.lon <= 180.0):
            raise HTTPException(status_code=422, detail="lat/lon out of range")

    if len(points) < 5:
        return {"gps_score": 0, "risk": "High", "reason": "insufficient_gps_data"}

    # Optionally enrich city/country
    points = await _ensure_cities(points)

    # Sort and normalize to UTC
    pts = sorted(
        (GPSPointIn(ts=_to_utc(p.ts), lat=p.lat, lon=p.lon, city=p.city, country=p.country) for p in points),
        key=lambda p: p.ts
    )

    # Windows in UTC
    night = [p for p in pts if p.ts.hour >= 22 or p.ts.hour < 6]
    work  = [p for p in pts if 9 <= p.ts.hour < 18 and p.ts.weekday() < 5]

    def coverage_ratio(sub: List[GPSPointIn], radius_m: int) -> float:
        if len(sub) < 3:
            return 0.0
        c = _centroid([(p.lat, p.lon) for p in sub])
        within = sum(1 for p in sub if _haversine_km_tuple((p.lat,p.lon), c)*1000 <= radius_m)
        return within / len(sub)

    home_cov = coverage_ratio(night, HOME_RADIUS_M)
    work_cov = coverage_ratio(work, WORK_RADIUS_M)

    # Daily radius median
    by_day: Dict[str, List[Tuple[float,float]]] = {}
    for p in pts:
        k = p.ts.strftime("%Y-%m-%d")
        by_day.setdefault(k, []).append((p.lat, p.lon))

    daily_radii = []
    for coords in by_day.values():
        c = _centroid(coords)
        dists = [_haversine_km_tuple(c, xy) for xy in coords]
        daily_radii.append(max(dists) if dists else 0.0)
    daily_radius_med = median(daily_radii) if daily_radii else 0.0

    # Trips normalized by window length (~per week)
    total_hours = (pts[-1].ts - pts[0].ts).total_seconds() / 3600.0
    trips = sum(1 for a, b in zip(pts, pts[1:]) if _haversine_km_tuple((a.lat,a.lon),(b.lat,b.lon)) > TRIP_HOP_KM)
    trips_per_week = trips if total_hours <= 0 else trips * (24*7) / total_hours

    # Impossible jumps
    imp = 0
    for a, b in zip(pts, pts[1:]):
        dt_h = (b.ts - a.ts).total_seconds()/3600.0
        if dt_h > 0 and (_haversine_km_tuple((a.lat,a.lon),(b.lat,b.lon)) / dt_h) > IMPOSSIBLE_SPEED_KMH:
            imp += 1
    imp_rate = imp / max(1, len(pts)-1)

    # Scoring
    score = 0
    score += 25 if home_cov >= 0.70 else (15 if home_cov >= 0.40 else 5)
    score += 20 if work_cov >= 0.60 else (10 if work_cov >= 0.30 else 0)
    score += 15 if daily_radius_med <= 3 else (8 if daily_radius_med <= 10 else 3)

    if 3 <= trips_per_week <= 10:
        score += 10
    elif 1 <= trips_per_week <= 2:
        score += 6
    elif trips_per_week >= 11:
        score += 4

    if imp_rate > 0.03:
        score -= 20
    elif imp_rate > 0.01:
        score -= 10

    score_rounded = max(0, min(100, round(score, 0)))
    risk = "Low" if score_rounded >= 70 else ("Medium" if score_rounded >= 50 else "High")

    details_extra: Dict[str, Any] = {
        "home_city": _dominant_city(night),
        "home_country": _dominant_country(night),
        "work_city": _dominant_city(work),
        "work_country": _dominant_country(work),
        "distinct_cities_last_window": len({p.city for p in pts if p.city}),
        "distinct_countries_last_window": len({p.country for p in pts if p.country}),
    }

    return {
        "gps_score": score_rounded,
        "risk": risk,
        "details": {
            "home_coverage": round(home_cov, 2),
            "work_coverage": round(work_cov, 2),
            "daily_radius_median_km": round(daily_radius_med, 2),
            "trips": int(trips),
            "trips_per_week": round(trips_per_week, 2),
            "impossible_jump_rate": round(imp_rate, 4),
            **details_extra,
        }
    }

# ---- Single public endpoint (scoring namespace) ----
@app.post(
    "/users/{user_id}/score/gps",
    summary="Compute GPS-only credit score (scoring namespace)",
    tags=["scoring"]
)
async def score_gps_under_scoring(
    user_id: str = Path(..., description="Owner of the points; must match token"),
    payload: GPSScoreIn = ...,
    current_user: dict = Depends(get_current_user),
):
    caller = _caller_id(current_user)
    if user_id != caller:
        raise HTTPException(status_code=403, detail="Forbidden: user mismatch")
    return await compute_gps_score_async(payload.points)