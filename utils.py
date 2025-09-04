# utils.py
from __future__ import annotations

import os
import math
import csv
from datetime import datetime
from typing import Tuple, List, Dict, Any, Optional
from fastapi import HTTPException, UploadFile
from sqlalchemy.orm import Session
from models import User
from PIL import Image, ExifTags

# ------------------------- User helper -------------------------
def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user

# ------------------------- GPS helpers -------------------------
def _to_float(x) -> float:
    """Handle PIL EXIF rationals like (num, den) or PIL.Rational."""
    try:
        num = getattr(x, "numerator", None)
        den = getattr(x, "denominator", None)
        if num is not None and den:
            return float(num) / float(den)
        if isinstance(x, (tuple, list)) and len(x) == 2:
            return float(x[0]) / float(x[1])
        return float(x)
    except Exception:
        return float(x)

def _dms_to_decimal(dms, ref) -> float:
    d, m, s = (_to_float(dms[0]), _to_float(dms[1]), _to_float(dms[2]))
    decimal = d + m/60.0 + s/3600.0
    if ref in ("S", "W"):
        decimal = -decimal
    return decimal

def extract_gps_pillow(file: UploadFile) -> Tuple[float, float] | None:
    """Extract GPS coordinates from image EXIF using Pillow.
    Returns (lat, lon) if available, else None.
    """
    try:
        # Ensure pointer is at start
        try:
            file.file.seek(0)
        except Exception:
            pass

        image = Image.open(file.file)
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            key = ExifTags.TAGS.get(tag, tag)
            if key == "GPSInfo":
                for t in value:
                    sub_key = ExifTags.GPSTAGS.get(t, t)
                    gps_info[sub_key] = value[t]

        if not gps_info:
            return None

        lat = lon = None
        if "GPSLatitude" in gps_info and "GPSLatitudeRef" in gps_info:
            lat = _dms_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        if "GPSLongitude" in gps_info and "GPSLongitudeRef" in gps_info:
            lon = _dms_to_decimal(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])

        return (lat, lon) if (lat is not None and lon is not None) else None
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"GPS extraction failed: {e}")

def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Calculate distance between two coordinates in km using Haversine formula."""
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

# ------------------------- Call-logs scoring -------------------------
def _parse_epoch_ms(v: str | int | float):
    try:
        ms = int(float(v))
        return datetime.utcfromtimestamp(ms / 1000.0)
    except Exception:
        return None

def _read_calllogs_csv(file_obj) -> List[Dict[str, Any]]:
    """Read CSV rows into canonical dicts with keys: number, dt, duration, type."""
    file_obj.seek(0)
    text = file_obj.read()
    if isinstance(text, bytes):
        text = text.decode("utf-8", errors="ignore")
    rows = list(csv.DictReader(text.splitlines()))
    calls = []
    for r in rows:
        dt = _parse_epoch_ms(r.get("Time") or r.get("Date"))
        if not dt:
            continue
        try:
            duration = int(float(r.get("Duration", 0)))
        except Exception:
            duration = 0
        t = r.get("Type")
        call_type = int(t) if t and str(t).isdigit() else 0  # 1/2 real, 3 missed
        calls.append({
            "number": (r.get("Number") or "").strip(),
            "duration": duration,
            "type": call_type,
            "dt": dt,
        })
    return calls

def _metrics_from_calls(calls: List[Dict[str, Any]]) -> Dict[str, Any]:
    from collections import Counter
    if not calls:
        return {
            "total_all": 0, "total_real": 0, "observed_days": 0,
            "calls_per_day": 0.0, "avg_duration": 0.0,
            "stable_contact_ratio": 0.0, "distinct_weekdays": 0,
            "daytime_share": 0.0, "night_share": 0.0, "missed_ratio": 0.0,
        }
    real = [c for c in calls if c["type"] in (1, 2) and c["duration"] > 0]
    total_real = len(real)
    total_all = len(calls)
    days = len({c["dt"].date() for c in calls}) or 1
    calls_per_day = total_real / max(1, days)
    avg_dur = (sum(c["duration"] for c in real) / total_real) if total_real else 0.0

    counts = Counter(c["number"] for c in real if c["number"])
    top5 = sum(n for _, n in counts.most_common(5))
    stable_ratio = (top5 / total_real) if total_real else 0.0

    weekdays = {c["dt"].weekday() for c in real}  # 0=Mon
    hours = [c["dt"].hour for c in real]
    day_calls = sum(1 for h in hours if 8 <= h < 20)
    night_calls = sum(1 for h in hours if h >= 22 or h < 6)
    missed = sum(1 for c in calls if c["type"] == 3)

    return {
        "total_all": total_all,
        "total_real": total_real,
        "observed_days": days,
        "calls_per_day": calls_per_day,
        "avg_duration": avg_dur,
        "stable_contact_ratio": stable_ratio,
        "distinct_weekdays": len(weekdays),
        "daytime_share": (day_calls / total_real) if total_real else 0.0,
        "night_share": (night_calls / total_real) if total_real else 0.0,
        "missed_ratio": (missed / total_all) if total_all else 0.0,
    }

def _get_weight(name: str, default: float) -> float:
    return float(os.getenv(name, default))

def score_calllogs_from_csv(file_obj) -> Dict[str, Any]:
    """Compute a rule-based score from a calllogs CSV file object.

    Returns dict with keys:
      - score (float)
      - decision (str)
      - awarded (dict of criterion->points or labels)
      - metrics (dict with raw computed metrics)
    """
    calls = _read_calllogs_csv(file_obj)
    m = _metrics_from_calls(calls)

    weights = {
        "call_frequency_high": _get_weight("W_CALL_FREQ_HIGH", 20),
        "call_frequency_mid": _get_weight("W_CALL_FREQ_MID", 10),
        "call_duration_high": _get_weight("W_CALL_DUR_HIGH", 15),
        "call_duration_mid": _get_weight("W_CALL_DUR_MID", 8),
        "stable_contact_ratio_max": _get_weight("W_STABLE_CONTACT_MAX", 30),
        "weekday_consistent": _get_weight("W_WEEKDAY_CONSISTENT", 10),
        "weekday_some": _get_weight("W_WEEKDAY_SOME", 5),
        "regular_consistent": _get_weight("W_REGULAR_CONSISTENT", 10),
        "regular_some": _get_weight("W_REGULAR_SOME", 5),
        "night_low": _get_weight("W_NIGHT_LOW", 10),
        "missed_low": _get_weight("W_MISSED_LOW", 10),
    }

    score = 0.0
    awarded: Dict[str, Any] = {}

    # Frequency
    if m["calls_per_day"] >= 5:
        awarded["call_frequency_high"] = weights["call_frequency_high"]; score += awarded["call_frequency_high"]
    elif m["calls_per_day"] >= 2:
        awarded["call_frequency_mid"] = weights["call_frequency_mid"]; score += awarded["call_frequency_mid"]
    else:
        awarded["call_frequency"] = "low"

    # Duration
    if m["avg_duration"] >= 120:
        awarded["call_duration_high"] = weights["call_duration_high"]; score += awarded["call_duration_high"]
    elif m["avg_duration"] >= 60:
        awarded["call_duration_mid"] = weights["call_duration_mid"]; score += awarded["call_duration_mid"]
    else:
        awarded["call_duration"] = "low"

    # Stable contacts (linear up to 0.6)
    cap = 0.6
    val = min(m["stable_contact_ratio"] / cap, 1.0) * weights["stable_contact_ratio_max"]
    awarded["stable_contact_ratio"] = round(val, 2); score += val

    # Weekday spread
    if m["distinct_weekdays"] >= 5:
        awarded["weekday_consistent"] = weights["weekday_consistent"]; score += awarded["weekday_consistent"]
    elif m["distinct_weekdays"] >= 3:
        awarded["weekday_some"] = weights["weekday_some"]; score += awarded["weekday_some"]
    else:
        awarded["weekday"] = "low"

    # Regularity (daytime share)
    if m["daytime_share"] >= 0.8:
        awarded["regular_consistent"] = weights["regular_consistent"]; score += awarded["regular_consistent"]
    elif m["daytime_share"] >= 0.6:
        awarded["regular_some"] = weights["regular_some"]; score += awarded["regular_some"]
    else:
        awarded["regularity"] = "irregular"

    # Night calls low is good
    if m["night_share"] <= 0.10:
        awarded["night_low"] = weights["night_low"]; score += awarded["night_low"]

    # Missed-calls share low is good
    if m["missed_ratio"] <= 0.20:
        awarded["missed_low"] = weights["missed_low"]; score += awarded["missed_low"]

    approve_min = float(os.getenv("APPROVE_MIN", 60))
    review_min = float(os.getenv("REVIEW_MIN", 40))

    decision = "REJECT"
    if score >= approve_min:
        decision = "APPROVE"
    elif score >= review_min:
        decision = "REVIEW"

    return {
        "score": round(score, 2),
        "decision": decision,
        "awarded": awarded,
        "metrics": m,
    }

# ------------------------- Reverse geocoding (optional) -------------------------
# Install httpx in requirements.txt to use this.
import httpx

async def reverse_geocode_city(lat: float, lon: float) -> Tuple[Optional[str], Optional[str]]:
    """
    Reverse geocode (lat, lon) -> (city, country_code_2letters or None).
    Uses Nominatim (OSM). Respect their usage policy & rate limits.
    """
    url = "https://nominatim.openstreetmap.org/reverse"
    params = {"lat": lat, "lon": lon, "format": "jsonv2", "zoom": 10, "addressdetails": 1}
    headers = {"User-Agent": os.getenv("APP_USER_AGENT", "credit-scoring-api/1.0 (contact: you@example.com)")}
    async with httpx.AsyncClient(timeout=8.0) as client:
        r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
    addr = data.get("address", {})
    city = addr.get("city") or addr.get("town") or addr.get("village")
    country = addr.get("country_code")
    return city, (country.upper() if country else None)