from __future__ import annotations
from datetime import datetime
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, Depends, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sqlalchemy import desc
from sqlalchemy.orm import Session

from database import get_db, engine, Base
from models import (
    User, GPSData, ImageAsset,
    BankMetrics, MobileMoneyMetrics, CallLogMetrics,
    CreditScore
)

# ----------------- App & CORS -----------------
app = FastAPI(title="Credit Scoring API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],            # prod'da kısıtla
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------- Ensure Tables On Startup -----------------
@app.on_event("startup")
def _ensure_tables():
    """
    Shell kullanmadan, servis her ayağa kalktığında tabloların varlığını garanti eder.
    Idempotent: yoksa oluşturur, varsa dokunmaz.
    """
    try:
        # MODELLERİN YÜKLÜ OLDUĞUNDAN EMİN OLMAK İÇİN import models üstte.
        Base.metadata.create_all(bind=engine)
        print("✅ Ensured DB tables exist.")
    except Exception as e:
        print("⚠️ DB init error:", e)

# ----------------- Utilities -----------------
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

def row2dict(row):
    return {c.name: getattr(row, c.name) for c in row.__table__.columns} if row else None

# ----------------- Health -----------------
@app.get("/")
def read_root():
    return {"message": "Credit Scoring API is up and running."}

# ----------------- Schemas -----------------
class BankIn(BaseModel):
    external_user_id: str = Field(..., description="Upstream user id")
    income_avg_3m: float | None = None
    average_balance: float | None = None
    net_cash_flow_90d: float | None = None
    bounced_txn_90d: int | None = None
    overdraft_days_90d: int | None = None
    statement_period_days: int | None = None

class MobileIn(BaseModel):
    external_user_id: str
    mm_txn_90d: int | None = None
    mm_volume_90d: float | None = None
    mm_active_days_90d: int | None = None
    avg_ticket_90d: float | None = None
    last_txn_at: str | None = None   # ISO-8601

class CallLogsIn(BaseModel):
    external_user_id: str
    unique_contacts_30d: int | None = None
    call_days_30d: int | None = None
    incoming_outgoing_ratio_30d: float | None = None
    airtime_spend_30d: float | None = None

class AssetItem(BaseModel):
    external_user_id: str
    asset_type: str
    estimated_value: float | None = None
    image_verified: bool = False
    source_image: str | None = None

class AssetsIn(BaseModel):
    assets: List[AssetItem]

class ScoreIn(BaseModel):
    external_user_id: str

# ----------------- Ingest: Bank -----------------
@app.post("/api/v1/bank/ingest")
def ingest_bank(payload: BankIn, db: Session = Depends(get_db)):
    try:
        user = get_or_create_user(db, payload.external_user_id)
        row = BankMetrics(
            user_id=user.id,
            income_avg_3m=payload.income_avg_3m,
            average_balance=payload.average_balance,
            net_cash_flow_90d=payload.net_cash_flow_90d,
            bounced_txn_90d=payload.bounced_txn_90d,
            overdraft_days_90d=payload.overdraft_days_90d,
            statement_period_days=payload.statement_period_days,
        )
        db.add(row)
        db.commit()
        return {"status": "ok", "saved": 1, "user_id": user.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Ingest: Mobile Money -----------------
@app.post("/api/v1/mobile/ingest")
def ingest_mobile(payload: MobileIn, db: Session = Depends(get_db)):
    try:
        user = get_or_create_user(db, payload.external_user_id)
        last_txn_dt = None
        if payload.last_txn_at:
            last_txn_dt = datetime.fromisoformat(payload.last_txn_at.replace("Z", ""))
        row = MobileMoneyMetrics(
            user_id=user.id,
            mm_txn_90d=payload.mm_txn_90d,
            mm_volume_90d=payload.mm_volume_90d,
            mm_active_days_90d=payload.mm_active_days_90d,
            avg_ticket_90d=payload.avg_ticket_90d,
            last_txn_at=last_txn_dt,
        )
        db.add(row)
        db.commit()
        return {"status": "ok", "saved": 1, "user_id": user.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Ingest: Call Logs -----------------
@app.post("/api/v1/calllogs/ingest")
def ingest_calllogs(payload: CallLogsIn, db: Session = Depends(get_db)):
    try:
        user = get_or_create_user(db, payload.external_user_id)
        row = CallLogMetrics(
            user_id=user.id,
            unique_contacts_30d=payload.unique_contacts_30d,
            call_days_30d=payload.call_days_30d,
            incoming_outgoing_ratio_30d=payload.incoming_outgoing_ratio_30d,
            airtime_spend_30d=payload.airtime_spend_30d,
        )
        db.add(row)
        db.commit()
        return {"status": "ok", "saved": 1, "user_id": user.id}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Ingest: Image Assets -----------------
@app.post("/api/v1/assets/ingest")
def ingest_assets(payload: AssetsIn, db: Session = Depends(get_db)):
    try:
        saved = 0
        for item in payload.assets:
            user = get_or_create_user(db, item.external_user_id)
            row = ImageAsset(
                user_id=user.id,
                asset_type=item.asset_type,
                estimated_value=item.estimated_value,
                image_verified=item.image_verified,
                source_image=item.source_image,
            )
            db.add(row)
            saved += 1
        db.commit()
        return {"status": "ok", "saved": saved}
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(e))

# ----------------- Ingest: GPS (EXIF) -----------------
from PIL import Image, ExifTags

def _dms_to_decimal(dms, ref):
    deg = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]
    val = deg + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        val = -val
    return val

@app.post("/api/v1/gps/ingest")
async def ingest_gps(
    images: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None),
    db: Session = Depends(get_db)
):
    import json
    meta = json.loads(metadata) if metadata else {}
    external_user_id = meta.get("external_user_id")
    if not external_user_id:
        raise HTTPException(status_code=400, detail="external_user_id missing in metadata")
    user = get_or_create_user(db, external_user_id)

    reference_lat = meta.get("reference_lat")
    reference_lon = meta.get("reference_lon")

    saved = 0
    items = []
    for file in images:
        try:
            file.file.seek(0)
            image = Image.open(file.file)
            exif = image._getexif() if hasattr(image, "_getexif") else None
            gps_lat = gps_lon = None
            taken_at = None

            if exif:
                exif_data = {ExifTags.TAGS.get(k, k): v for k, v in exif.items()}
                gps_info = exif_data.get("GPSInfo")
                if gps_info:
                    gps_parsed = {ExifTags.GPSTAGS.get(t, t): gps_info[t] for t in gps_info}
                    if all(k in gps_parsed for k in ["GPSLatitude", "GPSLatitudeRef", "GPSLongitude", "GPSLongitudeRef"]):
                        gps_lat = _dms_to_decimal(gps_parsed["GPSLatitude"], gps_parsed["GPSLatitudeRef"])
                        gps_lon = _dms_to_decimal(gps_parsed["GPSLongitude"], gps_parsed["GPSLongitudeRef"])
                if "DateTimeOriginal" in exif_data:
                    try:
                        taken_at = datetime.strptime(exif_data["DateTimeOriginal"], "%Y:%m:%d %H:%M:%S")
                    except Exception:
                        taken_at = None

            distance_km = None
            flag = None
            if gps_lat is not None and gps_lon is not None and reference_lat and reference_lon:
                from math import radians, sin, cos, sqrt, atan2
                R = 6371.0
                dlat = radians(reference_lat - gps_lat)
                dlon = radians(reference_lon - gps_lon)
                a = sin(dlat/2)**2 + cos(radians(gps_lat))*cos(radians(reference_lat))*sin(dlon/2)**2
                c = 2*atan2(sqrt(a), sqrt(1-a))
                distance_km = R * c
                flag = "abnormal" if distance_km > 100 else "normal"

            row = GPSData(
                user_id=user.id,
                filename=file.filename,
                latitude=gps_lat,
                longitude=gps_lon,
                distance_km=distance_km,
                flag=flag,
                taken_at=taken_at
            )
            db.add(row)
            saved += 1

            items.append({
                "user_id": external_user_id,
                "filename": file.filename,
                "latitude": gps_lat,
                "longitude": gps_lon,
                "distance_km": distance_km,
                "flag": flag,
                "taken_at": taken_at.isoformat() if taken_at else None
            })
        except Exception as e:
            items.append({
                "user_id": external_user_id,
                "filename": getattr(file, "filename", None),
                "error": str(e)
            })

    db.commit()
    return {"status": "ok", "saved": saved, "items": items}

# ----------------- Feature Snapshot -----------------
@app.get("/api/v1/users/{external_user_id}/features")
def get_user_features(external_user_id: str, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.external_user_id == external_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    latest_bank = db.query(BankMetrics).filter_by(user_id=user.id).order_by(desc(BankMetrics.created_at)).first()
    latest_mobile = db.query(MobileMoneyMetrics).filter_by(user_id=user.id).order_by(desc(MobileMoneyMetrics.created_at)).first()
    latest_call  = db.query(CallLogMetrics).filter_by(user_id=user.id).order_by(desc(CallLogMetrics.created_at)).first()
    latest_gps   = db.query(GPSData).filter_by(user_id=user.id).order_by(desc(GPSData.created_at)).first()
    latest_assets = db.query(ImageAsset).filter_by(user_id=user.id).order_by(desc(ImageAsset.created_at)).all()

    return {
        "external_user_id": external_user_id,
        "bank": row2dict(latest_bank),
        "mobile": row2dict(latest_mobile),
        "calllogs": row2dict(latest_call),
        "gps": row2dict(latest_gps),
        "assets": [row2dict(a) for a in latest_assets] if latest_assets else []
    }

# ----------------- Scoring -----------------
def score_bank(feat: Dict[str, Any] | None) -> int:
    if not feat:
        return 0
    s = 0
    if (feat.get("average_balance") or 0) >= 1500: s += 3
    if (feat.get("net_cash_flow_90d") or 0) > 0: s += 2
    if (feat.get("bounced_txn_90d") or 0) >= 1: s -= 2
    if (feat.get("overdraft_days_90d") or 0) > 5: s -= 3
    if (feat.get("income_avg_3m") or 0) >= 800: s += 3
    return s

def score_mobile(feat: Dict[str, Any] | None) -> int:
    if not feat:
        return 0
    s = 0
    if (feat.get("mm_txn_90d") or 0) >= 30: s += 2
    if (feat.get("mm_volume_90d") or 0) >= 500: s += 2
    if (feat.get("mm_active_days_90d") or 0) >= 20: s += 1
    if (feat.get("avg_ticket_90d") or 9999) < 2: s -= 1
    return s

def score_calllogs(feat: Dict[str, Any] | None) -> int:
    if not feat:
        return 0
    s = 0
    if (feat.get("unique_contacts_30d") or 0) >= 15: s += 1
    ratio = feat.get("incoming_outgoing_ratio_30d")
    if ratio is not None and ratio < 0.3:
        s -= 1
    return s

def score_assets(asset_list: List[Dict[str, Any]] | None) -> int:
    if not asset_list:
        return 0
    plus = 0
    for a in asset_list:
        if a.get("image_verified") and (a.get("estimated_value") or 0) >= 1000:
            plus += 2
    return min(plus, 4)

def score_gps(feat: Dict[str, Any] | None) -> int:
    if not feat:
        return 0
    return -1 if feat.get("flag") == "abnormal" else 0

def decision_from_score(total: int) -> str:
    if total >= 20: return "approve_500"
    if 15 <= total <= 19: return "approve_400"
    if 10 <= total <= 14: return "approve_150"
    return "deny"

@app.post("/api/v1/score/compute")
def compute_score(payload: ScoreIn, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.external_user_id == payload.external_user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    features = get_user_features(payload.external_user_id, db)

    bank_s = score_bank(features.get("bank"))
    mob_s = score_mobile(features.get("mobile"))
    call_s = score_calllogs(features.get("calllogs"))
    gps_s = score_gps(features.get("gps"))
    assets_s = score_assets(features.get("assets"))

    total = bank_s + mob_s + call_s + gps_s + assets_s
    decision = decision_from_score(total)

    row = CreditScore(
        user_id=user.id,
        score=total,
        decision=decision,
        explanation_json={
            "bank": bank_s, "mobile": mob_s, "calllogs": call_s,
            "gps": gps_s, "assets": assets_s
        }
    )
    db.add(row)
    db.commit()

    return {
        "external_user_id": payload.external_user_id,
        "score": total,
        "decision": decision,
        "explanation": {
            "bank": bank_s, "mobile": mob_s, "calllogs": call_s,
            "gps": gps_s, "assets": assets_s
        }
    }