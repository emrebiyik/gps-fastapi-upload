import math
from fastapi import HTTPException
from sqlalchemy.orm import Session
from models import User
from PIL import Image, ExifTags


def get_or_create_user(db: Session, user_id: str) -> User:
    user = db.query(User).filter(User.user_id == user_id).first()
    if not user:
        user = User(user_id=user_id)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user


def get_decimal_from_dms(dms, ref):
    """Convert GPS coordinates from DMS format to decimal degrees."""
    degrees, minutes, seconds = [float(x) for x in dms]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ["S", "W"]:
        decimal = -decimal
    return decimal


def extract_gps_pillow(file) -> tuple | None:
    """Extract GPS latitude and longitude from image EXIF using Pillow."""
    try:
        image = Image.open(file.file)
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag, tag)
            if decoded == "GPSInfo":
                gps_info = {
                    ExifTags.GPSTAGS.get(t, t): v for t, v in value.items()
                }

        if not gps_info:
            return None

        lat = get_decimal_from_dms(
            gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"]
        )
        lon = get_decimal_from_dms(
            gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"]
        )
        return (lat, lon)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"GPS extraction failed: {e}")


def haversine_distance(lat1, lon1, lat2, lon2) -> float:
    """Calculate distance between two coordinates in km using Haversine formula."""
    R = 6371  # Earth radius in km
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = math.sin(d_lat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(
        math.radians(lat2)
    ) * math.sin(d_lon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c