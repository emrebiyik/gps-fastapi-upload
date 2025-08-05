from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from PIL import Image, ExifTags
from database import SessionLocal
from models import ImageGPSData
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def get_decimal_from_dms(dms, ref):
    degrees = dms[0][0] / dms[0][1]
    minutes = dms[1][0] / dms[1][1]
    seconds = dms[2][0] / dms[2][1]
    decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
    if ref in ['S', 'W']:
        decimal = -decimal
    return decimal

def extract_gps_pillow(file: UploadFile):
    try:
        file.file.seek(0)
        image = Image.open(file.file)
        exif_data = image._getexif()

        if not exif_data:
            return None

        gps_info = {}
        for tag, value in exif_data.items():
            decoded = ExifTags.TAGS.get(tag)
            if decoded == "GPSInfo":
                gps_info = value
                break

        if not gps_info:
            return None

        gps_lat = gps_info.get(2)
        gps_lat_ref = gps_info.get(1)
        gps_lon = gps_info.get(4)
        gps_lon_ref = gps_info.get(3)

        if gps_lat and gps_lat_ref and gps_lon and gps_lon_ref:
            lat = get_decimal_from_dms(gps_lat, gps_lat_ref)
            lon = get_decimal_from_dms(gps_lon, gps_lon_ref)
            return {"latitude": lat, "longitude": lon}
    except Exception as e:
        print(f"Error extracting GPS: {e}")
    return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)

    a = math.sin(dphi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

@app.post("/upload-images")
async def upload_images(images: List[UploadFile] = File(...)):
    gps_info_list = []
    reference_lat = None
    reference_lon = None

    db = SessionLocal()

    for idx, image in enumerate(images):
        gps = extract_gps_pillow(image)
        if gps:
            lat = gps["latitude"]
            lon = gps["longitude"]

            if idx == 0:
                reference_lat = lat
                reference_lon = lon
                distance_km = 0
                flag = "normal"
            else:
                distance_km = haversine(reference_lat, reference_lon, lat, lon)
                flag = "abnormal" if distance_km > 1.0 else "normal"

            # Veritabanına kaydet
            record = ImageGPSData(
                filename=image.filename,
                latitude=lat,
                longitude=lon,
                distance_km=distance_km,
                flag=flag
            )
            db.add(record)
            db.commit()

            gps_info_list.append({
                "filename": image.filename,
                "gps": {"latitude": lat, "longitude": lon},
                "distance_from_home_km": round(distance_km, 3),
                "flag": flag
            })
        else:
            gps_info_list.append({
                "filename": image.filename,
                "gps": None,
                "distance_from_home_km": None,
                "flag": "abnormal"
            })

    db.close()
    return {"status": "ok", "gps_info": gps_info_list}