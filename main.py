from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from typing import List
from database import SessionLocal
from models import ImageGPSData
import exifread
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_gps_exifread(file: UploadFile):
    try:
        file.file.seek(0)
        tags = exifread.process_file(file.file, details=False)

        gps_latitude = tags.get("GPS GPSLatitude")
        gps_latitude_ref = tags.get("GPS GPSLatitudeRef")
        gps_longitude = tags.get("GPS GPSLongitude")
        gps_longitude_ref = tags.get("GPS GPSLongitudeRef")

        if gps_latitude and gps_latitude_ref and gps_longitude and gps_longitude_ref:
            def to_decimal(coord, ref):
                d, m, s = [float(x.num) / float(x.den) for x in coord.values]
                decimal = d + m / 60.0 + s / 3600.0
                if ref.values[0] in ['S', 'W']:
                    decimal = -decimal
                return decimal

            lat = to_decimal(gps_latitude, gps_latitude_ref)
            lon = to_decimal(gps_longitude, gps_longitude_ref)
            return {"latitude": lat, "longitude": lon}
    except Exception as e:
        print(f"EXIF GPS error: {e}")
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
        gps = extract_gps_exifread(image)
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