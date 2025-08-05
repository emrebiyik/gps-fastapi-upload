from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
import exifread
import json
import math
import sqlite3
import os

app = FastAPI()

DB_FILE = "gps_data.db"
if not os.path.exists(DB_FILE):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gps_images (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            filename TEXT,
            latitude REAL,
            longitude REAL
        )
    ''')
    conn.commit()
    conn.close()


def dms_to_decimal(dms, ref):
    try:
        degrees = float(dms[0].num) / float(dms[0].den)
        minutes = float(dms[1].num) / float(dms[1].den)
        seconds = float(dms[2].num) / float(dms[2].den)
        decimal = degrees + minutes / 60 + seconds / 3600
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except Exception as e:
        print(f"[ERROR] DMS conversion failed: {e}")
        return None


def extract_gps_with_exifread(file: UploadFile):
    try:
        file.file.seek(0)
        tags = exifread.process_file(file.file, details=False)

        lat = tags.get("GPS GPSLatitude")
        lat_ref = tags.get("GPS GPSLatitudeRef")
        lon = tags.get("GPS GPSLongitude")
        lon_ref = tags.get("GPS GPSLongitudeRef")

        if lat and lat_ref and lon and lon_ref:
            latitude = dms_to_decimal(lat.values, lat_ref.values)
            longitude = dms_to_decimal(lon.values, lon_ref.values)
            return {"latitude": latitude, "longitude": longitude}
    except Exception as e:
        print(f"[ERROR] Failed to extract EXIF GPS: {e}")
        return None

    return None


def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2.0)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0)**2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


@app.post("/upload-images")
async def upload_images(
    images: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    gps_results = []
    reference_coords = None
    status_flag = "normal"

    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()

    for image in images:
        gps = extract_gps_with_exifread(image)

        if gps:
            cursor.execute(
                "INSERT INTO gps_images (filename, latitude, longitude) VALUES (?, ?, ?)",
                (image.filename, gps["latitude"], gps["longitude"])
            )
            conn.commit()

            if reference_coords is None:
                reference_coords = gps
                gps_results.append({
                    "filename": image.filename,
                    "gps": gps
                })
            else:
                distance = calculate_distance_km(
                    reference_coords["latitude"], reference_coords["longitude"],
                    gps["latitude"], gps["longitude"]
                )
                gps_results.append({
                    "filename": image.filename,
                    "gps": gps,
                    "distance_from_reference_km": round(distance, 2)
                })
                if distance > 1:
                    status_flag = "abnormal"
        else:
            gps_results.append({
                "filename": image.filename,
                "gps": None,
                "note": "No readable GPS data found"
            })

    conn.close()

    metadata_info = {}
    if metadata:
        try:
            metadata_info = json.loads(metadata)
        except Exception:
            metadata_info["error"] = "Invalid metadata format"

    return JSONResponse(content={
        "status": status_flag,
        "gps_info": gps_results,
        "metadata": metadata_info
    })


@app.get("/")
def read_root():
    return {"message": "GPS FastAPI (with exifread) is up and running."}