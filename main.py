from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
import io
import json
import math
import sqlite3
import os

app = FastAPI()

# Database setup
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

def insert_into_db(filename: str, latitude: float, longitude: float):
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute('''
        INSERT INTO gps_images (filename, latitude, longitude)
        VALUES (?, ?, ?)
    ''', (filename, latitude, longitude))
    conn.commit()
    conn.close()

def extract_gps_info(image_bytes: bytes):
    try:
        image = Image.open(io.BytesIO(image_bytes))
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for key in value:
                    decode = GPSTAGS.get(key, key)
                    gps_info[decode] = value[key]

        def to_decimal(coords, ref):
            degrees = coords[0][0] / coords[0][1]
            minutes = coords[1][0] / coords[1][1]
            seconds = coords[2][0] / coords[2][1]
            decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
            if ref in ['S', 'W']:
                decimal *= -1
            return decimal

        lat = to_decimal(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
        lon = to_decimal(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
        return {"latitude": lat, "longitude": lon}
    except:
        return None

def haversine(lat1, lon1, lat2, lon2):
    R = 6371  # km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + \
        math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c

@app.post("/upload-images")
async def upload_images(
    images: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    gps_results = []
    status = "normal"
    base_coords = None

    for idx, image in enumerate(images):
        content = await image.read()
        gps = extract_gps_info(content)
        result = {
            "filename": image.filename,
            "gps": gps
        }

        if gps:
            insert_into_db(image.filename, gps["latitude"], gps["longitude"])

            if idx == 0:
                base_coords = gps
            else:
                distance = haversine(base_coords["latitude"], base_coords["longitude"], gps["latitude"], gps["longitude"])
                result["distance_from_base_km"] = round(distance, 2)
                result["flag"] = "abnormal" if distance > 1.0 else "normal"
                if distance > 1.0:
                    status = "abnormal"
        gps_results.append(result)

    try:
        meta_dict = json.loads(metadata) if metadata else {}
    except json.JSONDecodeError:
        meta_dict = {"error": "Invalid metadata format"}

    return JSONResponse(content={
        "status": status,
        "gps_info": gps_results,
        "metadata": meta_dict
    })