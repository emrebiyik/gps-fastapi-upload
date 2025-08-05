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

def extract_gps_info(image_file):
    image_file.file.seek(0)
    image = Image.open(image_file.file)

    exif_data = image.getexif()
    if not exif_data:
        return None

    gps_info = {}
    for key, val in exif_data.items():
        tag = TAGS.get(key)
        if tag == "GPSInfo":
            for gps_key in val:
                gps_tag = GPSTAGS.get(gps_key)
                gps_info[gps_tag] = val[gps_key]

    if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
        def convert_to_decimal(coord, ref):
            degrees, minutes, seconds = coord
            decimal = degrees + minutes / 60 + seconds / 3600
            if ref in ["S", "W"]:
                decimal = -decimal
            return float(decimal)

        lat = convert_to_decimal(gps_info["GPSLatitude"], gps_info["GPSLatitudeRef"])
        lon = convert_to_decimal(gps_info["GPSLongitude"], gps_info["GPSLongitudeRef"])
        return {"latitude": lat, "longitude": lon}

    return None

def calculate_distance_km(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
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

    for idx, image in enumerate(images):
        gps = extract_gps_info(image)

        if gps:
            conn = sqlite3.connect(DB_FILE)
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO gps_images (filename, latitude, longitude) VALUES (?, ?, ?)",
                (image.filename, gps["latitude"], gps["longitude"])
            )
            conn.commit()
            conn.close()

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
                "gps": None
            })

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
    return {"message": "GPS FastAPI is up and running."}