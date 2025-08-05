from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List, Optional
from PIL import Image, ExifTags
import math
import json
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


def extract_gps_pillow(file: UploadFile):
    try:
        file.file.seek(0)
        image = Image.open(file.file)
        exif_data = image._getexif()

        if not exif_data:
            print("[WARN] No EXIF data found.")
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id)
            if tag == "GPSInfo":
                for key in value:
                    gps_tag = ExifTags.GPSTAGS.get(key)
                    gps_info[gps_tag] = value[key]

        if "GPSLatitude" in gps_info and "GPSLongitude" in gps_info:
            def convert_to_decimal(coord, ref):
                try:
                    degrees, minutes, seconds = coord
                    decimal = degrees[0] / degrees[1] + \
                              minutes[0] / minutes[1] / 60 + \
                              seconds[0] / seconds[1] / 3600
                    if ref in ["S", "W"]:
                        decimal = -decimal
                    return round(decimal, 6)
                except Exception as e:
                    print(f"[ERROR] Decimal conversion failed: {e}")
                    return None

            lat = convert_to_decimal(gps_info["GPSLatitude"], gps_info.get("GPSLatitudeRef", "N"))
            lon = convert_to_decimal(gps_info["GPSLongitude"], gps_info.get("GPSLongitudeRef", "E"))

            if lat is not None and lon is not None:
                return {"latitude": lat, "longitude": lon}
            else:
                print("[WARN] Latitude or Longitude conversion failed.")
                return None

        print("[WARN] No GPS tags found in EXIF.")
        return None

    except Exception as e:
        print(f"[ERROR] Pillow GPS extraction failed: {e}")
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
        gps = extract_gps_pillow(image)

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
                "note": "No GPS data found"
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
def root():
    return {"message": "GPS FastAPI (Pillow-based) is running."}