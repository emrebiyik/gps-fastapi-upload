from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from typing import List, Optional
from PIL import Image, ExifTags
import shutil
import json
import os
import math

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def read_root():
    return {"message": "GPS FastAPI is up and running."}

def get_decimal_from_dms(dms, ref):
    try:
        degrees = float(dms[0])
        minutes = float(dms[1])
        seconds = float(dms[2])
        decimal = degrees + (minutes / 60.0) + (seconds / 3600.0)
        if ref in ['S', 'W']:
            decimal = -decimal
        return decimal
    except Exception as e:
        return {"error": str(e)}

def extract_gps_data(file_path):
    try:
        image = Image.open(file_path)
        exif_data = image._getexif()
        if not exif_data:
            return None

        gps_info = {}
        for tag_id, value in exif_data.items():
            tag = ExifTags.TAGS.get(tag_id, tag_id)
            if tag == 'GPSInfo':
                for key in value:
                    gps_tag = ExifTags.GPSTAGS.get(key, key)
                    gps_info[gps_tag] = value[key]

        if 'GPSLatitude' in gps_info and 'GPSLongitude' in gps_info:
            lat = get_decimal_from_dms(gps_info['GPSLatitude'], gps_info['GPSLatitudeRef'])
            lon = get_decimal_from_dms(gps_info['GPSLongitude'], gps_info['GPSLongitudeRef'])
            return {"latitude": lat, "longitude": lon}
    except Exception as e:
        return {"error": str(e)}

    return None

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371  # Earth radius in km
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)

    a = math.sin(d_phi/2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c

@app.post("/upload-images")
async def upload_images(
    images: List[UploadFile] = File(...),
    metadata: Optional[str] = Form(None)
):
    gps_results = []
    metadata_dict = {}

    if metadata:
        try:
            metadata_dict = json.loads(metadata)
        except Exception:
            metadata_dict = {"error": "Invalid metadata format"}

    prev_coords = None

    for image in images:
        temp_path = f"temp_{image.filename}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(image.file, buffer)

        gps_data = extract_gps_data(temp_path)
        result = {
            "filename": image.filename,
            "gps": gps_data
        }

        if prev_coords and gps_data and isinstance(gps_data, dict):
            lat1, lon1 = prev_coords
            lat2, lon2 = gps_data.get("latitude"), gps_data.get("longitude")
            if lat2 is not None and lon2 is not None:
                distance_km = haversine_distance(lat1, lon1, lat2, lon2)
                result["distance_from_previous_km"] = round(distance_km, 2)
                if distance_km > 1:
                    result["flag"] = "suspicious movement"

        if gps_data and isinstance(gps_data, dict) and "latitude" in gps_data:
            prev_coords = (gps_data["latitude"], gps_data["longitude"])

        gps_results.append(result)
        os.remove(temp_path)

    return {
        "status": "ok",
        "gps_info": gps_results,
        "metadata": metadata_dict
    }