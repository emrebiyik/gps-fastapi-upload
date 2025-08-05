#!/bin/bash

URL="https://gps-fastapi-upload.onrender.com/upload-images"
METADATA='{"user_id": 42, "location": "Test location"}'

curl -X "POST" "$URL" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "images=@./image1.jpg;type=image/jpeg" \
  -F "images=@./image_italy.jpeg;type=image/jpeg" \
  -F "images=@./image_germany.jpeg;type=image/jpeg" \
  -F "metadata=$METADATA"