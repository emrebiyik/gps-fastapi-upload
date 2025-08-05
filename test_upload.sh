curl -X "POST" http://localhost:8000/upload-images \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "images=@./image1.jpg" \
  -F "images=@./image_germany.jpg" \
  -F "images=@./image_italy.jpg" \
  -F "metadata={\"user_id\":123}"