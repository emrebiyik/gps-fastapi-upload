# 📌 Credit Scoring API - Usage Guide

## 🔑 Authentication
Obtain a JWT token using your username and password:

```bash
curl -X POST https://<render-app>/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=<USER_ID>&password=<PASSWORD>"

Response:

{
  "access_token": "eyJhbGciOiJIUzI1NiIsInR...",
  "token_type": "bearer"
}

Use this token in all subsequent requests:

-H "Authorization: Bearer <TOKEN>"


⸻

🛰️ Endpoints

1️⃣ Health Check

Verify that the service is running:

curl https://<render-app>/health

Response:

{ "status": "ok" }


⸻

2️⃣ Upload GPS Images

Extract GPS coordinates from EXIF and compute distance relative to the first image.

curl -X POST "https://<render-app>/users/<USER_ID>/images" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "files=@photo1.jpg" \
  -F "files=@photo2.jpg"

Response:

{
  "processed": 2,
  "first_lat": -1.286389,
  "first_lon": 36.817223,
  "abnormalities": 1
}

Rules:
	•	Distance is relative to the first image
	•	If distance > 1 km → flag="abnormal"
	•	If GPS data is missing → flag="no_gps"

⚠️ Note: GPS data is only stored for verification and anomaly detection.
👉 It is not included in the credit scoring calculation.

⸻

3️⃣ Score CallLogs CSV

Upload a call log CSV to compute a rule-based credit score.

curl -X POST "https://<render-app>/users/<USER_ID>/score/calllogs" \
  -H "Authorization: Bearer <TOKEN>" \
  -F "loan_id=L001" \
  -F "calllogs_csv=@CallLogs-1753699040817.csv"

Response:

{
  "user_id": "emre",
  "loan_id": "L001",
  "score": 52.7,
  "decision": "REVIEW",
  "details": {
    "calllogs": {
      "metrics": { "...": "..." },
      "awarded": { "...": "..." }
    }
  }
}


⸻

⚙️ Environment Variables

You can override defaults in .env or via Render environment settings:

APPROVE_MIN=60
REVIEW_MIN=40
W_CALL_FREQ_HIGH=20
W_CALL_FREQ_MID=10
W_CALL_DUR_HIGH=15
W_CALL_DUR_MID=8
W_STABLE_CONTACT_MAX=30
W_WEEKDAY_CONSISTENT=10
W_WEEKDAY_SOME=5
W_REGULAR_CONSISTENT=10
W_REGULAR_SOME=5
W_NIGHT_LOW=10
W_MISSED_LOW=10

	•	APPROVE_MIN → minimum score for automatic approval
	•	REVIEW_MIN → minimum score for manual review (below this = reject)
	•	All W_* variables are point weights for call log metrics

⸻

📦 Deployment on Render
	1.	Ensure requirements.txt is present with all dependencies
	2.	Add a start.sh file:

#!/bin/bash
uvicorn main:app --host 0.0.0.0 --port 10000

	3.	Set environment variables in Render dashboard
	4.	Deploy 🎉

⸻

📚 Summary
	•	GPS endpoint → only verifies and stores anomalies (no scoring)
	•	CallLogs endpoint → performs actual rule-based credit scoring
	•	Authentication is required for all endpoints (Authorization: Bearer <TOKEN>)

---
