Update: GPS Ingest + Credit Score API (aligned with user_id & loan_id, JWT enabled)

Summary
	•	I simplified the service to two endpoints: GPS ingest and Score compute.
	•	Authentication is now JWT (OAuth2 password flow) and enforced on both endpoints.
	•	Request IDs are standardized to user_id (user) and loan_id (application), matching the payslip service.
	•	GPS ingest supports multiple images; distance is calculated relative to the first image.
	•	Flags: "normal" (<= 1 km), "abnormal" (> 1 km), "no_gps" (no EXIF).
	•	DB schema updated to support user_id and loan_id uniqueness.

⸻

Endpoints

1) Auth – get token

POST /auth/token (x-www-form-urlencoded)
	•	username = USER123 (demo)
	•	password = secret123 (demo)
Returns { access_token, token_type }.

2) GPS ingest (multi-file, protected)

POST /api/v1/gps/ingest?user_id=<USER_ID>
Body: multipart/form-data → files (one or more images)

Rules
	•	First image must have EXIF GPS; otherwise 400 No GPS data in the FIRST image.
	•	Every other image’s distance is computed vs. the first image.
	•	Threshold = 1 km → flag="abnormal" if distance > 1 km, else "normal". If image has no GPS → "no_gps".
	•	Response returns distance_km with 2 decimal places (DB stores full precision).

Sample response

{
  "status": "ok",
  "user_id": "USER123",
  "reference": { "filename": "image_germany.jpg", "latitude": 47.4669, "longitude": 10.20375 },
  "items": [
    { "index": 0, "gps_id": 23, "filename": "image_germany.jpg", "latitude": 47.4669, "longitude": 10.20375, "distance_km": 0.00, "flag": "normal" },
    { "index": 1, "gps_id": 24, "filename": "image_italy.jpg",    "latitude": 45.8776, "longitude": 10.85716, "distance_km": 183.61, "flag": "abnormal" | "normal" },
    { "index": 2, "gps_id": 25, "filename": "imageX.jpg",          "latitude": null,    "longitude": null,     "distance_km": null,   "flag": "no_gps" }
  ]
}

3) Score compute (protected)

POST /api/v1/score/compute
Body (JSON):

{ "user_id": "USER123", "loan_id": "LOAN-2025-0001" }

	•	Uses the latest GPS record only (Phase 3 scope).
	•	Simple rule-based score: "abnormal" → -5, else +10.
	•	Upsert behavior on (user_id, loan_id) → repeat calls update the same application.

Sample response

{ "user_id": "USER123", "loan_id": "LOAN-2025-0001", "score": 10, "decision": "approve" }


⸻

How to test in Swagger
	1.	Open /docs.
	2.	Call POST /auth/token with username/password → copy access_token.
	3.	Click Authorize (top-right) → paste Bearer <access_token>.
	4.	Call POST /api/v1/gps/ingest?user_id=USER123 with 1+ images in files.
	5.	Call **POST /api/v1/score/computewith body{ “user_id”:“USER123”,“loan_id”:“LOAN-2025-0001” }`.

⸻

cURL (ready to run)

# 1) Get token
TOKEN=$(curl -s -X POST "<BASE_URL>/auth/token" \
  -H "content-type: application/x-www-form-urlencoded" \
  -d "grant_type=password&username=USER123&password=secret123" \
  | python -c 'import sys,json; print(json.load(sys.stdin)["access_token"])')

# 2) GPS ingest (multi-file)
curl -X POST "<BASE_URL>/api/v1/gps/ingest?user_id=USER123" \
  -H "Authorization: Bearer $TOKEN" \
  -F "files=@/path/image_germany.jpg;type=image/jpeg" \
  -F "files=@/path/image_italy.jpg;type=image/jpeg"

# 3) Score compute
curl -X POST "<BASE_URL>/api/v1/score/compute" \
  -H "Authorization: Bearer $TOKEN" \
  -H "content-type: application/json" \
  -d '{"user_id":"USER123","loan_id":"LOAN-2025-0001"}'


⸻

Database & Migrations
	•	users: added/standardized user_id (unique, indexed).
	•	credit_scores: added loan_id (NOT NULL) and unique (user_id, loan_id) + composite index.
	•	Behavior: DB stores full-precision distance_km; API response rounds to 2 decimals.

⸻

Breaking Changes
	•	Request field names standardized: user_id and loan_id (previously external_user_id was used).
	•	GPS ingest now expects files (multi-part list) instead of single file.
	•	JWT is required for both endpoints; token sub must equal user_id.

⸻

Next Actions (for the other team)
	•	Update frontend/service calls to:
	•	Send Bearer token on every request.
	•	Use user_id and loan_id consistently.
	•	Use files array for GPS ingest (first image must contain EXIF GPS).
	•	If needed, adjust the 1 km threshold (we can parameterize it).
