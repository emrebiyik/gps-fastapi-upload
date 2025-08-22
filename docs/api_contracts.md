# API Contracts

## GPS Ingest
POST /api/v1/gps/ingest
Form-Data: images[] (binary), metadata (json: {user_id})
Response 200:
{
  "status": "ok",
  "items": [
    {
      "user_id": "u_123",
      "filename": "img_001.jpg",
      "latitude": 50.1109,
      "longitude": 8.6821,
      "distance_km": 1.2,
      "flag": "normal",
      "taken_at": "2025-08-20T14:21:00Z"
    }
  ]
}

## Assets Ingest
POST /api/v1/assets/ingest
Body (json):
{
  "assets": [
    {
      "user_id": "u_123",
      "asset_type": "motorbike",
      "estimated_value": 1200.0,
      "image_verified": true,
      "source_image": "img_001.jpg",
      "currency_code": "USD"
    }
  ]
}
Response:
{"status":"ok","saved":1}

## Bank Ingest
POST /api/v1/bank/ingest
Body (json):
{
  "user_id": "u_123",
  "income_avg_3m": 900.0,
  "average_balance": 1600.0,
  "net_cash_flow_90d": 250.0,
  "bounced_txn_90d": 0,
  "overdraft_days_90d": 2,
  "statement_period_days": 90,
  "currency_code": "USD"
}
Response: {"status":"ok","saved":1}

## Mobile Ingest
POST /api/v1/mobile/ingest
Body (json):
{
  "user_id":"u_123",
  "mm_txn_90d": 42,
  "mm_volume_90d": 780.0,
  "mm_active_days_90d": 26,
  "avg_ticket_90d": 18.5,
  "last_txn_at": "2025-08-18T09:11:00Z",
  "currency_code":"USD"
}
Response: {"status":"ok","saved":1}

## CallLogs Ingest (varsa)
POST /api/v1/calllogs/ingest
Body (json):
{
  "user_id":"u_123",
  "unique_contacts_30d": 22,
  "call_days_30d": 15,
  "incoming_outgoing_ratio_30d": 0.78,
  "airtime_spend_30d": 6.3
}
Response: {"status":"ok","saved":1}

## Feature Snapshot
GET /api/v1/users/{user_id}/features
Response:
{
  "user_id":"u_123",
  "gps": {... son kayıt ...},
  "assets": [... son kayıtlar ...],
  "bank": {... son ...},
  "mobile": {... son ...},
  "calllogs": {... son ...}
}

## Score Compute
POST /api/v1/score/compute
Body: {"user_id":"u_123"}
Response:
{
  "user_id":"u_123",
  "score": 21,
  "decision":"approve_500",
  "explanation": {"bank":"+5","mobile":"+4","assets":"+2","gps":"-0"},
  "created_at":"2025-08-21T19:00:00Z"
}