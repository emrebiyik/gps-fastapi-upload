# Credit Scoring Engine

## Overview
The engine computes credit scores based solely on GPS-derived features (Phase 3 scope).

## Rules
- If latest GPS record has:
  - `flag = "abnormal"` → -5 points
  - `flag = "normal"` → +10 points
  - `flag = "no_gps"` → +0 points

## Decision Thresholds
- Score ≥ 10 → **approve**
- Score ≥ 0 → **review**
- Score < 0 → **deny**

## Endpoint
`POST /api/v1/score/compute`

### Request
```json
{ "user_id": "USER123", "loan_id": "LOAN-2025-0001" }