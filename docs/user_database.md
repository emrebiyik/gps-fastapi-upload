# User Database

## Overview
The user database stores registered users and links them to their GPS data and credit score records.  

## Tables

### `users`
- `id` (PK, integer, auto-increment)
- `user_id` (string, unique, indexed, **used in API & JWT sub**)
- `created_at` (timestamp)

### `gps_data`
- `id` (PK)
- `user_id` (FK → users.id)
- `filename` (string)
- `latitude` (float, nullable)
- `longitude` (float, nullable)
- `distance_km` (float, persisted full precision)
- `flag` (string: `"normal"`, `"abnormal"`, `"no_gps"`)
- `created_at` (timestamp)

### `credit_scores`
- `id` (PK)
- `user_id` (FK → users.id)
- `loan_id` (string, NOT NULL)
- `score` (integer)
- `decision` (string: `"approve" | "review" | "deny"`)
- `explanation_json` (JSON)
- `created_at` (timestamp)

**Constraints**
- `(user_id, loan_id)` unique in `credit_scores` → ensures idempotent writes.