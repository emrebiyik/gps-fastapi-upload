# Schema (Field Dictionary)

## Global Rules
- **Naming:** `snake_case`  
- **Datetime:** ISO-8601, UTC (e.g., `2025-08-21T18:45:00Z`)  
- **Money:** USD (float), use integer `USD_cents` if needed  
- **Boolean:** `true/false`  
- **Missing data:** `null`, also add `source_missing: true/false` flag  
- **Versioning:** `schema_version=1.0`  

---

## Identity Fields
- **user_id** (string) – internal system identifier  
- **external_user_id** (string) – external microservice source identifier  
- **source** (enum): `["gps","assets","bank","mobile","calllogs"]`  

---

## GPS
- **latitude** (float, deg)  
- **longitude** (float, deg)  
- **distance_km** (float, km, relative to reference point)  
- **flag** (enum: `["normal","abnormal"]`)  
- **filename** (string)  
- **taken_at** (datetime, UTC)  

---

## ImageAssets
- **asset_type** (string; e.g., `"motorbike","fridge"`)  
- **estimated_value** (float, USD)  
- **image_verified** (bool)  
- **source_image** (string)  
- **created_at** (datetime, UTC)  

---

## BankMetrics
- **income_avg_3m** (float, USD)  
- **average_balance** (float, USD)  
- **net_cash_flow** (float, USD, specify 30/90d)  
- **bounced_txn_90d** (int)  
- **overdraft_days_90d** (int)  
- **statement_period_days** (int)  
- **created_at** (datetime, UTC)  

---

## MobileMoneyMetrics
- **mm_txn_90d** (int)  
- **mm_volume_90d** (float, USD)  
- **mm_active_days_90d** (int)  
- **avg_ticket_90d** (float, USD)  
- **last_txn_at** (datetime, UTC)  
- **created_at** (datetime, UTC)  

---

## CallLogMetrics (if available)
- **unique_contacts_30d** (int)  
- **call_days_30d** (int)  
- **incoming_outgoing_ratio_30d** (float)  
- **airtime_spend_30d** (float, USD)  
- **created_at** (datetime, UTC)  

---

## CreditScores
- **score** (int)  
- **decision** (enum: `["approve_500","approve_400","approve_150","deny","review"]`)  
- **explanation_json** (json)  
- **created_at** (datetime, UTC)  