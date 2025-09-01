# Credit Scoring Engine

## Purpose
Implements the hybrid scoring model for healthcare financing decisions.  
Combines **rule-based logic** with **ML-based predictions**.

## Inputs
- Bank Data (39 metrics)
- Mobile Money Transactions
- Call Logs
- Demographic Data
- Previous Loan History

## Outputs
- Credit score (0-1000)
- Decision: APPROVED / REJECTED / REVIEW

## Architecture
- Rule Engine (Python)
- ML Model (Logistic Regression / XGBoost)
- Integrated via FastAPI

## Endpoints
- `POST /score` → Compute credit score
- `GET /score/{user_id}` → Retrieve score