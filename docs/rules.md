# Rule-Based Scoring v1 (Deterministic)

## Score Bands (Decision)
- **score >= 20** → `approve_500`
- **15 <= score <= 19** → `approve_400`
- **10 <= score <= 14** → `approve_150`
- **score < 10** → `deny`
- *(Optional)* Manual review conditions → `review`

---

## Bank Rules
- **average_balance >= 1500 USD** → +3  
- **net_cash_flow_90d > 0** → +2  
- **bounced_txn_90d >= 1** → -2  
- **overdraft_days_90d > 5** → -3  
- **income_avg_3m >= 800 USD** → +3  
- *(… add custom rules)*  

---

## Mobile Money Rules
- **mm_txn_90d >= 30** → +2  
- **mm_volume_90d >= 500 USD** → +2  
- **mm_active_days_90d >= 20** → +1  
- **avg_ticket_90d < 2 USD** → -1  
- *(… add custom rules)*  

---

## Call Logs Rules (if available)
- **unique_contacts_30d >= 15** → +1  
- **incoming_outgoing_ratio_30d very low** → -1  
- *(… add custom rules)*  

---

## GPS / Assets Rules
- **image_verified == true** AND **estimated_value >= 1000** → +2  
- **distance_km shows frequent "abnormal" pattern** → -2  
- *(… add custom rules)*  

---

## Conflict & Priority
- Negative risk rules take precedence over positive ones.  
- For overlapping rules affecting the same field, apply the **highest or one-time effect** only.  