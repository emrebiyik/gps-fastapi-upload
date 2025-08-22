# Data Mapping

| Source         | Incoming Field          | Internal Field        | Type   | Notes                       |
|----------------|-------------------------|-----------------------|--------|-----------------------------|
| gps_service    | lat                     | latitude              | float  | degrees                     |
| gps_service    | lon                     | longitude             | float  | degrees                     |
| asset_service  | value_est               | estimated_value       | float  | USD                         |
| bank_service   | avg_bal                 | average_balance       | float  | USD                         |
| mobile_service | txn_count_90d           | mm_txn_90d            | int    |                             |
| mobile_service | volume_90d_USD          | mm_volume_90d         | float  | USD                         |
| calllogs       | uniq_cnt_30d            | unique_contacts_30d   | int    |                             |