# Dashboard (Admin)

## Overview
The admin dashboard provides visibility into:
- User profiles
- Uploaded GPS data
- Credit score decisions

## Features
- **User List** → shows `user_id`, created date.
- **GPS Records** → displays filename, coordinates, distance (rounded 2 decimals), flag.
- **Flags**:
  - `"normal"` → distance ≤ 1 km
  - `"abnormal"` → distance > 1 km
  - `"no_gps"` → EXIF missing
- **Credit Scores** → latest decision and score per `(user_id, loan_id)`.

## Actions
- Filter by `user_id` or `loan_id`
- Review abnormal GPS jumps
- Export results for auditing