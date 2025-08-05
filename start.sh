#!/bin/bash

# Run init_db.py to create the database tables if they don't exist
if [ ! -f "gps_data.db" ]; then
  echo "Creating GPS database..."
  python init_db.py
fi

# Start the FastAPI app with Uvicorn
exec uvicorn main:app --host=0.0.0.0 --port=10000