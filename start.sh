#!/bin/bash
set -e

# go to app folder
cd v2

echo "Starting initialization..."
python init_db.py

echo "Starting server..."
uvicorn main:app --host 0.0.0.0 --port $PORT