#!/bin/bash
set -e

echo "Starting initialization..."
python init_db.py

echo "Starting server..."
uvicorn main:app --host 0.0.0.0 --port $PORT