#!/bin/bash

echo "Starting initialization..."
python init_db.py
echo "DB initialized. Starting server..."
uvicorn main:app --host 0.0.0.0 --port 8000