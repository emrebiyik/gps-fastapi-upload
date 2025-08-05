#!/bin/bash

# Initialize the database before starting the server
python init_db.py

# Start the FastAPI server
uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000}