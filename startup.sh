#!/bin/bash
set -e  # Exit if any command fails

echo "[INFO] Setting up environment..."

# 1. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn uvicorn fastapi

# 2. Download FAISS + Excel data from Azure Blob
echo "[INFO] Downloading FAISS + data from Azure Blob..."
python download.py

# 3. Start FastAPI app via Gunicorn (Uvicorn workers)
echo "[INFO] Launching FastAPI app with Gunicorn..."
exec gunicorn api:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --timeout 600
