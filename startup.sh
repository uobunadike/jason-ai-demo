#!/bin/bash
set -e  # Exit if any command fails

echo "📦 Setting up environment..."

# 1️⃣ Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn uvicorn fastapi

# 2️⃣ Download FAISS + Excel data from Azure Blob
echo "☁️ Downloading FAISS + data from Azure Blob..."
python download.py

# 3️⃣ Start FastAPI app via Gunicorn (Uvicorn workers)
echo "🚀 Launching FastAPI app with Gunicorn..."
exec gunicorn api:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --timeout 600
