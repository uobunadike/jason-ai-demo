#!/bin/bash
set -e  # Exit immediately if any command fails

echo "📦 Installing Python dependencies..."
pip install -r requirements.txt

echo "☁️ Downloading FAISS indexes from Azure Blob..."
python download.py

echo "🚀 Launching FastAPI app with Gunicorn..."
exec gunicorn api:app \
 --workers 1 \
 --worker-class uvicorn.workers.UvicornWorker \
 --bind 0.0.0.0:8000 \
 --timeout 600