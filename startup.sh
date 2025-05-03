#!/bin/bash
echo "📦 Installing Python dependencies..."
pip install -r requirements.txt
echo "☁️ Downloading FAISS + data from Azure Blob..."
python download.py
echo "🚀 Launching FastAPI app with Gunicorn..."
gunicorn api:app \
 --workers 1 \
 --worker-class uvicorn.workers.UvicornWorker \
 --bind 0.0.0.0:8000 \
 --timeout 600