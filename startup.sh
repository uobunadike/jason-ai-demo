#!/bin/bash
set -e  # Exit immediately if any command fails

echo "📦 Activating virtual environment..."
if [ -d "antenv/bin" ]; then
    source antenv/bin/activate
elif [ -d "antenv/Scripts" ]; then
    source antenv/Scripts/activate
fi

echo "📦 Installing Python dependencies..."
python -m pip install --upgrade pip
pip install --no-cache-dir -r requirements.txt

echo "☁️ Downloading FAISS indexes from Azure Blob..."
python download.py

echo "🚀 Launching FastAPI app with Gunicorn..."
exec gunicorn api:app \
 --workers 1 \
 --worker-class uvicorn.workers.UvicornWorker \
 --bind 0.0.0.0:8000 \
 --timeout 600
