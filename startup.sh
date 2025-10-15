#!/bin/bash
set -e  # Exit immediately if a command fails

echo "🔹 Setting up virtual environment..."
if [ ! -d "antenv" ]; then
  python3 -m venv antenv
fi
source antenv/bin/activate

echo "⬆️ Upgrading pip and installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

echo "☁️ Downloading FAISS index and data from Azure Blob..."
python download.py

echo "🚀 Starting FastAPI app with Gunicorn..."
exec gunicorn api:app \
  --workers 2 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --timeout 600
