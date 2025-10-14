#!/bin/bash
set -e  # Exit if any command fails

echo "📦 Activating virtual environment and installing Python dependencies..."
if [ -d "antenv" ]; then
  source antenv/bin/activate
else
  echo "⚠️ Virtual environment not found, creating one..."
  python3 -m venv antenv || python -m venv antenv
  source antenv/bin/activate
fi

echo "⬆️ Upgrading pip and verifying dependencies..."
pip install --upgrade pip
pip install -r requirements.txt
pip install gunicorn uvicorn fastapi  # ensure web server packages are present

echo "☁️ Downloading FAISS indexes from Azure Blob..."
python download.py || echo "⚠️ Download script failed or skipped"

echo "🚀 Launching FastAPI app with Gunicorn..."
exec gunicorn api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:8000 \
  --timeout 600
