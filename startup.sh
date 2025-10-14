#!/bin/bash
set -e  # exit on error

echo "📦 Activating virtual environment..."
if [ -d "antenv" ]; then
  source antenv/bin/activate
else
  echo "⚠️ Virtual environment missing, creating one in /home/site/wwwroot/antenv"
  python3 -m venv antenv
  source antenv/bin/activate
fi

echo "⬆️ Upgrading pip and installing dependencies inside antenv..."
# Use the venv’s pip explicitly
./antenv/bin/pip install --upgrade pip
./antenv/bin/pip install -r requirements.txt
./antenv/bin/pip install gunicorn uvicorn fastapi

echo "🚀 Launching FastAPI app with Gunicorn..."
exec ./antenv/bin/gunicorn api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --timeout 600
