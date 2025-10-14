#!/bin/bash
set -e

echo "🚀 Activating virtual environment and starting FastAPI app..."

# Activate the prebuilt virtual environment created by Oryx
if [ -d "antenv" ]; then
  source antenv/bin/activate
else
  echo "⚠️ No antenv found; relying on Oryx virtual environment."
fi

# Start the FastAPI app via Gunicorn and UvicornWorker
exec gunicorn api:app \
  --workers 1 \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind 0.0.0.0:${PORT:-8000} \
  --timeout 600
