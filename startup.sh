#!/bin/bash
set -e

echo "🚀 Starting FastAPI app..."

# Activate existing virtual environment (Oryx creates it during build).
if [ -d "antenv" ]; then
  echo "Activating virtual environment..."
  source antenv/bin/activate
else
  echo "⚠️ Virtual environment not found — app may not start correctly."
fi

# Launch the app using Gunicorn and Uvicorn workers
exec gunicorn -w 4 -k uvicorn.workers.UvicornWorker api:app --bind=0.0.0.0:${PORT:-8000}
