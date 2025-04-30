#!/bin/bash
uvicorn to_api:app --host 0.0.0.0 --port 8000

echo "🔧 Installing system dependencies..."
apt-get update
apt-get install -y build-essential libomp-dev

echo "🚀 Starting your FastAPI app..."
gunicorn -w 1 -k uvicorn.workers.UvicornWorker to_api:app --bind=0.0.0.0:8000
pip install -r requirements.txt && gunicorn --bind=0.0.0.0 --timeout 600 to_api:app
