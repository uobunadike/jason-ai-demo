#!/bin/bash
set -e

echo "🚀 Starting FastAPI app via Gunicorn..."
gunicorn api:app \
 --workers 1 \
 --worker-class uvicorn.workers.UvicornWorker \
 --bind 0.0.0.0:8000 \
 --timeout 600