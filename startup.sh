#!/bin/bash
set -euo pipefail

# Always run from the app root so "import api" works
cd /home/site/wwwroot

# Ensure the app folder is on PYTHONPATH (Oryx no longer adds it)
export PYTHONPATH="/home/site/wwwroot:${PYTHONPATH:-}"

# Azure will honor WEBSITES_PORT=8000; still default PORT for Gunicorn
export PORT="${PORT:-8000}"

echo "🚀 Starting FastAPI app via Gunicorn from $(pwd)"
echo "PYTHONPATH=$PYTHONPATH"

# More verbose logs so we SEE import/traceback if the worker crashes
exec gunicorn api:app \
  --workers "${WEB_CONCURRENCY:-1}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT}" \
  --timeout 600 \
  --access-logfile - \
  --error-logfile -
