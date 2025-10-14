#!/bin/bash
set -euo pipefail

cd /home/site/wwwroot
source antenv/bin/activate

export PYTHONPATH="/home/site/wwwroot:${PYTHONPATH:-}"
export PORT="${PORT:-8000}"

echo "🚀 Running in virtualenv: $(which python)"
pip list | grep uvicorn || echo "⚠️ Uvicorn not found in active environment!"

exec gunicorn api:app \
  --workers "${WEB_CONCURRENCY:-1}" \
  --worker-class uvicorn.workers.UvicornWorker \
  --bind "0.0.0.0:${PORT}" \
  --timeout 600 \
  --access-logfile - \
  --error-logfile -
