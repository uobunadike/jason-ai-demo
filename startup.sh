#!/bin/bash
set -euo pipefail

APP_HOME=/home/site/wwwroot
VENV=$APP_HOME/antenv
PY=$VENV/bin/python
PIP=$VENV/bin/pip
GUNICORN=$VENV/bin/gunicorn

echo "🧹 Removing old virtual environment..."
rm -rf $VENV

echo "🐍 Creating new virtual environment..."
python3 -m venv $VENV

echo "⬆️ Upgrading pip..."
$PY -m pip install --upgrade pip

echo "📦 Installing dependencies..."
$PIP install -r $APP_HOME/requirements.txt

echo "🚀 Starting Gunicorn with UvicornWorker..."
exec $GUNICORN --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:${PORT:-8000} api:app

