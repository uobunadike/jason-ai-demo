#!/bin/bash
set -e  # Exit immediately if any command fails

VENV_PATH="/home/site/wwwroot/antenv"

echo "📦 Ensuring virtual environment is used..."
if [ -d "$VENV_PATH/bin" ]; then
    PYTHON="$VENV_PATH/bin/python"
    PIP="$VENV_PATH/bin/pip"
else
    PYTHON="$VENV_PATH/Scripts/python"
    PIP="$VENV_PATH/Scripts/pip"
fi

echo "📦 Installing Python dependencies into $VENV_PATH..."
$PYTHON -m pip install --upgrade pip
$PIP install --no-cache-dir -r requirements.txt

echo "☁️ Downloading FAISS indexes from Azure Blob..."
$PYTHON download.py

echo "🚀 Launching FastAPI app with Gunicorn..."
exec $VENV_PATH/bin/gunicorn api:app \
 --workers 1 \
 --worker-class uvicorn.workers.UvicornWorker \
 --bind 0.0.0.0:8000 \
 --timeout 600