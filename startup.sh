#!/bin/bash
echo "📦 Installing dependencies..."
pip install -r requirements.txt

echo "⬇️ Downloading FAISS + data from Blob..."
python download.py

echo "🚀 Launching FastAPI app with Gunicorn..."
gunicorn api:app --bind=0.0.0.0:8000 --timeout 900
