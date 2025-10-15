#!/bin/bash
# Remove old virtual environment to ensure a clean install
rm -rf /home/site/wwwroot/antenv

# Create a new virtual environment in the persistent directory
python3 -m venv /home/site/wwwroot/antenv

# Activate the new virtual environment
source /home/site/wwwroot/antenv/bin/activate

# Upgrade pip within the new virtual environment
pip install --upgrade pip

# Install dependencies, ignoring system packages
pip install --ignore-installed -r requirements.txt

# Run the application
gunicorn --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind=0.0.0.0:8000 api:app