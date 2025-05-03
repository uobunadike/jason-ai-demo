# --- Core Imports ---
import os
import sys
import re
import json
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Ensure correct path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import run from your model logic
from model import run

# Initialize FastAPI app
app = FastAPI(
    title="RAG Inventory Assistant API",
    description="Use POST /query for full AI answers. Use POST /compact-query for KPI-only extractions.",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---
def extract_compact_kpi(text: str):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1] if matches else text

def clean_response_text(text: str) -> str:
    return text.replace("\\n", " ").replace("\n", " ").replace("  ", " ").strip()

# --- Manual parsing to avoid __fields_set__ issue ---
@app.post("/query")
async def query_endpoint(request: Request):
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        model_type = body.get("model_type", "azure")
        model_name = body.get("model_name", "gpt-4")

        result = run(prompt, model_type, model_name)
        return {"response": clean_response_text(result)}
    except Exception as e:
        return {"error": str(e)}

@app.post("/compact-query")
async def compact_query_endpoint(request: Request):
    try:
        body = await request.json()
        prompt = body.get("prompt", "")
        model_type = body.get("model_type", "azure")
        model_name = body.get("model_name", "gpt-4")

        raw_result = run(prompt, model_type, model_name)
        compact_result = extract_compact_kpi(raw_result)
        return {"result": clean_response_text(compact_result)}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "✅ RAG Inventory Assistant API is running. Use POST /query."}

# Local dev entrypoint
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

