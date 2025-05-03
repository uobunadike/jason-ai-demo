# --- Core Imports ---
import os
import sys
import re
from dotenv import load_dotenv
from fastapi import FastAPI, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()

# Ensure correct path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

# Import run from the production-ready model file
from model import run

# Initialize FastAPI app
app = FastAPI(
    title="RAG Inventory Assistant API",
    description="Use POST /query for full AI answers. Use POST /compact-query for KPI-only extractions.",
    version="1.0.0"
)

# Enable CORS (allow frontend to connect)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Open CORS for now
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Helpers ---
def extract_compact_kpi(text: str):
    """Extract the last number from the AI response"""
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1] if matches else text

def clean_response_text(text: str) -> str:
    """Clean up response text"""
    return text.replace("\\n", " ").replace("\n", " ").replace("  ", " ").strip()

# --- Request Schema ---
class Query(BaseModel):
    prompt: str
    model_type: str = "azure"
    model_name: str = "gpt-4"

# --- Endpoints ---
@app.post("/query")
def query_endpoint(query: Query = Body(...)):
    """Return full AI answer based on the input prompt"""
    try:
        result = run(query.prompt, query.model_type, query.model_name)
        cleaned_result = clean_response_text(result)
        return {"response": cleaned_result}
    except Exception as e:
        return {"error": str(e)}

@app.post("/compact-query")
def compact_query_endpoint(query: Query = Body(...)):
    """Return just the compact KPI number"""
    try:
        raw_result = run(query.prompt, query.model_type, query.model_name)
        compact_result = extract_compact_kpi(raw_result)
        cleaned_compact = clean_response_text(compact_result)
        return {"result": cleaned_compact}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    """Healthcheck endpoint"""
    return {"message": "✅ RAG Inventory Assistant API is running. Use POST /query."}

# --- Local Server Entrypoint ---
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
