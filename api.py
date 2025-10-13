# --- Core Imports ------- d
import os
import sys
import re
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, Body, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Load environment variables
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import run

# --- FastAPI Setup ---
app = FastAPI(
    title="Multi-Persona RAG Assistant API",
    description="Unified API for Jason (Inventory) and Claire (Onboarding). Use POST /query to interact.",
    version="2.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- Utility Functions ---
def extract_compact_kpi(text: str):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1] if matches else text

def clean_response_text(text: str) -> str:
    return text.replace("\\n", " ").replace("\n", " ").replace("  ", " ").strip()

# --- Request/Response Models ---
class Query(BaseModel):
    prompt: str
    model_type: str = "azure"
    model_name: str = "gpt-4"
    persona: str = "jason"  # 👈 NEW FIELD (defaults to Jason)

class QueryResponse(BaseModel):
    result: str

# --- Endpoints ---
@app.post("/query", response_model=QueryResponse)
def query_endpoint(query: Query = Body(...)):
    """Main endpoint for Claire and Jason queries."""
    logging.info(f"Received query for {query.persona}: {query.prompt}")
    try:
        result = run(
            query.prompt,
            model_type=query.model_type,
            model_name=query.model_name,
            persona=query.persona
        )
        cleaned_result = clean_response_text(result)
        return QueryResponse(result=cleaned_result)
    except Exception as e:
        logging.error(f"Error processing the query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compact-query", response_model=QueryResponse)
def compact_query_endpoint(query: Query = Body(...)):
    """Compact endpoint that extracts numeric KPI or summary values."""
    logging.info(f"Received compact query for {query.persona}: {query.prompt}")
    try:
        raw_result = run(
            query.prompt,
            model_type=query.model_type,
            model_name=query.model_name,
            persona=query.persona
        )
        compact_result = extract_compact_kpi(raw_result)
        cleaned_compact = clean_response_text(compact_result)
        return QueryResponse(result=cleaned_compact)
    except Exception as e:
        logging.error(f"Error processing compact query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": "✅ Multi-Persona RAG Assistant API is running.",
        "personas": ["jason (inventory)", "claire (onboarding)"],
        "usage": "POST /query with { prompt, persona }"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

