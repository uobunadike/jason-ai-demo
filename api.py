# --- Core Imports -----
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

app = FastAPI(
    title="RAG Inventory Assistant API",
    description="Use POST /query for full AI answers. Use POST /compact-query for KPI-only extractions.",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def extract_compact_kpi(text: str):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1] if matches else text

def clean_response_text(text: str) -> str:
    return text.replace("\\n", " ").replace("\n", " ").replace("  ", " ").strip()

class Query(BaseModel):
    prompt: str
    model_type: str = "azure"
    model_name: str = "gpt-4"

class QueryResponse(BaseModel):
    result: str

@app.post("/query", response_model=QueryResponse)
def query_endpoint(query: Query = Body(...)):
    logging.info(f"Received query: {query}")
    try:
        result = run(query.prompt, query.model_type, query.model_name)
        cleaned_result = clean_response_text(result)
        return QueryResponse(result=cleaned_result)
    except Exception as e:
        logging.error(f"Error processing the query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/compact-query", response_model=QueryResponse)
def compact_query_endpoint(query: Query = Body(...)):
    logging.info(f"Received compact query: {query}")
    try:
        raw_result = run(query.prompt, query.model_type, query.model_name)
        compact_result = extract_compact_kpi(raw_result)
        cleaned_compact = clean_response_text(compact_result)
        return QueryResponse(result=cleaned_compact)
    except Exception as e:
        logging.error(f"Error processing the compact query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
def home():
    return {"message": "✅ RAG Inventory Assistant API is running. Use POST /query."}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)

