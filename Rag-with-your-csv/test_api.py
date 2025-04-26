import os
import sys
import re
import asyncio
from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from uvicorn import Config, Server

# Load environment variables
load_dotenv()

# Import your RAG pipeline function
from model import run

# FastAPI app
app = FastAPI(
    title="RAG Inventory Assistant API",
    description="Use /query for full AI answers. Use /compact-query for KPI-only numbers.",
    version="1.0.0"
)

# Enable CORS (adjust for production if needed)
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)

# Helper to extract numeric KPIs from the text
def extract_compact_kpi(text: str):
    matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return matches[-1] if matches else text

# Helper to clean extra line breaks
def clean_response_text(text: str) -> str:
    return text.replace("\\n", " ").replace("\n", " ").replace("  ", " ").strip()

# Define input schema
class Query(BaseModel):
    prompt: str
    model_name: str = "gpt-35-turbo"  # Now default model is gpt-35-turbo

# Route to get full AI response
@app.post("/query")
def query_endpoint(query: Query):
    try:
        result = run(query.prompt, query.model_name)  # Only prompt and model_name
        cleaned_result = clean_response_text(result)
        return {"response": cleaned_result}
    except Exception as e:
        return {"error": str(e)}

# Route to get compact KPI (number only)
@app.post("/compact-query")
def compact_query_endpoint(query: Query):
    try:
        raw_result = run(query.prompt, query.model_name)
        compact_result = extract_compact_kpi(raw_result)
        cleaned_compact = clean_response_text(compact_result)
        return {"result": cleaned_compact}
    except Exception as e:
        return {"error": str(e)}

# Healthcheck
@app.get("/")
def home():
    return {"message": "RAG API is running. Use POST /query."}

# Local server launcher
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("m_api:app", host="0.0.0.0", port=8000, reload=True)
