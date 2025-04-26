import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from model import run  # assuming this is your full RAG pipeline
# Load .env if needed
from dotenv import load_dotenv
load_dotenv()
import asyncio
#rom asyncio.windows_events import ProactorEventLoop

from fastapi import FastAPI
from uvicorn import Config, Server
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
import re

app = FastAPI()


# Initialize FastAPI app
#app = FastAPI()
# Enable CORS for frontend (adjust origin in production)
app.add_middleware(
   CORSMiddleware,
   allow_origins=["*"],  # Use specific domain in production
   allow_credentials=True,
   allow_methods=["*"],
   allow_headers=["*"],
)
# Adding a helper to extract numbers 
def extract_compact_kpi(text):
   matches = re.findall(r"[-+]?\d*\.\d+|\d+", text)
   return matches[-1] if matches else text

# Define input schema
class Query(BaseModel):
   prompt: str
   model_type: str = "ollama"  # "ollama" or "openai"
   model_name: str = "llama3.1"  # or "gpt-4-turbo"

def clean_response_text(text: str) -> str:
    return text.replace("\\n", " ").replace("\n", " ").replace("  ", " ").strip()

# Define route
@app.post("/query")
def query_endpoint(query: Query):
   try:
       result = run(query.prompt, query.model_type, query.model_name)
       cleaned_result = clean_response_text(result)
       return {"response": cleaned_result}
   except Exception as e:
       return {"error": str(e)}

@app.post("/compact-query")
def compact_query_endpoint(query: Query):
    try:
        raw_result = run(query.prompt, query.model_type, query.model_name)
        compact_result = extract_compact_kpi(raw_result)
        cleaned_compact = clean_response_text(compact_result)
        return {"result": cleaned_compact}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"message": "RAG API is running. Use POST /query."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000)
