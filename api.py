# --- Core Imports & Azure Fix ---
import os
import sys
import site

# --- Fix Azure import path issue (legacy /agents/python) ---
try:
    # 1) Remove Azure system Python path if present
    sys.path = [p for p in sys.path if not (isinstance(p, str) and p.startswith("/agents/python"))]

    # 2) Ensure our venv's site-packages take priority
    paths = []
    try:
        paths.extend(site.getsitepackages())
    except Exception:
        pass
    try:
        usp = site.getusersitepackages()
        if usp:
            paths.append(usp)
    except Exception:
        pass

    for p in reversed([p for p in paths if p and p in sys.path]):
        sys.path.remove(p)
    for p in [p for p in paths if p]:
        sys.path.insert(0, p)
except Exception:
    pass
# --- End Azure Fix ---

from fastapi import FastAPI, Body, HTTPException
import re
import logging
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Dict, Any


# Load environment variabless
load_dotenv()
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from model import run, run_legacy, JASON_INTENTS, CLAIRE_INTENTS

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
    persona: str = "jason"


class ActionItem(BaseModel):
    id: str
    label: str
    type: str  # "primary", "secondary", "danger"


class ResponseItem(BaseModel):
    text: str
    sku: Optional[str] = None
    quantity: Optional[int] = None
    vendor: Optional[str] = None
    cost: Optional[float] = None
    status: Optional[str] = None
    customer: Optional[str] = None
    date: Optional[str] = None


class StructuredQueryResponse(BaseModel):
    """
    Structured JSON response for frontend formatting.
    
    Types:
    - "text": Simple text response (title only)
    - "list": List response with items
    - "action": Response with actionable items
    - "follow_up": Agent needs more information from user
    """
    type: str  # "text", "list", "action", "follow_up"
    title: str
    message: Optional[str] = None
    items: List[Dict[str, Any]] = []
    actions: List[ActionItem] = []


class LegacyQueryResponse(BaseModel):
    """Legacy response format for backward compatibility."""
    result: str


class IntentInfo(BaseModel):
    """Information about an intent."""
    name: str
    description: str
    example_queries: List[str]


class PersonaIntentsResponse(BaseModel):
    """Response containing intents for a persona."""
    persona: str
    intents: List[IntentInfo]


# --- Endpoints ---
@app.post("/query", response_model=LegacyQueryResponse)
def query_endpoint(query: Query = Body(...)):
    """
    Main endpoint for Claire and Jason queries.
    Returns plain text response in {"result": "..."} format.
    This is the original format for backward compatibility.
    """
    logging.info(f"Received query for {query.persona}: {query.prompt}")
    try:
        result = run_legacy(
            query.prompt,
            model_type=query.model_type,
            model_name=query.model_name,
            persona=query.persona
        )
        cleaned_result = clean_response_text(result)
        return LegacyQueryResponse(result=cleaned_result)
    except Exception as e:
        logging.error(f"Error processing the query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/query-structured", response_model=StructuredQueryResponse)
def structured_query_endpoint(query: Query = Body(...)):
    """
    NEW: Structured JSON endpoint for enhanced frontend formatting.
    
    Response structure:
    - type: "text" | "list" | "action" | "follow_up"
    - title: Header/summary text
    - message: Additional context (for follow_up type)
    - items: List of structured items with relevant fields (sku, quantity, vendor, etc.)
    - actions: Available actions/buttons for the user (e.g., "Process Order")
    
    Use this endpoint when you want structured data for rich UI rendering.
    """
    logging.info(f"Received structured query for {query.persona}: {query.prompt}")
    try:
        result = run(
            query.prompt,
            model_type=query.model_type,
            model_name=query.model_name,
            persona=query.persona
        )
        
        # Convert actions to ActionItem models
        actions = [
            ActionItem(id=a["id"], label=a["label"], type=a["type"])
            for a in result.get("actions", [])
        ]
        
        return StructuredQueryResponse(
            type=result["type"],
            title=result["title"],
            message=result.get("message"),
            items=result.get("items", []),
            actions=actions
        )
    except Exception as e:
        logging.error(f"Error processing structured query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/compact-query", response_model=LegacyQueryResponse)
def compact_query_endpoint(query: Query = Body(...)):
    """Compact endpoint that extracts numeric KPI or summary values."""
    logging.info(f"Received compact query for {query.persona}: {query.prompt}")
    try:
        raw_result = run_legacy(
            query.prompt,
            model_type=query.model_type,
            model_name=query.model_name,
            persona=query.persona
        )
        compact_result = extract_compact_kpi(raw_result)
        cleaned_compact = clean_response_text(compact_result)
        return LegacyQueryResponse(result=cleaned_compact)
    except Exception as e:
        logging.error(f"Error processing compact query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/intents/{persona}", response_model=PersonaIntentsResponse)
def get_persona_intents(persona: str):
    """
    Get the list of supported intents/questions for a persona.
    Use this to display suggested questions in the UI.
    """
    persona_lower = persona.lower()
    
    if persona_lower == "jason":
        intents = JASON_INTENTS
    elif persona_lower == "claire":
        intents = CLAIRE_INTENTS
    else:
        raise HTTPException(status_code=400, detail=f"Unknown persona: {persona}")
    
    intent_list = []
    for name, config in intents.items():
        intent_list.append(IntentInfo(
            name=name,
            description=config["description"],
            example_queries=config["patterns"][:3]  # First 3 patterns as examples
        ))
    
    return PersonaIntentsResponse(
        persona=persona_lower,
        intents=intent_list
    )


@app.get("/suggested-questions/{persona}")
def get_suggested_questions(persona: str):
    """
    Get 5 suggested questions for a persona.
    These are the top use cases the agent is optimized for.
    """
    persona_lower = persona.lower()
    
    if persona_lower == "jason":
        return {
            "persona": "jason",
            "role": "Inventory Manager",
            "suggested_questions": [
                {
                    "id": 1,
                    "question": "Where are my inventory issues today?",
                    "description": "View current inventory issues and problems"
                },
                {
                    "id": 2,
                    "question": "Which SKUs are running low with no incoming stock order?",
                    "description": "Find SKUs that are low on stock with no incoming orders"
                },
                {
                    "id": 3,
                    "question": "Which vendors have regular delivery issues?",
                    "description": "Identify vendors with regular delivery problems"
                },
                {
                    "id": 4,
                    "question": "Give me the inventory rush order I should place now?",
                    "description": "Get rush order recommendations for critical inventory needs"
                },
                {
                    "id": 5,
                    "question": "Give me the inventory transfer order I need now?",
                    "description": "Get inventory transfer recommendations between locations"
                }
            ]
        }
    elif persona_lower == "claire":
        return {
            "persona": "claire",
            "role": "Sales Representative",
            "suggested_questions": [
                {
                    "id": 1,
                    "question": "Which customer onboarding is delayed?",
                    "description": "View customers with delayed onboarding"
                },
                {
                    "id": 2,
                    "question": "Which customers have requested for amendment this week?",
                    "description": "View customers who requested amendments this week"
                },
                {
                    "id": 3,
                    "question": "Which customers have been onboarded last week and are ready for first order?",
                    "description": "View recently onboarded customers ready for their first order"
                },
                {
                    "id": 4,
                    "question": "Show me recent material added to master data?",
                    "description": "View recent materials added to master data"
                },
                {
                    "id": 5,
                    "question": "Show my last month sales commission?",
                    "description": "View last month's sales commission"
                }
            ]
        }
    else:
        raise HTTPException(status_code=400, detail=f"Unknown persona: {persona}")


@app.get("/")
def home():
    """Health check endpoint."""
    return {
        "message": "Multi-Persona RAG Assistant API is running.",
        "version": "2.1.0",
        "personas": ["jason (inventory)", "claire (onboarding)"],
        "endpoints": {
            "query": "POST /query - Plain text responses (original format)",
            "query_structured": "POST /query-structured - NEW: Structured JSON responses",
            "compact_query": "POST /compact-query - Extract numeric KPI values",
            "intents": "GET /intents/{persona} - List supported intents",
            "suggested_questions": "GET /suggested-questions/{persona} - Get 5 suggested questions"
        },
        "query_response_format": {
            "result": "Plain text response string"
        },
        "query_structured_response_format": {
            "type": "text | list | action | follow_up",
            "title": "Response header",
            "message": "Additional context (optional)",
            "items": "List of structured items (sku, quantity, vendor, status, etc.)",
            "actions": "Available action buttons (id, label, type)"
        }
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
