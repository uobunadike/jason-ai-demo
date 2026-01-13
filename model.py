# --- Core Imports -----
import os
import json
import re
from typing import Literal, Dict, Any, List, Optional
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -----------------------
# Load environment variables (Render and Azure pick from env)
# -----------------------
from dotenv import load_dotenv
load_dotenv()

# Azure OpenAI configs
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# Paths
BASE_DIR = os.getcwd()
FAISS_ROOT = os.path.join(BASE_DIR, "faiss_index")

print(f"[INFO] BASE_DIR: {BASE_DIR}")
print(f"[INFO] FAISS_ROOT: {FAISS_ROOT}")


# -----------------------
# Intent Definitions for Each Persona
# -----------------------
JASON_INTENTS = {
    "inventory_issues": {
        "patterns": ["inventory issues", "issues today", "inventory problems", "stock issues", "what are my issues", "where are my inventory issues"],
        "requires_list": True,
        "description": "View current inventory issues and problems",
        "suggested_question": "Where are my inventory issues today?"
    },
    "low_stock_no_order": {
        "patterns": ["running low", "low with no", "no incoming", "no stock order", "skus running low", "low inventory no order"],
        "requires_list": True,
        "has_action": True,
        "description": "Find SKUs that are low on stock with no incoming orders",
        "suggested_question": "Which SKUs are running low with no incoming stock order?"
    },
    "vendor_delivery_issues": {
        "patterns": ["vendor issues", "delivery issues", "vendor delays", "regular delivery issues", "supplier delays", "vendors have issues"],
        "requires_list": True,
        "description": "Identify vendors with regular delivery problems",
        "suggested_question": "Which vendors have regular delivery issues?"
    },
    "rush_order": {
        "patterns": ["rush order", "urgent order", "rush order now", "place rush", "inventory rush", "emergency order"],
        "requires_list": True,
        "has_action": True,
        "description": "Get rush order recommendations for critical inventory needs",
        "suggested_question": "Give me the inventory rush order I should place now?"
    },
    "transfer_order": {
        "patterns": ["transfer order", "inventory transfer", "transfer now", "stock transfer", "move inventory", "transfer suggestion"],
        "requires_list": True,
        "has_action": True,
        "description": "Get inventory transfer recommendations between locations",
        "suggested_question": "Give me the inventory transfer order I need now?"
    }
}

CLAIRE_INTENTS = {
    "onboarding_delayed": {
        "patterns": ["onboarding delayed", "delayed onboarding", "customer delayed", "onboarding issues", "stuck onboarding"],
        "requires_list": True,
        "description": "View customers with delayed onboarding",
        "suggested_question": "Which customer onboarding is delayed?"
    },
    "amendment_requests": {
        "patterns": ["amendment", "requested amendment", "amendment this week", "change request", "modification request"],
        "requires_list": True,
        "description": "View customers who requested amendments this week",
        "suggested_question": "Which customers have requested for amendment this week?"
    },
    "ready_for_first_order": {
        "patterns": ["first order", "ready for order", "onboarded last week", "new customers ready", "ready to order"],
        "requires_list": True,
        "has_action": True,
        "description": "View recently onboarded customers ready for their first order",
        "suggested_question": "Which customers have been onboarded last week and are ready for first order?"
    },
    "master_data_materials": {
        "patterns": ["master data", "material added", "recent material", "new material", "material master"],
        "requires_list": True,
        "description": "View recent materials added to master data",
        "suggested_question": "Show me recent material added to master data?"
    },
    "sales_commission": {
        "patterns": ["sales commission", "my commission", "last month commission", "commission report", "earnings"],
        "requires_list": True,
        "description": "View last month's sales commission",
        "suggested_question": "Show my last month sales commission?"
    }
}


# -----------------------
# Intent Detection
# -----------------------
def detect_intent(query: str, persona: str) -> Optional[Dict[str, Any]]:
    """Detect the user's intent based on their query and persona."""
    query_lower = query.lower()
    intents = JASON_INTENTS if persona == "jason" else CLAIRE_INTENTS
    
    for intent_name, intent_config in intents.items():
        for pattern in intent_config["patterns"]:
            if pattern in query_lower:
                return {"name": intent_name, **intent_config}
    
    return None


def check_missing_info(query: str, intent: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Check if the query is missing required information."""
    query_lower = query.lower()
    
    # Check if SKU is required but missing
    if intent.get("requires_sku"):
        # Look for SKU patterns like "SKU-123" or specific product mentions
        sku_pattern = r'sku[-\s]?\d+'
        if not re.search(sku_pattern, query_lower):
            return {
                "type": "follow_up",
                "title": "I'd be happy to help create an order.",
                "message": "Please provide the SKU(s) you would like to include in this order.",
                "items": [],
                "actions": []
            }
    
    return None


# -----------------------
# Load FAISS Index Helper
# -----------------------
def load_faiss_index(persona: Literal["jason", "claire"]) -> FAISS:
    """Load prebuilt FAISS index for the given persona."""
    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment=embedding_deployment,
        api_version=api_version
    )

    faiss_dir = os.path.join(FAISS_ROOT, persona)
    if not os.path.exists(faiss_dir):
        raise FileNotFoundError(f"[ERROR] FAISS index folder not found for {persona}: {faiss_dir}")

    print(f"[OK] Loading FAISS index for {persona} from {faiss_dir}...")
    return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)


# -----------------------
# Model selector
# -----------------------
def _get_llm(model_type: Literal["ollama", "azure"], model_name: str):
    """Helper to select between Azure OpenAI and Ollama models."""
    if model_type == "ollama":
        return OllamaLLM(model=model_name, temperature=0.3)
    return AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=chat_deployment,
        api_version=api_version,
        temperature=0.4
    )


# -----------------------
# JSON Response Builder
# -----------------------
def build_structured_response(
    response_type: Literal["text", "list", "action", "follow_up"],
    title: str,
    items: List[Dict[str, Any]] = None,
    message: str = None,
    actions: List[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Build a structured JSON response for the frontend."""
    return {
        "type": response_type,
        "title": title,
        "message": message,
        "items": items or [],
        "actions": actions or []
    }


def parse_llm_response_to_structured(raw_response: str, intent: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Parse the LLM's raw text response into a structured JSON format."""
    
    # Clean up the response
    raw_response = raw_response.strip()
    
    # Try to detect if this is a list response
    lines = raw_response.split('\n')
    items = []
    title_parts = []
    
    # Patterns for list items
    list_patterns = [
        r'^[\d]+[\.\)]\s*',      # 1. or 1)
        r'^[-•]\s*',              # - or •
        r'^\*\s*',                # *
    ]
    
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        is_list_item = False
        for pattern in list_patterns:
            if re.match(pattern, line):
                is_list_item = True
                # Clean the list marker
                cleaned_line = re.sub(pattern, '', line).strip()
                
                # Try to extract structured data from the line
                item = parse_list_item(cleaned_line)
                items.append(item)
                break
        
        if not is_list_item:
            title_parts.append(line)
    
    # Determine response type
    if items:
        response_type = "list"
        title = ' '.join(title_parts) if title_parts else "Here's what I found:"
    else:
        response_type = "text"
        title = raw_response
        items = []
    
    # Add actions if the intent supports it
    actions = []
    if intent and intent.get("has_action") and items:
        intent_name = intent.get("name", "")
        
        # Jason intents
        if intent_name == "low_stock_no_order":
            actions.append({
                "id": "create_order",
                "label": "Create Order",
                "type": "primary"
            })
        elif intent_name == "rush_order":
            actions.append({
                "id": "process_rush_order",
                "label": "Process Rush Order",
                "type": "primary"
            })
        elif intent_name == "transfer_order":
            actions.append({
                "id": "process_transfer",
                "label": "Process Transfer",
                "type": "primary"
            })
        
        # Claire intents
        elif intent_name == "ready_for_first_order":
            actions.append({
                "id": "create_first_order",
                "label": "Create First Order",
                "type": "primary"
            })
    
    return build_structured_response(
        response_type=response_type,
        title=title,
        items=items,
        actions=actions
    )


def parse_list_item(line: str) -> Dict[str, Any]:
    """Parse a single list item line into structured data."""
    item = {"text": line}
    
    # Try to extract SKU
    sku_match = re.search(r'(SKU[-\s]?\d+)', line, re.IGNORECASE)
    if sku_match:
        item["sku"] = sku_match.group(1).upper().replace(' ', '-')
    
    # Try to extract quantity
    qty_patterns = [
        r'(\d+)\s*units?',
        r'qty[:\s]*(\d+)',
        r'quantity[:\s]*(\d+)',
        r'order\s*(\d+)',
    ]
    for pattern in qty_patterns:
        qty_match = re.search(pattern, line, re.IGNORECASE)
        if qty_match:
            item["quantity"] = int(qty_match.group(1))
            break
    
    # Try to extract vendor
    vendor_match = re.search(r'(?:from|vendor[:\s]*|supplier[:\s]*)([A-Za-z][A-Za-z\s]+?)(?:\.|,|$)', line, re.IGNORECASE)
    if vendor_match:
        item["vendor"] = vendor_match.group(1).strip()
    
    # Try to extract cost/price
    cost_match = re.search(r'\$?([\d,]+\.?\d*)', line)
    if cost_match:
        cost_str = cost_match.group(1).replace(',', '')
        try:
            item["cost"] = float(cost_str)
        except ValueError:
            pass
    
    # Try to extract status
    status_keywords = ["pending", "approved", "rejected", "in progress", "completed", "low stock", "critical", "ok"]
    for status in status_keywords:
        if status.lower() in line.lower():
            item["status"] = status.title()
            break
    
    return item


# -----------------------
# Main Run Function (FAISS-only)
# -----------------------
def run(
    query: str,
    model_type: Literal["ollama", "azure"] = "azure",
    model_name: str = "gpt-4",
    persona: Literal["jason", "claire"] = "jason",
) -> Dict[str, Any]:
    """
    Main RAG function that returns structured JSON responses.
    
    Returns:
        Dict with structure:
        {
            "type": "text" | "list" | "action" | "follow_up",
            "title": str,
            "message": str | None,
            "items": [{"text": str, "sku": str, "quantity": int, ...}, ...],
            "actions": [{"id": str, "label": str, "type": str}, ...]
        }
    """
    persona_lower = persona.lower()
    
    # --- Detect Intent ---
    intent = detect_intent(query, persona_lower)
    
    # --- Check for missing required information ---
    if intent:
        missing_info = check_missing_info(query, intent)
        if missing_info:
            return missing_info
    
    # --- Load FAISS directly (no CSVs) ---
    db = load_faiss_index(persona_lower)

    # --- LLM selection ---
    llm = _get_llm(model_type, model_name)

    # --- System prompt (Ugo assists Jason or Claire) ---
    if persona_lower == "claire":
        system_prompt = (
            "You are Ugo, an intelligent and proactive AI assistant supporting Claire, "
            "a sales representative responsible for customer onboarding and account setup.\n"
            "Do not mention data, sources, or files. Never say 'based on data provided' or similar.\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- Start with ONE short sentence (max 10-15 words) as introduction, then go straight to the list.\n"
            "- Do NOT add explanations, recommendations, or extra context before or after the list.\n"
            "- Use numbered format (1. 2. 3.) with each item on a new line.\n"
            "- For each customer: include customer ID/name, status, due date, owner.\n"
            "- Be direct. No filler words. No disclaimers.\n\n"
            "Example format:\n"
            "Here are the delayed onboardings:\n"
            "1. CUST0012 | Blocked | Due: 2025-10-04 | Owner: Ops Intake\n"
            "2. CUST0018 | Blocked | Due: 2025-10-04 | Owner: Credit Desk"
        )
    else:
        system_prompt = (
            "You are Ugo, an intelligent and efficient AI assistant supporting Jason, "
            "an inventory manager responsible for vendor performance, replenishment, and supply optimization.\n"
            "You provide sharp, actionable insights — focusing on what Jason should know or do next.\n"
            "Never mention data, context, or sources explicitly. Speak like a trusted operations strategist.\n\n"
            "CRITICAL FORMATTING RULES:\n"
            "- Start with ONE short sentence (max 10-15 words) as introduction, then go straight to the list.\n"
            "- Do NOT add explanations, recommendations, or extra context before or after the list.\n"
            "- Use numbered format (1. 2. 3.) with each item on a new line.\n"
            "- For each SKU: include SKU ID, quantity needed, vendor name, site/location.\n"
            "- Be direct. No filler words. No disclaimers.\n\n"
            "Example format:\n"
            "Here are the SKUs to order today:\n"
            "1. SKU-606 | 160 units | QuickParts Inc. | ATL-01\n"
            "2. SKU-303 | 100 units | Northwind Supply | DAL-01"
        )

    # --- Prompt Template ---
    prompt_template = """{system_prompt}

{context}

Question: {question}
"""
    prompt_text = prompt_template.format(
        system_prompt=system_prompt, context="{context}", question="{question}"
    )

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_text,
    )

    # --- Build Retrieval Chain ---
    retriever = db.as_retriever(search_kwargs={"k": 5})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        input_key="question",
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
    )

    # --- Run Query ---
    result = chain.invoke({"question": query})
    answer = result["result"]

    # --- Post-process cleanup ---
    cleanup_phrases = [
        "based on the data provided",
        "based on data provided",
        "according to the context",
        "according to the retrieved data",
        "from the provided context",
    ]
    for phrase in cleanup_phrases:
        answer = answer.replace(phrase, "").strip()

    # --- Parse into structured response ---
    structured_response = parse_llm_response_to_structured(answer, intent)
    
    return structured_response


# -----------------------
# Legacy Run Function (for backward compatibility)
# -----------------------
def run_legacy(
    query: str,
    model_type: Literal["ollama", "azure"] = "azure",
    model_name: str = "gpt-4",
    persona: Literal["jason", "claire"] = "jason",
) -> str:
    """
    Legacy run function that returns plain text.
    Use this for backward compatibility with existing integrations.
    """
    result = run(query, model_type, model_name, persona)
    
    # Convert structured response back to plain text
    if result["type"] == "text":
        return result["title"]
    elif result["type"] == "follow_up":
        return f"{result['title']} {result.get('message', '')}"
    else:
        # Combine title and items
        text_parts = [result["title"]]
        for i, item in enumerate(result.get("items", []), 1):
            text_parts.append(f"{i}. {item.get('text', '')}")
        return "\n".join(text_parts)
