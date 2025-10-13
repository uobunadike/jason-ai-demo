# --- Core Imports ---
import os
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache
from typing import Literal, Tuple, List, Dict, Any

# LangChain + FAISS Imports
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Load environment variables
load_dotenv()

# Azure OpenAI configs
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

DATA_ROOT = os.path.join(os.getcwd(), "data")
FAISS_ROOT = os.path.join(os.getcwd(), "faiss_index")


# -----------------------
# Helpers
# -----------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _row_to_text(row: pd.Series) -> str:
    # Convert a record to a readable, dense line
    parts = []
    for k, v in row.items():
        if pd.notna(v):
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


# -----------------------
# CLAIRE: Metadata + Loader
# -----------------------
def _extract_claire_metadata(df: pd.DataFrame, filename: str) -> List[Dict[str, Any]]:
    cols = set(df.columns)
    metas = []
    for _, row in df.iterrows():
        meta = {
            "persona": "claire",
            "source_file": filename,
            "domain": filename.replace(".csv", ""),
            "customer_id": row.get("customer_id", None),
        }

        # Owner / Reviewer
        for key in ("owner", "reviewer"):
            if key in cols:
                meta["owner"] = None if pd.isna(row.get(key)) else str(row.get(key))
                if meta["owner"]:
                    break

        # Status-ish
        for key in ("status", "esign_status", "result", "cleared"):
            if key in cols:
                meta["status"] = None if pd.isna(row.get(key)) else str(row.get(key))
                if meta["status"]:
                    break

        # Time-ish
        for key in ("timestamp", "due_date", "sent_time", "cleared_date", "cycle_time_days"):
            if key in cols:
                meta["timestamp"] = None if pd.isna(row.get(key)) else str(row.get(key))
                if meta.get("timestamp"):
                    break

        # Short summary (helps retrieval even when text is dense)
        summary_bits = []
        for key in ("stage", "task", "blockers", "result", "status"):
            if key in cols and pd.notna(row.get(key)):
                summary_bits.append(f"{key}: {row.get(key)}")
        meta["summary"] = " | ".join(summary_bits)

        metas.append(meta)
    return metas


@lru_cache(maxsize=1)
def _load_claire_texts_and_meta() -> Tuple[List[str], List[Dict[str, Any]]]:
    data_dir = os.path.join(DATA_ROOT, "claire")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Claire data folder not found: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        raise ValueError("No CSV files found in data/claire/")

    texts, metas = [], []
    for fn in files:
        path = os.path.join(data_dir, fn)
        df = pd.read_csv(path)
        df = _normalize_cols(df)

        # Texts
        for _, row in df.iterrows():
            texts.append(_row_to_text(row))

        # Metadata
        metas.extend(_extract_claire_metadata(df, fn))

    # Ensure alignment
    assert len(texts) == len(metas), "Texts and metadata count mismatch for Claire."
    return texts, metas


# -----------------------
# JASON: Metadata + Loader (CSV + XLSX)
# -----------------------
def _extract_jason_metadata(df: pd.DataFrame, filename: str) -> List[Dict[str, Any]]:
    cols = set(df.columns)
    metas = []
    for _, row in df.iterrows():
        meta = {
            "persona": "jason",
            "source_file": filename,
            "domain": filename.replace(".csv", "").replace(".xlsx", ""),
        }

        # SKU / Product
        for key in ("sku", "sku_id", "product_id", "item_id"):
            if key in cols and pd.notna(row.get(key)):
                meta["sku"] = str(row.get(key))
                break

        # Vendor / Supplier
        for key in ("vendor", "supplier_name", "supplier_id"):
            if key in cols and pd.notna(row.get(key)):
                meta["vendor"] = str(row.get(key))
                break

        # Metric-ish (risk_score, price, avg consumption, etc.)
        for key in ("risk_score", "price", "avg_consumption", "cogs_calculated"):
            if key in cols and pd.notna(row.get(key)):
                meta["metric"] = str(row.get(key))
                break

        # Quantity-ish
        for key in ("quantity", "qty", "qty_suggested", "on_hand", "reorder_point"):
            if key in cols and pd.notna(row.get(key)):
                meta["quantity"] = str(row.get(key))
                break

        # Time-ish
        for key in ("date", "week", "timestamp", "event_timestamp"):
            if key in cols and pd.notna(row.get(key)):
                meta["timestamp"] = str(row.get(key))
                break

        # Quick summary
        summary_bits = []
        for key in ("event_type", "risk_reason", "location", "supplier_name", "lead_time"):
            if key in cols and pd.notna(row.get(key)):
                summary_bits.append(f"{key}: {row.get(key)}")
        meta["summary"] = " | ".join(summary_bits)

        metas.append(meta)
    return metas


def _read_any_table(path: str) -> pd.DataFrame:
    if path.endswith(".csv"):
        return _normalize_cols(pd.read_csv(path))
    elif path.endswith(".xlsx") or path.endswith(".xls"):
        # Assumes single logical table per file (your new source). If multiple sheets exist,
        # we concatenate them. No header-row hunting (old KPI logic) because new files are clean.
        xls = pd.ExcelFile(path, engine="openpyxl")
        frames = []
        for sheet in xls.sheet_names:
            frames.append(_normalize_cols(xls.parse(sheet)))
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames, ignore_index=True)
    else:
        raise ValueError(f"Unsupported file type: {path}")


@lru_cache(maxsize=1)
def _load_jason_texts_and_meta() -> Tuple[List[str], List[Dict[str, Any]]]:
    data_dir = os.path.join(DATA_ROOT, "jason")
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Jason data folder not found: {data_dir}")
    files = [f for f in os.listdir(data_dir) if f.endswith((".csv", ".xlsx", ".xls"))]
    if not files:
        raise ValueError("No CSV/XLSX files found in data/jason/")

    texts, metas = [], []
    for fn in files:
        path = os.path.join(data_dir, fn)
        df = _read_any_table(path)
        if df.empty:
            continue

        # Texts
        for _, row in df.iterrows():
            texts.append(_row_to_text(row))

        # Metadata
        metas.extend(_extract_jason_metadata(df, fn))

    # Ensure alignment
    assert len(texts) == len(metas), "Texts and metadata count mismatch for Jason."
    return texts, metas


# -----------------------
# FAISS build/load
# -----------------------
def _build_or_load_faiss(persona: str, texts: List[str], metas: List[Dict[str, Any]], force_rebuild: bool = False):
    """
    Loads persona-specific FAISS if present; otherwise builds and saves it.
    Falls back to legacy top-level 'faiss_index' only when not forcing rebuild.
    """
    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment=embedding_deployment,
        api_version=api_version
    )

    faiss_dir = os.path.join(FAISS_ROOT, persona)
    os.makedirs(faiss_dir, exist_ok=True)

    index_path = os.path.join(faiss_dir, "index.faiss")
    legacy_path = os.path.join(FAISS_ROOT, "index.faiss")

    # ✅ When forcing rebuild, always create a fresh index
    if not force_rebuild:
        if os.path.exists(index_path):
            return FAISS.load_local(faiss_dir, embeddings, allow_dangerous_deserialization=True)
        if not os.path.exists(index_path) and os.path.exists(legacy_path) and persona == "jason":
            return FAISS.load_local(FAISS_ROOT, embeddings, allow_dangerous_deserialization=True)

    print(f"🧠 Building new FAISS index for {persona}...")
    db = FAISS.from_texts(texts, embedding=embeddings, metadatas=metas)
    db.save_local(faiss_dir)
    print(f"✅ New FAISS index saved to {faiss_dir}")
    return db


def build_index(persona: Literal["jason", "claire"] = "claire") -> str:
    persona_lower = persona.lower()
    if persona_lower == "claire":
        texts, metas = _load_claire_texts_and_meta()
    else:
        texts, metas = _load_jason_texts_and_meta()

    print(f"🔄 Building FAISS index for {persona.capitalize()}...")
    _build_or_load_faiss(persona_lower, texts, metas, force_rebuild=True)
    print(f"✅ FAISS index saved under faiss_index/{persona_lower}/")
    return f"Index built successfully for {persona.capitalize()}."

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
# Unified RUN (keeps your old positional order)
# -----------------------
def run(
    query: str,
    model_type: Literal["ollama", "azure"] = "azure",
    model_name: str = "gpt-4",
    persona: Literal["jason", "claire"] = "jason",
) -> str:
    persona_lower = persona.lower()
    persona_cap = persona.capitalize()

    # --- build / rebuild short-circuit (no chain at all) ---
    build_triggers = {"build", "build index", "rebuild", "create index"}
    if query.strip().lower() in build_triggers:
        return build_index(persona_lower)

    # --- load persona data (for fresh index creation if needed) ---
    if persona_lower == "claire":
        texts, metas = _load_claire_texts_and_meta()
    else:
        texts, metas = _load_jason_texts_and_meta()

    # --- load or build FAISS (returns a ready vectorstore) ---
    db = _build_or_load_faiss(persona_lower, texts, metas)

    # --- LLM ---
    llm = _get_llm(model_type, model_name)

    # --- greeting bypass (kept from your old behavior) ---
    if query.strip().lower() in {"hi", "hello", "hey", "what's up?", "how are you?"}:
        return llm.predict(query)

    # --- prompt: use 'question' (RetrievalQA maps input_key -> question_key) ---
    # --- static system prompt ---
    if persona_lower == "claire":
        system_prompt = (
        "You are Claire, the Customer Onboarding Assistant.\n"
        "You help users manage onboarding cases, credit applications, compliance checks, and SAP readiness.\n"
        "If the user’s question involves starting, checking, sending, approving, or exporting a process or document,\n"
        "respond using this structured format:\n\n"
        "Outcome: (Summarize what you accomplished or simulated)\n"
        "Data written: (List key data updates or fields that changed)\n"
        "Visibility: (Explain what the user would see or where it appears)\n"
        "Audit: (Describe what was logged and by whom)\n\n"
        "If the question is general or analytical (like 'what is the status' or 'who owns this case'),\n"
        "respond conversationally and focus on insights or facts.\n"
        "Be concise, confident, and friendly."
    )
    else:
       system_prompt = (
        f"You are {persona_cap}, an intelligent assistant focused on inventory and operations insights.\n"
        "Use the provided context and metadata (like SKU, vendor, quantity, or site) to explain clearly.\n"
        "When possible, highlight key trends, alerts, or recommendations.\n"
        "Always be concise, structured, and professional."
    )
    # --- create a template that only expects 'question' and 'context' ---
    prompt_template = """{system_prompt}

{context}

Question: {question}
"""

    # Insert the system prompt text directly so LangChain doesn't need it as input
    prompt_text = prompt_template.format(
        system_prompt=system_prompt, context="{context}", question="{question}"
    )

    prompt = PromptTemplate(
        input_variables=["question", "context"],
        template=prompt_text,
    )

    # --- build RetrievalQA with explicit keys to avoid version mismatch ---
    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        input_key="question",  
        # output_key="result",  
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context",
        },
    )

    # --- run the chain; RetrievalQA will fetch docs & fill {context} for us ---
    result = chain.invoke({"question": query})
    return result["result"]