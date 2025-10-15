# --- Core Imports ----
import os
import pandas as pd
from dotenv import load_dotenv
from functools import lru_cache
from typing import Literal, Tuple, List, Dict, Any
from pathlib import Path

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

# Detect the true project base directory (works on Render, Azure, and locally)
BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR  # jason and claire live directly under project root
FAISS_ROOT = BASE_DIR / "faiss_index"

print("📁 BASE_DIR:", BASE_DIR)
print("📁 DATA_ROOT:", DATA_ROOT)
print("📁 FAISS_ROOT:", FAISS_ROOT)

# --- Auto-create folders if missing (for Render) ---
for persona in ["jason", "claire"]:
    folder = BASE_DIR / persona
    if not folder.exists():
        print(f"⚠️ {persona.capitalize()} folder missing — creating empty one.")
        folder.mkdir(parents=True, exist_ok=True)

# -----------------------
# Helpers
# -----------------------
def _normalize_cols(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip().lower() for c in df.columns]
    return df


def _row_to_text(row: pd.Series) -> str:
    parts = []
    for k, v in row.items():
        if pd.notna(v):
            parts.append(f"{k}: {v}")
    return " | ".join(parts)


# -----------------------
# CLAIRE
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
        for key in ("owner", "reviewer"):
            if key in cols and pd.notna(row.get(key)):
                meta["owner"] = str(row.get(key))
                break
        for key in ("status", "esign_status", "result", "cleared"):
            if key in cols and pd.notna(row.get(key)):
                meta["status"] = str(row.get(key))
                break
        for key in ("timestamp", "due_date", "sent_time", "cleared_date", "cycle_time_days"):
            if key in cols and pd.notna(row.get(key)):
                meta["timestamp"] = str(row.get(key))
                break
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
        print(f"⚠️ Claire folder missing, rebuilding FAISS index...")
        from model import build_index
        build_index("claire")
        return [], []
    files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if not files:
        raise ValueError("No CSV files found in data/claire/")
    texts, metas = [], []
    for fn in files:
        path = os.path.join(data_dir, fn)
        df = pd.read_csv(path)
        df = _normalize_cols(df)
        for _, row in df.iterrows():
            texts.append(_row_to_text(row))
        metas.extend(_extract_claire_metadata(df, fn))
    assert len(texts) == len(metas), "Texts and metadata count mismatch for Claire."
    return texts, metas


# -----------------------
# JASON
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
        for key in ("sku", "sku_id", "product_id", "item_id"):
            if key in cols and pd.notna(row.get(key)):
                meta["sku"] = str(row.get(key))
                break
        for key in ("vendor", "supplier_name", "supplier_id"):
            if key in cols and pd.notna(row.get(key)):
                meta["vendor"] = str(row.get(key))
                break
        for key in ("risk_score", "price", "avg_consumption", "cogs_calculated"):
            if key in cols and pd.notna(row.get(key)):
                meta["metric"] = str(row.get(key))
                break
        for key in ("quantity", "qty", "qty_suggested", "on_hand", "reorder_point"):
            if key in cols and pd.notna(row.get(key)):
                meta["quantity"] = str(row.get(key))
                break
        for key in ("date", "week", "timestamp", "event_timestamp"):
            if key in cols and pd.notna(row.get(key)):
                meta["timestamp"] = str(row.get(key))
                break
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
    elif path.endswith((".xlsx", ".xls")):
        xls = pd.ExcelFile(path, engine="openpyxl")
        frames = [_normalize_cols(xls.parse(sheet)) for sheet in xls.sheet_names]
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
    else:
        raise ValueError(f"Unsupported file type: {path}")


@lru_cache(maxsize=1)
def _load_jason_texts_and_meta() -> Tuple[List[str], List[Dict[str, Any]]]:
    data_dir = os.path.join(DATA_ROOT, "jason")
    if not os.path.exists(data_dir):
        print(f"⚠️ Jason folder missing, rebuilding FAISS index...")
        from model import build_index
        build_index("jason")
        return [], []
    files = [f for f in os.listdir(data_dir) if f.endswith((".csv", ".xlsx", ".xls"))]
    if not files:
        raise ValueError("No CSV/XLSX files found in data/jason/")
    texts, metas = [], []
    for fn in files:
        path = os.path.join(data_dir, fn)
        df = _read_any_table(path)
        if df.empty:
            continue
        for _, row in df.iterrows():
            texts.append(_row_to_text(row))
        metas.extend(_extract_jason_metadata(df, fn))
    assert len(texts) == len(metas), "Texts and metadata count mismatch for Jason."
    return texts, metas


# -----------------------
# FAISS build/load
# -----------------------
def _build_or_load_faiss(persona: str, texts: List[str], metas: List[Dict[str, Any]], force_rebuild: bool = False):
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
# RUN
# -----------------------
def run(
    query: str,
    model_type: Literal["ollama", "azure"] = "azure",
    model_name: str = "gpt-4",
    persona: Literal["jason", "claire"] = "jason",
) -> str:
    persona_lower = persona.lower()
    persona_cap = persona.capitalize()
    build_triggers = {"build", "build index", "rebuild", "create index"}
    if query.strip().lower() in build_triggers:
        return build_index(persona_lower)

    if persona_lower == "claire":
        texts, metas = _load_claire_texts_and_meta()
    else:
        texts, metas = _load_jason_texts_and_meta()

    db = _build_or_load_faiss(persona_lower, texts, metas)
    llm = _get_llm(model_type, model_name)

    if query.strip().lower() in {"hi", "hello", "hey", "what's up?", "how are you?"}:
        return llm.predict(query)

    system_prompt = (
        "You are Claire, the Customer Onboarding Assistant.\n"
        if persona_lower == "claire"
        else f"You are {persona_cap}, an intelligent assistant focused on inventory and operations insights.\n"
    )

    prompt_template = """{system_prompt}

{context}

Question: {question}
"""
    prompt_text = prompt_template.format(
        system_prompt=system_prompt, context="{context}", question="{question}"
    )
    prompt = PromptTemplate(input_variables=["question", "context"], template=prompt_text)

    retriever = db.as_retriever(search_kwargs={"k": 3})
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        return_source_documents=False,
        input_key="question",
        chain_type_kwargs={"prompt": prompt, "document_variable_name": "context"},
    )
    result = chain.invoke({"question": query})
    return result["result"]
