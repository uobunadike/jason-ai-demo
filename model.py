# --- Core Imports ----
import os
from typing import Literal
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

print(f"📁 BASE_DIR: {BASE_DIR}")
print(f"📁 FAISS_ROOT: {FAISS_ROOT}")

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
        raise FileNotFoundError(f"❌ FAISS index folder not found for {persona}: {faiss_dir}")

    print(f"✅ Loading FAISS index for {persona} from {faiss_dir}...")
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
# Main Run Function (FAISS-only)
# -----------------------
def run(
    query: str,
    model_type: Literal["ollama", "azure"] = "azure",
    model_name: str = "gpt-4",
    persona: Literal["jason", "claire"] = "jason",
) -> str:
    persona_lower = persona.lower()
    persona_cap = persona.capitalize()

    # --- Load FAISS directly (no CSVs) ---
    db = load_faiss_index(persona_lower)

    # --- LLM selection ---
    llm = _get_llm(model_type, model_name)

    # --- System prompts ---
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
    retriever = db.as_retriever(search_kwargs={"k": 3})
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
    return result["result"]
