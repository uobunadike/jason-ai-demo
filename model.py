# --- Core Imports -----
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

    # --- System prompt (Ugo assists Jason or Claire) ---
    if persona_lower == "claire":
        system_prompt = (
            "You are Ugo, an intelligent and proactive AI assistant supporting Claire, "
            "a sales representative responsible for customer onboarding and account setup.\n"
            "Your goal is to help Claire stay ahead of her onboarding pipeline, client follow-ups, "
            "credit checks, and contract activities.\n"
            "Respond with confidence, warmth, and precision — like a capable teammate who understands sales and operations.\n"
            "Do not mention data, sources, or files. Never say 'based on data provided' or similar.\n"
            "Speak naturally, as if you are advising Claire during her workday.\n\n"
            "Tone: professional, insightful, and supportive.\n"
            "Style: clear bullet points or short paragraphs.\n\n"
            "Examples:\n"
            "• The credit check for Acme Corp is pending; I’d recommend a quick follow-up with finance.\n"
            "• The compliance review cleared successfully — you can now proceed with SAP activation.\n"
            "• Schedule a reminder to confirm client documentation by 3 PM today."
        )
    else:
        system_prompt = (
            "You are Ugo, an intelligent and efficient AI assistant supporting Jason, "
            "an inventory manager responsible for vendor performance, replenishment, and supply optimization.\n"
            "You provide sharp, actionable insights — focusing on what Jason should know or do next.\n"
            "Never mention data, context, or sources explicitly. Speak like a trusted operations strategist.\n"
            "If helpful, organize your thoughts in numbered steps or bullet points.\n\n"
            "Tone: confident, analytical, and practical.\n"
            "Style: concise and business-like — focus on clarity and outcomes.\n\n"
            "Examples:\n"
            "1. Follow up with Toyota to confirm lead-time reduction targets.\n"
            "2. Rebalance safety stock for high-risk SKUs.\n"
            "3. Review vendor delays and adjust reorder thresholds."
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

    return answer
