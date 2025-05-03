# --- Core Imports ---
import os
import re
import pandas as pd
from functools import lru_cache
from typing import Literal, List
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Hardcoded Azure Credentials (for production) ---
api_key = "E5B0O2ag7srAVPZG7aHGXyMrNqFP2KK9Zn05BzPwRoKxX2bSyd4nJQQJ99BDACHYHv6XJ3w3AAAAACOGB1UW"
endpoint = "https://uobun-m9rm8tai-eastus2.openai.azure.com/"
embedding_deployment = "text-embedding-3-small"
chat_deployment = "gpt-4"
api_version = "2024-12-01-preview"

# --- File Paths in Azure ML ---
excel_path = "/mnt/batch/tasks/shared/LS_root/mounts/clusters/uobunadike1/code/Users/uobunadike/revenue_at_risk.xlsx"
faiss_index_dir = "faiss_index"

# --- SKU Utilities ---
def extract_skus(text: str) -> List[str]:
    return [sku.replace(" ", "-").upper() for sku in re.findall(r"SKU[-\s]?\d+", text, re.IGNORECASE)]

def normalize_sku(sku: str) -> str:
    sku = sku.upper().replace(" ", "-")
    if not sku.startswith("SKU-"):
        sku = "SKU-" + re.sub(r"^SKU[-\s]?", "", sku)
    return sku

# --- Load Revenue at Risk Data ---
@lru_cache(maxsize=1)
def load_revenue_docs(file_path: str = excel_path) -> List[Document]:
    try:
        df = pd.read_excel(file_path, engine="openpyxl")
        df.columns = [str(col).lower() for col in df.columns]
        docs = []

        for _, row in df.iterrows():
            text = " | ".join([f"{col.capitalize()}: {row[col]}" for col in df.columns if pd.notna(row[col])])
            skus = extract_skus(text)
            docs.append(Document(page_content=text, metadata={"skus": [normalize_sku(s) for s in skus]}))

        return docs
    except Exception as e:
        raise ValueError(f"Failed to load revenue_at_risk.xlsx: {str(e)}")

# --- Vector Store Setup ---
@lru_cache(maxsize=1)
def get_vector_store():
    docs = load_revenue_docs()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
    split_docs = splitter.split_documents(docs)

    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment=embedding_deployment,
        api_version=api_version
    )

    if os.path.exists(faiss_index_dir):
        return FAISS.load_local(faiss_index_dir, embeddings, allow_dangerous_deserialization=True)

    db = FAISS.from_documents(split_docs, embedding=embeddings)
    db.save_local(faiss_index_dir)
    return db

# --- Hybrid Retrieval ---
def retrieve_with_sku_priority(query: str, db, k=3) -> List[Document]:
    query_skus = [normalize_sku(s) for s in extract_skus(query)]
    if query_skus:
        results = db.similarity_search(
            query,
            k=10,
            filter=lambda metadata: any(sku in metadata.get("skus", []) for sku in query_skus)
        )
        if results:
            return results[:k]
    return db.similarity_search(query, k=k)

# --- Run Method ---
def run(query: str, model_type: Literal["ollama", "azure"] = "azure", model_name: str = "gpt-4") -> str:
    db = get_vector_store()

    llm = (
        OllamaLLM(model=model_name, temperature=0.3)
        if model_type == "ollama"
        else AzureChatOpenAI(
            api_key=api_key,
            azure_endpoint=endpoint,
            deployment_name=chat_deployment,
            api_version=api_version,
            temperature=0.5
        )
    )

    if query.strip().lower() in ["hi", "hello", "hey", "what's up?", "how are you?"]:
        return llm.predict(query)

    prompt_template = """You are a smart, confident data analyst with a helpful and casual tone.

Start with a direct answer to the user's question.

Then briefly explain it and suggest any next steps—like chatting with a teammate casually. Be natural, not robotic. Avoid over-explaining.

Keep it short, clear, and human.

{context}

Question: {question}
"""

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template
    )

    docs = retrieve_with_sku_priority(query, db, k=3)
    context = "\n\n".join(doc.page_content for doc in docs)
    if not context:
        return "Sorry, I couldn't find information about that SKU or topic in the data."

    chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",
        return_source_documents=False,
        chain_type_kwargs={
            "prompt": prompt,
            "document_variable_name": "context"
        }
    )

    return chain.invoke({"query": query})["result"]