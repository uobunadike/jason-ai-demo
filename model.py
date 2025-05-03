# --- Core Imports ---
import sys
import os
from dotenv import load_dotenv
import pandas as pd
from functools import lru_cache
from typing import Literal
# Run in Python terminal
import faiss



# LangChain + FAISS Imports
from langchain.text_splitter import CharacterTextSplitter
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

# --- Sheet Detection ---
def detect_sheet_type(df: pd.DataFrame) -> str:
    columns = pd.Index([str(col).lower() for col in df.columns])
    if 'event_type' in columns and 'event_timestamp' in columns:
        return 'inventory_events'
    elif 'unit_cost' in columns and 'cogs_calculated' in columns:
        return 'financial_metrics'
    elif 'revenue_at_risk' in columns:
        return 'revenue_risk'
    return 'unknown'

# --- Sheet Processing ---
def process_sheet(df: pd.DataFrame, sheet_type: str) -> str:
    text_chunks = []
    df.columns = [str(col).lower() for col in df.columns]

    if sheet_type == 'inventory_events':
        df = df.dropna(subset=['event_type', 'sku_id'])
        for _, row in df.iterrows():
            try:
                text_chunks.append(
                    f"[Inventory Event] Item: {row.get('item_name', 'Unknown Item')} | SKU: {row['sku_id']} | Event Type: {row['event_type']} | Date: {pd.to_datetime(row['event_timestamp']).strftime('%Y-%m-%d')} | Quantity Sold: {row.get('quantity_sold', 'N/A')}"
                )
            except Exception:
                continue

    elif sheet_type == 'financial_metrics':
        df = df.dropna(subset=['sku_id', 'unit_cost'])
        for _, row in df.iterrows():
            try:
                item_name = row.get('item_name', 'Unknown Item')
                sku = row.get('sku_id', 'Unknown SKU')
                unit_cost = float(row.get('unit_cost', 0))
                cogs = float(row.get('cogs_calculated', 0.0))
                quantity = row.get('quantity_sold', 'N/A')
                text_chunks.append(
                    f"[Financial Record] Item: {item_name} | SKU: {sku} | Unit Cost: ${unit_cost:.2f} | Quantity Sold: {quantity} | COGS: ${cogs:.2f}"
                )
            except Exception:
                continue

    elif sheet_type == 'revenue_risk':
        for _, row in df.iterrows():
            text_chunks.append(" | ".join([f"{col.capitalize()}: {row[col]}" for col in df.columns if pd.notna(row[col])]))

    return "\n".join(text_chunks)

# --- Data Loader ---
@lru_cache(maxsize=1)
def load_all_data():
    all_text = []

    try:
        # --- Load KPI Excel ---
        kpi_sheets = pd.read_excel("Kpi_tables.xlsx", sheet_name=None, header=None, engine="openpyxl")
        for _, df in kpi_sheets.items():
            header_row = None
            for idx, row in df.iterrows():
                str_row = row.astype(str).str.lower()
                if any(col in str_row.tolist() for col in ['event_type', 'sku_id', 'unit_cost']):
                    header_row = idx
                    break

            if header_row is None:
                continue

            df.columns = df.iloc[header_row]
            df = df.iloc[header_row + 1:].reset_index(drop=True)
            sheet_type = detect_sheet_type(df)
            sheet_text = process_sheet(df, sheet_type)
            if sheet_text:
                all_text.append(sheet_text)

        # --- Load Revenue Risk Excel ---
        df_risk = pd.read_excel("revenue_at_risk.xlsx", engine="openpyxl")
        df_risk.columns = [str(col).lower() for col in df_risk.columns]
        risk_text = process_sheet(df_risk, 'revenue_risk')
        if risk_text:
            all_text.append(risk_text)

        return "\n\n".join(all_text)

    except Exception as e:
        raise ValueError(f"Data processing error: {str(e)}")

# --- Vector Store Setup ---
@lru_cache(maxsize=1)
def get_vector_store():
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
    chunks = splitter.split_text(load_all_data())

    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment=embedding_deployment,
        api_version=api_version
    )

    faiss_index_dir = "faiss_index"
    if os.path.exists(faiss_index_dir):
        return FAISS.load_local(faiss_index_dir, embeddings, allow_dangerous_deserialization=True)

    db = FAISS.from_texts(chunks, embedding=embeddings)
    db.save_local(faiss_index_dir)
    return db

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

    prompt_template = """You are a smart, observant data analyst. Your tone is calm and conversational.

Begin with a clear, factual answer to the user's question.

Then share a brief, 1-2 sentence insight that helps the user understand the numbers or their implications — like you're chatting with a colleague. Avoid giving instructions or suggesting actions. Avoid filler words like 'interestingly' — focus on the facts and what they mean.

Keep it light, natural, and informative — not robotic or directive.

{context}

Question: {question}
"""

    prompt = PromptTemplate(
        input_variables=["question"],
        template=prompt_template
    )

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


