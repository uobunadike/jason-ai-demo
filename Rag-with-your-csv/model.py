# --- Core Imports ---
import sys
import os
from dotenv import load_dotenv
import pandas as pd
from functools import lru_cache
from typing import Literal

# SQLite override for Azure compatibility
try:
    __import__('pysqlite3')
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# --- LangChain + Chroma Imports ---
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.chat_models import AzureChatOpenAI
from langchain_openai import AzureOpenAIEmbeddings  # NEW: Azure native
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# --- Load environment variables ---
load_dotenv()

# Azure OpenAI configs
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
chat_deployment = os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

# --- Data Processing ---
def detect_sheet_type(df: pd.DataFrame) -> str:
    columns = pd.Index([str(col).lower() for col in df.columns])
    if 'event_type' in columns and 'event_timestamp' in columns:
        return 'inventory_events'
    elif 'unit_cost' in columns and 'cogs_calculated' in columns:
        return 'financial_metrics'
    return 'unknown'

def process_sheet(df: pd.DataFrame, sheet_type: str) -> str:
    text_chunks = []
    df.columns = [str(col).lower() for col in df.columns]

    if sheet_type == 'inventory_events':
        df = df.dropna(subset=['event_type', 'sku_id'])
        for _, row in df.iterrows():
            try:
                text_chunks.append(
                    f"{row['event_type']} ({pd.to_datetime(row['event_timestamp']).strftime('%Y-%m-%d')}): "
                    f"Sold {row.get('quantity_sold', 'N/A')} units of {row.get('item_name', 'Unknown Item')} (SKU: {row['sku_id']})"
                )
            except Exception:
                continue

    elif sheet_type == 'financial_metrics':
        df = df.dropna(subset=['sku_id', 'unit_cost'])
        for _, row in df.iterrows():
            try:
                text_chunks.append(
                    f"{row.get('item_name', 'Unknown Item')} (SKU: {row['sku_id']}): "
                    f"Unit Cost: ${float(row['unit_cost']):.2f}, COGS: ${float(row.get('cogs_calculated', 0.0)):.2f}"
                )
            except Exception:
                continue

    return "\n".join(text_chunks)

@lru_cache(maxsize=1)
def load_kpi_data(file_name: str = "Kpi_tables.xlsx") -> str:
    try:
        raw_sheets = pd.read_excel(file_name, sheet_name=None, header=None, engine="openpyxl")
        all_text = []

        for _, df in raw_sheets.items():
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

        return "\n\n".join(all_text)

    except Exception as e:
        raise ValueError(f"Excel processing error: {str(e)}")

@lru_cache(maxsize=1)
def get_vector_store():
    """Create and cache the vector store from KPI data."""
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )

    docs = text_splitter.split_text(load_kpi_data())

    return Chroma.from_texts(
        texts=docs,
        embedding=AzureOpenAIEmbeddings(
            api_key=api_key,
            azure_endpoint=endpoint,
            deployment=embedding_deployment,
            api_version=api_version
        ),
        persist_directory="./chroma_db"
    )

# --- Core Run Method ---
def run(query: str, model_name: str = "gpt-35-turbo") -> str:
    """Run the RAG pipeline with Azure OpenAI and Chroma."""
    db = get_vector_store()

    llm = AzureChatOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment_name=chat_deployment,
        api_version=api_version,
        temperature=0.3
    )

    prompt_template = """You are a data analyst reviewing multi-sheet inventory and financial data.

Use the following extracted data context to answer the question accurately:

{context}

Now answer this question: {question}
"""

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=prompt_template
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type="stuff",  # Force stuff to avoid chunk_size error
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.invoke({"query": query})['result']

# --- End of file ---
