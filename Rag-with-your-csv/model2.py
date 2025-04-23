import os
from dotenv import load_dotenv  # Load environment variables from a .env file

import textwrap  # (Optional) For formatting text if needed later

# Import various libraries used in the script
import langchain
import chromadb
import transformers
import openai
import torch
import requests
import json

# Import specific classes and functions from libraries
from transformers import AutoTokenizer
from langchain_community.llms import HuggingFacePipeline  # Updated import
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import CSVLoader  # Updated import
from langchain_community.embeddings import OpenAIEmbeddings  # Updated import
from langchain_community.vectorstores import Chroma  # Updated import
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM



import os
import pandas as pd
from typing import Literal
from dotenv import load_dotenv

# Langchain and embedding/LLM components
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOpenAI
import chromadb


import os
import pandas as pd
from dotenv import load_dotenv
from typing import Literal
from functools import lru_cache

# LangChain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter


import os
import pandas as pd
from dotenv import load_dotenv
from typing import Literal
from functools import lru_cache

# LangChain imports
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import ChatOpenAI
from langchain.text_splitter import CharacterTextSplitter

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")


def detect_sheet_type(df: pd.DataFrame) -> str:
    """Identify sheet type based on column patterns"""
    columns = pd.Index([str(col).lower() for col in df.columns])
    if 'event_type' in columns and 'event_timestamp' in columns:
        return 'inventory_events'
    elif 'unit_cost' in columns and 'cogs_calculated' in columns:
        return 'financial_metrics'
    return 'unknown'


def process_sheet(df: pd.DataFrame, sheet_type: str) -> str:
    """Convert DataFrame to formatted text based on sheet type"""
    text_chunks = []

    # Normalize column names
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
    """Process all Excel sheets and extract formatted text"""
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
    """Create persistent vector store"""
    text_splitter = CharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separator="\n"
    )

    docs = text_splitter.split_text(load_kpi_data())

    return Chroma.from_texts(
        texts=docs,
        embedding=OpenAIEmbeddings(
            openai_api_key=openai_api_key,
            model="text-embedding-3-small"
        ),
        persist_directory="./chroma_db"
    )


def run(query: str, model_type: Literal["ollama", "openai"] = "ollama", model_name: str = "llama3.1") -> str:
    """Run QA pipeline focused on answering questions directly from the data"""
    db = get_vector_store()

    if model_type == "ollama":
        llm = OllamaLLM(model=model_name, temperature=0.3)
    else:
        llm = ChatOpenAI(model=model_name, temperature=0.5, openai_api_key=openai_api_key)

    # Lightweight response for casual/generic greetings
    if query.strip().lower() in ["hi", "hello", "hey", "what's up?", "how are you?"]:
        return llm.predict(query)

    # Adaptive prompt - focuses on answering based on data
    template = """You are a data analyst reviewing multi-sheet inventory and financial data.

Use the following extracted data context to answer the question accurately:

{context}

Now answer this question: {question}
"""

    prompt = PromptTemplate(
        template=template,
        input_variables=["context", "question"]
    )

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        chain_type_kwargs={"prompt": prompt}
    )

    return qa_chain.invoke({"query": query})['result']
