import os
import pandas as pd
from typing import Literal
from functools import lru_cache
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# LangChain and model imports
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain.embeddings import AzureOpenAIEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain_community.chat_models import AzureChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


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
def load_kpi_data(file_name: str = os.getenv("AZURE_BLOB_NAME", "Kpi_tables.xlsx")) -> str:
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
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50, separator="\n")
    docs = text_splitter.split_text(load_kpi_data())
    
    embedding = AzureOpenAIEmbeddings(
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
    openai_api_type="azure",
    openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
)

    return AzureSearch(
    endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
    key=os.getenv("AZURE_SEARCH_KEY"),
    index_name=os.getenv("AZURE_SEARCH_INDEX"),
    embedding_function=embedding.embed_query
)

def run(query: str, model_type: Literal["ollama", "openai"] = "ollama", model_name: str = "llama3.1") -> str:
    db = get_vector_store()

    if model_type == "ollama":
        llm = OllamaLLM(model=model_name, temperature=0.3)
    else:
        llm = AzureChatOpenAI(
            deployment_name=os.getenv("AZURE_OPENAI_CHAT_DEPLOYMENT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            openai_api_base=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_type="azure",
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            temperature=0.5
        )

    if query.strip().lower() in ["hi", "hello", "hey", "what's up?", "how are you?"]:
        return llm.predict(query)

    template = """You are an AI inventory insights assistant.

Based on the extracted data, answer the question concisely using the format below:

- **Summary**: 1-2 sentence answer to the question.
- **Root Cause**: Short explanation if there’s a risk or issue.
- **Business Impact**: Estimated dollar or operational impact (if applicable).
- **Recommendation**: Actionable next step (if possible).

Context:
{context}

Question: {question}
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
