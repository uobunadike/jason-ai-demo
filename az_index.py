import os
import uuid
import pandas as pd
from dotenv import load_dotenv
from azure.search.documents import SearchClient
#from azure.search.documents.indexes.models import Vector
from azure.core.credentials import AzureKeyCredential
from langchain_openai import AzureOpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

# Load .env
load_dotenv(dotenv_path=".env", override=True)
AZURE_SEARCH_KEY = os.getenv("AZURE_SEARCH_KEY")
AZURE_SEARCH_ENDPOINT = os.getenv("AZURE_SEARCH_ENDPOINT")
AZURE_SEARCH_INDEX = os.getenv("AZURE_SEARCH_INDEX")

AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
AZURE_OPENAI_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")

FILE_NAME = "Kpi_tables.xlsx"

# Step 1: Define Excel logic
def detect_sheet_type(df: pd.DataFrame) -> str:
    columns = pd.Index([str(col).lower() for col in df.columns])
    if 'event_type' in columns and 'event_timestamp' in columns:
        return 'inventory_events'
    elif 'unit_cost' in columns and 'cogs_calculated' in columns:
        return 'financial_metrics'
    return 'unknown'

def process_sheet(df: pd.DataFrame, sheet_type: str) -> str:
    df.columns = [str(col).lower() for col in df.columns]
    text_chunks = []
    if sheet_type == 'inventory_events':
        df = df.dropna(subset=['event_type', 'sku_id'])
        for _, row in df.iterrows():
            try:
                text_chunks.append(
                    f"{row['event_type']} ({pd.to_datetime(row['event_timestamp']).strftime('%Y-%m-%d')}): "
                    f"Sold {row.get('quantity_sold', 'N/A')} units of {row.get('item_name', 'Unknown Item')} (SKU: {row['sku_id']})"
                )
            except:
                continue
    elif sheet_type == 'financial_metrics':
        df = df.dropna(subset=['sku_id', 'unit_cost'])
        for _, row in df.iterrows():
            try:
                text_chunks.append(
                    f"{row.get('item_name', 'Unknown Item')} (SKU: {row['sku_id']}): "
                    f"Unit Cost: ${float(row['unit_cost']):.2f}, COGS: ${float(row.get('cogs_calculated', 0.0)):.2f}"
                )
            except:
                continue
    return "\n".join(text_chunks)

def extract_text(file_path: str) -> list[str]:
    raw_sheets = pd.read_excel(file_path, sheet_name=None, header=None, engine="openpyxl")
    all_text = []
    for _, df in raw_sheets.items():
        header_row = None
        for idx, row in df.iterrows():
            if any(col.lower() in row.astype(str).str.lower().tolist() for col in ['event_type', 'sku_id', 'unit_cost']):
                header_row = idx
                break
        if header_row is None:
            continue
        df.columns = df.iloc[header_row]
        df = df.iloc[header_row + 1:].reset_index(drop=True)
        sheet_type = detect_sheet_type(df)
        processed = process_sheet(df, sheet_type)
        if processed:
            all_text.append(processed)
    return all_text

# Step 2: Embed + Upload
def embed_and_upload():
    print("📥 Reading Excel and extracting text...")
    raw_chunks = extract_text(FILE_NAME)

    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = []
    for block in raw_chunks:
        text_chunks.extend(splitter.split_text(block))

    print(f"✂️ Total text chunks: {len(text_chunks)}")

    print("🧠 Generating embeddings...")
    embeddings = AzureOpenAIEmbeddings(
        api_key=AZURE_OPENAI_API_KEY,
        azure_endpoint=AZURE_OPENAI_ENDPOINT,
        api_version=AZURE_OPENAI_API_VERSION,
        deployment=AZURE_OPENAI_EMBEDDING_DEPLOYMENT
    )

    search_client = SearchClient(
        endpoint=AZURE_SEARCH_ENDPOINT,
        index_name=AZURE_SEARCH_INDEX,
        credential=AzureKeyCredential(AZURE_SEARCH_KEY)
    )

    batch = []
    for chunk in text_chunks:
        vector = embeddings.embed_query(chunk)
        doc = {
            "id": str(uuid.uuid4()),
            "content": chunk,
            "embedding": vector
        }
        batch.append(doc)

    print("☁️ Uploading to Azure Cognitive Search...")
    result = search_client.upload_documents(documents=batch)
    print(f"✅ Upload complete. Uploaded {len(result)} documents.")

if __name__ == "__main__":
    embed_and_upload()
