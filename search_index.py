# index_documents.py
import os
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores.azuresearch import AzureSearch
from langchain_openai import AzureOpenAIEmbeddings
import pandas as pd

load_dotenv()

def load_kpi_data(file_name: str = "Kpi_tables.xlsx") -> str:
    df = pd.read_excel(file_name, sheet_name=None)
    text_chunks = []
    for sheet_name, sheet_df in df.items():
        text_chunks.append(sheet_df.to_string(index=False))
    return "\n\n".join(text_chunks)

def index_to_azure_search():
    print("📥 Loading and splitting data...")
    raw_text = load_kpi_data()
    splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = splitter.create_documents([raw_text])

    print(f"📄 Prepared {len(docs)} documents. Uploading to Azure Search...")

    embeddings = AzureOpenAIEmbeddings(
        deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
        openai_api_type="azure",
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION")
    )

    AzureSearch.from_documents(
        documents=docs,
        embedding=embeddings,
        index_name=os.getenv("AZURE_SEARCH_INDEX"),
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=os.getenv("AZURE_SEARCH_KEY"),
    )

    print("✅ Upload complete. Your index now contains documents.")

if __name__ == "__main__":
    index_to_azure_search()
