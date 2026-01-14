# rebuild_index.py - Rebuild FAISS index from CSV files
import os
from dotenv import load_dotenv
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.schema import Document
import pandas as pd

load_dotenv()

# Azure OpenAI configs
api_key = os.getenv("AZURE_OPENAI_API_KEY")
endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
embedding_deployment = os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
api_version = os.getenv("AZURE_OPENAI_API_VERSION")

def rebuild_faiss_index(persona: str):
    """Rebuild FAISS index from CSV files for a persona."""
    data_dir = os.path.join("data", persona)
    faiss_dir = os.path.join("faiss_index", persona)
    
    print(f"[INFO] Rebuilding FAISS index for {persona}...")
    print(f"[INFO] Data directory: {data_dir}")
    print(f"[INFO] FAISS directory: {faiss_dir}")
    
    # Load all CSV files
    documents = []
    for filename in os.listdir(data_dir):
        if filename.endswith(".csv"):
            filepath = os.path.join(data_dir, filename)
            print(f"[INFO] Loading {filename}...")
            
            try:
                df = pd.read_csv(filepath)
                # Convert each row to a document
                for idx, row in df.iterrows():
                    # Create text from row data
                    row_text = " | ".join([f"{col}: {val}" for col, val in row.items() if pd.notna(val)])
                    documents.append(Document(
                        page_content=row_text,
                        metadata={"source": filename, "row": idx}
                    ))
            except Exception as e:
                print(f"[WARN] Error loading {filename}: {e}")
    
    print(f"[INFO] Loaded {len(documents)} documents")
    
    # Show sample documents
    print("\n[INFO] Sample documents:")
    for doc in documents[:5]:
        print(f"  - {doc.page_content[:100]}...")
    
    # Check for Low Stock and Damaged
    print("\n[INFO] Checking for Low Stock/Damaged items:")
    for doc in documents:
        if "Low Stock" in doc.page_content or "Damaged" in doc.page_content:
            print(f"  - {doc.page_content}")
    
    # Create embeddings
    embeddings = AzureOpenAIEmbeddings(
        api_key=api_key,
        azure_endpoint=endpoint,
        deployment=embedding_deployment,
        api_version=api_version
    )
    
    # Build FAISS index
    print("\n[INFO] Building FAISS index...")
    db = FAISS.from_documents(documents, embedding=embeddings)
    
    # Save index
    os.makedirs(faiss_dir, exist_ok=True)
    db.save_local(faiss_dir)
    print(f"[OK] FAISS index saved to {faiss_dir}")
    
    return db


if __name__ == "__main__":
    import sys
    
    persona = sys.argv[1] if len(sys.argv) > 1 else "jason"
    rebuild_faiss_index(persona)
    
    print("\n[DONE] Now run: python upload.py")

