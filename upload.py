import os
from dotenv import load_dotenv 
from azure.storage.blob import BlobServiceClient

# --- Load environment variables ---
load_dotenv()  # 👈 Add this — reads your .env automatically

AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONN_STRING")
CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER", "azureml")
FAISS_DIR = "faiss_index"

def upload_directory(local_dir: str, blob_prefix: str = ""):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    for root, _, files in os.walk(local_dir):
        for filename in files:
            local_path = os.path.join(root, filename)
            # Preserve subfolder structure (e.g., jason/index.faiss)
            relative_path = os.path.relpath(local_path, local_dir)
            blob_name = os.path.join(blob_prefix, relative_path).replace("\\", "/")

            with open(local_path, "rb") as data:
                container_client.upload_blob(name=blob_name, data=data, overwrite=True)
                print(f"✅ Uploaded: {blob_name}")

def main():
    if not AZURE_CONN_STRING:
        raise ValueError("❌ Missing Azure connection string (AZURE_STORAGE_CONN_STRING).")

    print("🚀 Uploading all FAISS indexes...")
    upload_directory(FAISS_DIR, blob_prefix="faiss_index")
    print("✅ All FAISS indexes uploaded successfully.")

if __name__ == "__main__":
    main()
