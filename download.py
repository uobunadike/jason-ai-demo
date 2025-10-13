import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONN_STRING")
CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER", "azureml")
FAISS_DIR = "faiss_index"

def download_directory(blob_prefix: str, local_dir: str):
    blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STRING)
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)
    os.makedirs(local_dir, exist_ok=True)

    print(f"⬇️  Downloading from prefix '{blob_prefix}' into '{local_dir}'...")

    blobs = container_client.list_blobs(name_starts_with=blob_prefix)
    found_any = False

    for blob in blobs:
        found_any = True
        relative_path = os.path.relpath(blob.name, blob_prefix)
        local_path = os.path.join(local_dir, relative_path)
        os.makedirs(os.path.dirname(local_path), exist_ok=True)

        with open(local_path, "wb") as f:
            blob_data = container_client.download_blob(blob.name)
            f.write(blob_data.readall())
        print(f"✅ Downloaded: {blob.name}")

    if not found_any:
        print(f"⚠️ No blobs found under prefix '{blob_prefix}'")

def main():
    if not AZURE_CONN_STRING:
        raise ValueError("❌ Missing Azure connection string (AZURE_STORAGE_CONN_STRING).")

    print("🚀 Downloading all FAISS indexes from Azure Blob...")
    download_directory("faiss_index", FAISS_DIR)
    print("✅ All FAISS indexes downloaded successfully.")

if __name__ == "__main__":
    main()
