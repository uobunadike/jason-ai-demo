import os
from azure.storage.blob import BlobServiceClient

# These should be set in your environment or GitHub secrets
AZURE_CONN_STRING = os.getenv("AZURE_STORAGE_CONN_STRING")
CONTAINER_NAME = os.getenv("AZURE_BLOB_CONTAINER", "azureml")

FAISS_DIR = "src/faiss_index"
EXCEL_DIR = "src"
FILES_TO_DOWNLOAD = {
    "index.faiss": FAISS_DIR,
    "index.pkl": FAISS_DIR,
    "revenue_at_risk.xlsx": EXCEL_DIR,
}

def download_blobs():
    try:
        if not AZURE_CONN_STRING:
            raise ValueError("Missing Azure connection string environment variable.")

        blob_service_client = BlobServiceClient.from_connection_string(AZURE_CONN_STRING)
        container_client = blob_service_client.get_container_client(CONTAINER_NAME)

        for filename, target_dir in FILES_TO_DOWNLOAD.items():
            os.makedirs(target_dir, exist_ok=True)
            blob_client = container_client.get_blob_client(filename)
            download_path = os.path.join(target_dir, filename)

            with open(download_path, "wb") as f:
                f.write(blob_client.download_blob().readall())
            print(f"✅ Downloaded {filename} to {download_path}")

    except Exception as e:
        print(f"❌ Download failed: {str(e)}")

if __name__ == "__main__":
    download_blobs()
