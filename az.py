import os
from dotenv import load_dotenv
from azure.storage.blob import BlobServiceClient

load_dotenv()

conn_str = os.getenv("AZURE_STORAGE_CONN_STRING")
container = os.getenv("AZURE_BLOB_CONTAINER")
blob_name = os.getenv("AZURE_BLOB_NAME")

local_file = "Kpi_tables.xlsx"  # Match actual blob filename

blob_service = BlobServiceClient.from_connection_string(conn_str)
blob_client = blob_service.get_blob_client(container=container, blob=blob_name)

with open(local_file, "wb") as f:
    f.write(blob_client.download_blob().readall())

print(f"✅ Excel downloaded to: {local_file}")
