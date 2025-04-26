import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import AzureOpenAIEmbeddings

# Load env vars
load_dotenv()

# Set up embedding
embedding = AzureOpenAIEmbeddings(
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT")
)

# Load Chroma vector store
db = Chroma(persist_directory="./chroma_db", embedding_function=embedding)

# Run test similarity search
docs = db.similarity_search("test", k=3)

# Print types and content of retrieved documents
for i, d in enumerate(docs):
    print(f"📄 Doc {i}: type={type(d.page_content)}, content={d.page_content}")