from model import run

from dotenv import load_dotenv
import os

load_dotenv()  # make sure it's loading first
print("✅ Using Azure Endpoint:", os.getenv("AZURE_OPENAI_ENDPOINT"))
print("✅ Using Embedding Deployment:", os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"))
print("✅ Using API Key:", os.getenv("OPENAI_API_KEY")[:6] + "..." + os.getenv("OPENAI_API_KEY")[-4:])

print("🧪 Testing AI Inventory Assistant (Chroma + Azure OpenAI)")
query = input("Enter your question: ")
try:
    print("🔍 Response:\n")
    print(run(query))
except Exception as e:
    print(f"❌ Error during model run:\n{e}")

