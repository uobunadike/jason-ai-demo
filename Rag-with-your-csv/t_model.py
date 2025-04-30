from model3 import run
from dotenv import load_dotenv
import os

load_dotenv()

print("🧪 Testing AI Inventory Assistant (Chroma + Azure OpenAI)")
query = input("Enter your question: ")
try:
    print("🔍 Response:\n")
    print(run(query))  # 👈 This is passing only `query` — perfect
except Exception as e:
    print(f"❌ Error during model run:\n{e}")

