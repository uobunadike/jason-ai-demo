from model import run
from dotenv import load_dotenv
import os

load_dotenv()

print("🧪 Testing Multi-Persona AI Assistant (FAISS + Azure OpenAI)")

persona = input("Which persona do you want to test? (jason/claire): ").strip().lower()
query = input("Enter your question: ")

try:
    print("\n🔍 Response:\n")
    print(run(query, persona=persona))  # 👈 Now passes the persona parameter
except Exception as e:
    print(f"\n❌ Error during model run:\n{e}")