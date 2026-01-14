from model import run, run_legacy
from dotenv import load_dotenv
import json

load_dotenv()

print("[TEST] Multi-Persona AI Assistant (FAISS + Azure OpenAI)")
print("-" * 50)

persona = input("Which persona? (jason/claire): ").strip().lower()
query = input("Enter your question: ")

print("\n[INFO] Running query...")

try:
    # Test structured response
    print("\n--- STRUCTURED RESPONSE ---")
    result = run(query, persona=persona)
    print(json.dumps(result, indent=2))
    
    # Test legacy response
    print("\n--- LEGACY RESPONSE ---")
    legacy_result = run_legacy(query, persona=persona)
    print(legacy_result)
    
except Exception as e:
    print(f"\n[ERROR] {e}")
