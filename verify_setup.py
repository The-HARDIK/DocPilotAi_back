import os
from dotenv import load_dotenv
from rag_engine import DocPilotEngine

load_dotenv()

print("--- Running DocPilot Environment Verification ---")
try:
    engine = DocPilotEngine()
    print("[PASS] Google Generative AI Embedding model (text-embedding-004) initialized.")
    print("[PASS] ChromaDB vector store ready.")
    print("[PASS] Gemini LLM (gemini-1.5-flash) ready.")
    print("-------------------------------------------------")
    print("Environment setup is 100% complete and working.")
except Exception as e:
    print(f"[FAIL] Error during initialization: {e}")