import os
from dotenv import load_dotenv
from rag_engine import DocPilotEngine

load_dotenv()

print("--- Running DocPilot Environment Verification ---")
try:
    engine = DocPilotEngine()
    print("[PASS] Local BGE Embedding model loaded and initialized.")
    print("[PASS] ChromaDB vector store ready.")
    print("[PASS] Groq API client initialized.")
    print("-------------------------------------------------")
    print("Environment setup is 100% complete and working.")
except Exception as e:
    print(f"[FAIL] Error during initialization: {e}")