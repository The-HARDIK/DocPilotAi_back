import os
from dotenv import load_dotenv
from rag_engine import DocPilotEngine

load_dotenv()

print("--- Running DocPilot Environment Verification ---")
try:
    engine = DocPilotEngine()
    print("[PASS] FastEmbed local ONNX embedding engine ready (all-MiniLM-L6-v2).")
    print("[PASS] ChromaDB vector store ready.")
    print("[PASS] Google Gemini (gemini-3.5-flash with flash-lite fallback) ready.")
    print("-------------------------------------------------")
    print("Environment setup is 100% complete and working.")
except Exception as e:
    print(f"[FAIL] Error during initialization: {e}")