import os
import sys
from rag_engine import DocPilotEngine

print("=" * 60)
print("DocPilot AI: End-to-End Pipeline Test")
print("=" * 60)

try:
    print("[1/4] Initializing RAG Engine...")
    engine = DocPilotEngine()
    print("[✓] RAG Engine and Local Embeddings loaded successfully.\n")
except Exception as e:
    print(f"[✗] Failed to initialize engine: {e}")
    sys.exit(1)

sample_pdf = "sample.pdf"
if not os.path.exists(sample_pdf):
    print(f"[!] File not found: '{sample_pdf}'")
    print("    Please place a text-based PDF named 'sample.pdf' inside the backend folder and re-run.")
    sys.exit(0)

try:
    print(f"[2/4] Parsing and vectorizing '{sample_pdf}'...")
    chunks_created = engine.ingest_pdf(sample_pdf)
    print(f"[✓] Successfully indexed {chunks_created} text chunks into ChromaDB.\n")
except Exception as e:
    print(f"[✗] Failed during PDF ingestion: {e}")
    sys.exit(1)

test_query = "What are the core concepts or main topics covered in this document?"
print(f"[3/4] Querying: '{test_query}'")
print("[-] Retrieving relevant chunks and querying Groq LLM...\n")

try:
    result = engine.query(test_query)
    print("=" * 25 + " AI RESPONSE " + "=" * 25)
    print(result["answer"])
    print("\n" + "=" * 25 + " RETRIEVED SOURCES " + "=" * 25)
    for src in result.get("sources", []):
        print(f"• Chunk {src['chunk_id']} | Page {src['page']} (Relevance Score: {src['relevance_score']}):")
        print(f"  \"{src['snippet']}...\"\n")
    print("[✓] Pipeline execution finished successfully.")
except Exception as e:
    print(f"[✗] Failed during query execution: {e}")