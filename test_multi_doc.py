import os
import sys
import shutil

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from rag_engine import DocPilotEngine

def run_test():
    print("=== Testing DocPilot Multi-Document Support ===")
    sample_pdf = "sample.pdf"
    if not os.path.exists(sample_pdf):
        print(f"Sample PDF '{sample_pdf}' not found. Exiting test.")
        return

    # Use a separate test collection directory
    test_db = "./test_chroma_db"
    if os.path.exists(test_db):
        shutil.rmtree(test_db, ignore_errors=True)

    engine = DocPilotEngine(persist_directory=test_db, collection_name="test_multi_doc")

    print("\n[1] Adding Document 1 (Doc_Alpha.pdf)...")
    chunks1 = engine.add_pdf(sample_pdf, filename="Doc_Alpha.pdf")
    print(f"   Indexed {chunks1} chunks.")
    assert len(engine.get_documents()) == 1, "Expected 1 document"

    print("\n[2] Adding Document 2 (Doc_Beta.pdf)...")
    chunks2 = engine.add_pdf(sample_pdf, filename="Doc_Beta.pdf")
    print(f"   Indexed {chunks2} chunks.")
    assert len(engine.get_documents()) == 2, "Expected 2 documents"

    print("\n[3] Adding Document 3 (Doc_Gamma.pdf)...")
    chunks3 = engine.add_pdf(sample_pdf, filename="Doc_Gamma.pdf")
    print(f"   Indexed {chunks3} chunks.")
    assert len(engine.get_documents()) == 3, "Expected 3 documents"

    docs = engine.get_documents()
    print(f"   Current loaded documents: {[d['filename'] for d in docs]}")

    print("\n[4] Querying across loaded documents...")
    res = engine.query("What is discussed in these documents?")
    print(f"   Answer snippet: {res['answer'][:120]}...")
    print(f"   Retrieved sources count: {len(res.get('sources', []))}")
    for src in res.get("sources", [])[:3]:
        print(f"   - [{src.get('doc_name')}] Page {src.get('page')} (score: {src.get('relevance_score')})")
        assert "doc_name" in src, "Source missing 'doc_name'"

    print("\n[5] Removing Doc_Beta.pdf...")
    removed = engine.remove_pdf("Doc_Beta.pdf")
    assert removed, "Failed to remove Doc_Beta.pdf"
    docs_after = engine.get_documents()
    print(f"   Documents after removal: {[d['filename'] for d in docs_after]}")
    assert len(docs_after) == 2, "Expected 2 documents after removal"
    assert "Doc_Beta.pdf" not in [d["filename"] for d in docs_after], "Doc_Beta.pdf should not be present"

    print("\n[6] Testing 5-document limit...")
    engine.add_pdf(sample_pdf, filename="Doc_Delta.pdf")
    engine.add_pdf(sample_pdf, filename="Doc_Epsilon.pdf")
    engine.add_pdf(sample_pdf, filename="Doc_Zeta.pdf")
    assert len(engine.get_documents()) == 5, f"Expected 5 documents, got {len(engine.get_documents())}"
    print(f"   Currently at max capacity (5/5): {[d['filename'] for d in engine.get_documents()]}")

    limit_hit = False
    try:
        engine.add_pdf(sample_pdf, filename="Doc_OverLimit.pdf")
    except ValueError as e:
        limit_hit = True
        print(f"   Success: Caught expected limit exception: {e}")

    assert limit_hit, "Expected ValueError when exceeding 5 documents"

    print("\n[7] Cleaning up test vector store...")
    engine.clear_all()
    assert len(engine.get_documents()) == 0, "Expected 0 documents after clear_all"

    if os.path.exists(test_db):
        shutil.rmtree(test_db, ignore_errors=True)

    print("\n=== ALL MULTI-DOCUMENT TESTS PASSED SUCCESSFULLY! ===")

if __name__ == "__main__":
    run_test()
