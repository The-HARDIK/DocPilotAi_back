import os
import tempfile
import streamlit as st
from rag_engine import DocPilotEngine

st.set_page_config(
    page_title="DocPilot AI — Study Assistant",
    page_icon="📄",
    layout="wide"
)

# 1. Initialize Engine and Session States
if "engine" not in st.session_state:
    with st.spinner("Initializing Local Embeddings & Vector Database..."):
        st.session_state.engine = DocPilotEngine()

if "messages" not in st.session_state:
    st.session_state.messages = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = set()

if "study_guide" not in st.session_state:
    st.session_state.study_guide = None

# 2. Sidebar: Upload & Actions
with st.sidebar:
    st.title("📂 Document Hub")
    st.caption("Upload up to 5 PDFs to build your study corpus.")
    uploaded_files = st.file_uploader(
        "Upload PDF documents (max 5)",
        type=["pdf"],
        accept_multiple_files=True
    )

    if uploaded_files:
        if len(uploaded_files) > 5:
            st.warning("Only the first 5 PDFs will be indexed.")
            uploaded_files = uploaded_files[:5]

        current_file_names = {f.name for f in uploaded_files}
        # Ingest newly added files
        new_files = [f for f in uploaded_files if f.name not in st.session_state.indexed_files]
        if new_files:
            for f in new_files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(f.read())
                    tmp_path = tmp_file.name

                with st.spinner(f"Indexing '{f.name}' into ChromaDB..."):
                    try:
                        num_chunks = st.session_state.engine.add_pdf(tmp_path, filename=f.name)
                        st.session_state.indexed_files.add(f.name)
                        st.success(f"Indexed '{f.name}' ({num_chunks} chunks)!")
                    except Exception as e:
                        st.error(f"Error indexing '{f.name}': {e}")
                    finally:
                        if os.path.exists(tmp_path):
                            os.remove(tmp_path)

    st.markdown("---")
    loaded_docs = st.session_state.engine.get_documents()
    st.markdown(f"**Indexed Documents ({len(loaded_docs)}/5):**")
    if loaded_docs:
        for d in loaded_docs:
            st.markdown(f"• 🟢 `{d['filename']}` ({d['chunks']} chunks)")
    else:
        st.markdown("🔴 *No documents indexed*")

    if st.button("🗑 Clear All Documents", use_container_width=True):
        st.session_state.engine.clear_all()
        st.session_state.indexed_files = set()
        st.session_state.messages = []
        st.session_state.study_guide = None
        st.rerun()

    # Action: Generate Comprehensive Study Guide
    st.markdown("### 🛠 Tools")
    if st.button("📚 Generate Full Study Guide", disabled=len(loaded_docs) == 0, use_container_width=True):
        progress_bar = st.progress(0)
        status_text = st.empty()

        def update_progress(percent, text):
            progress_bar.progress(percent)
            status_text.caption(text)

        try:
            guide = st.session_state.engine.generate_study_guide(progress_callback=update_progress)
            st.session_state.study_guide = guide
            st.success("Study Guide Generated!")
        except Exception as e:
            st.error(f"Generation failed: {e}")

# 3. Main Workspace Tabs
tab_chat, tab_guide = st.tabs(["💬 Document Q&A (RAG)", "📘 Comprehensive Study Guide"])

# --- TAB 1: Conversational RAG ---
with tab_chat:
    st.caption("Ask questions strictly grounded in your uploaded documents with verifiable citations.")

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("🔍 View Retrieved Sources"):
                    for s in msg["sources"]:
                        doc_label = s.get("doc_name", "Document")
                        st.markdown(f"**{doc_label} • Page {s['page']}** (Relevance Score: `{s['relevance_score']}`)")
                        st.caption(f"\"{s['snippet']}...\"")

    if user_prompt := st.chat_input("Ask a question about your documents..."):
        if not st.session_state.engine.get_documents():
            st.warning("Please upload at least one PDF document first before asking questions.")
        else:
            st.session_state.messages.append({"role": "user", "content": user_prompt})
            with st.chat_message("user"):
                st.markdown(user_prompt)

            with st.chat_message("assistant"):
                with st.spinner("Searching document context & generating answer..."):
                    history_payload = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]]
                    result = st.session_state.engine.query(user_prompt, chat_history=history_payload)
                    st.markdown(result["answer"])
                    
                    if result.get("sources"):
                        with st.expander("🔍 View Retrieved Sources"):
                            for s in result["sources"]:
                                doc_label = s.get("doc_name", "Document")
                                st.markdown(f"**{doc_label} • Page {s['page']}** (Relevance Score: `{s['relevance_score']}`)")
                                st.caption(f"\"{s['snippet']}...\"")

            st.session_state.messages.append({
                "role": "assistant",
                "content": result["answer"],
                "sources": result.get("sources", [])
            })

# --- TAB 2: Study Guide Viewer ---
with tab_guide:
    if st.session_state.study_guide:
        st.download_button(
            label="📥 Download Study Guide (.md)",
            data=st.session_state.study_guide,
            file_name="Study_Guide_Master.md",
            mime="text/markdown",
            use_container_width=True
        )
        st.markdown(st.session_state.study_guide)
    else:
        st.info("No study guide generated yet. Upload PDF(s) and click **'📚 Generate Full Study Guide'** in the sidebar to create one.")