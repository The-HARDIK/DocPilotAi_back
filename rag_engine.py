import os
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma

load_dotenv()

class DocPilotEngine:
    MAX_DOCUMENTS = 5

    def __init__(self, persist_directory: str = "./chroma_db", collection_name: str = "docpilot_collection"):
        self.persist_directory = os.path.abspath(persist_directory)
        self.collection_name = collection_name
        
        self.api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("Missing GEMINI_API_KEY in .env file. Get one for free at https://aistudio.google.com/app/apikey")
        
        self.doc_registry: Dict[str, Dict[str, Any]] = {}
        self.raw_documents_map: Dict[str, List[Any]] = {}
        
        # Google Generative AI Embeddings (Free tier: models/text-embedding-004)
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=self.api_key
        )
        
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        self.default_model = "gemini-1.5-flash"

    def _get_llm(self, model: Optional[str] = None, temperature: float = 0.1, max_tokens: Optional[int] = None) -> ChatGoogleGenerativeAI:
        chosen_model = model or self.default_model
        # Map any legacy model names or default to gemini-1.5-flash
        if not chosen_model.startswith("gemini"):
            chosen_model = "gemini-1.5-flash"
        return ChatGoogleGenerativeAI(
            model=chosen_model,
            google_api_key=self.api_key,
            temperature=temperature,
            max_output_tokens=max_tokens
        )

    @property
    def raw_documents(self) -> List[Any]:
        """Aggregates all loaded pages across all active documents."""
        all_pages = []
        for pages in self.raw_documents_map.values():
            all_pages.extend(pages)
        return all_pages

    @raw_documents.setter
    def raw_documents(self, docs: List[Any]):
        self.raw_documents_map["default"] = docs

    def get_documents(self) -> List[Dict[str, Any]]:
        """Returns metadata for all currently indexed documents."""
        return [
            {
                "filename": info["filename"],
                "chunks": info["chunks"],
                "pages": info["pages"],
                "filepath": info["filepath"]
            }
            for info in self.doc_registry.values()
        ]

    def remove_pdf(self, filename: str) -> bool:
        """Removes an indexed PDF and its chunks from ChromaDB."""
        if filename not in self.doc_registry:
            return False

        info = self.doc_registry[filename]
        chunk_ids = info.get("chunk_ids", [])
        if chunk_ids:
            try:
                self.vector_store.delete(ids=chunk_ids)
            except Exception:
                try:
                    self.vector_store._collection.delete(ids=chunk_ids)
                except Exception:
                    pass

        del self.doc_registry[filename]
        if filename in self.raw_documents_map:
            del self.raw_documents_map[filename]
        return True

    def clear_all(self):
        """Removes all indexed documents and resets vector store."""
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        self.doc_registry.clear()
        self.raw_documents_map.clear()

    def add_pdf(self, pdf_path: str, filename: Optional[str] = None) -> int:
        """Loads, chunks, and adds a PDF document to ChromaDB (up to MAX_DOCUMENTS)."""
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        doc_name = filename or os.path.basename(pdf_path)

        if doc_name not in self.doc_registry and len(self.doc_registry) >= self.MAX_DOCUMENTS:
            raise ValueError(f"Maximum {self.MAX_DOCUMENTS} documents allowed. Please remove a document first.")

        # If already exists, remove previous chunks first to replace cleanly
        if doc_name in self.doc_registry:
            self.remove_pdf(doc_name)

        loader = PyPDFLoader(pdf_path)
        pages = loader.load()

        total_text_length = sum(len(d.page_content.strip()) for d in pages)
        if total_text_length == 0:
            raise ValueError(f"No extractable text found in '{doc_name}'.")

        # Tag pages with document name
        for page in pages:
            page.metadata["doc_name"] = doc_name
            page.metadata["source"] = pdf_path

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        chunks = text_splitter.split_documents(pages)

        # Generate unique sanitized chunk IDs
        safe_prefix = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in doc_name)
        chunk_ids = [f"{safe_prefix}__chunk_{i}" for i in range(len(chunks))]

        for chunk in chunks:
            chunk.metadata["doc_name"] = doc_name
            chunk.metadata["source"] = pdf_path

        self.vector_store.add_documents(documents=chunks, ids=chunk_ids)

        self.doc_registry[doc_name] = {
            "filename": doc_name,
            "filepath": pdf_path,
            "chunks": len(chunks),
            "pages": len(pages),
            "chunk_ids": chunk_ids
        }
        self.raw_documents_map[doc_name] = pages

        return len(chunks)

    def ingest_pdf(self, pdf_path: str, filename: Optional[str] = None, clear_existing: bool = False) -> int:
        """Loads and indexes a PDF document. If clear_existing is True, resets existing docs."""
        if clear_existing:
            self.clear_all()
        return self.add_pdf(pdf_path, filename=filename)

    def _contextualize_query(self, question: str, chat_history: List[Dict[str, str]], model: str) -> str:
        """Reformulates follow-up questions into standalone search queries using past turns."""
        if not chat_history:
            return question

        history_summary = "\n".join([f"{msg['role']}: {msg['content']}" for msg in chat_history[-4:]])
        
        rephrase_prompt = (
            "Given the chat history and the latest user question, rephrase the question to be a self-contained "
            "search query that captures full context. Do NOT answer the question, only return the rephrased query.\n\n"
            f"Chat History:\n{history_summary}\n\n"
            f"Latest Question: {question}\n"
            "Standalone Query:"
        )

        try:
            llm = self._get_llm(model=model, temperature=0.0, max_tokens=120)
            resp = llm.invoke(rephrase_prompt)
            return resp.content.strip()
        except Exception:
            return question

    def query(
        self, 
        question: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 8, 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Performs contextual retrieval and multi-turn response generation."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return {"answer": "Please provide a valid question.", "sources": []}

        collection_count = self.vector_store._collection.count()
        if collection_count == 0:
            return {"answer": "No documents are currently indexed.", "sources": []}

        selected_model = model or self.default_model

        search_query = cleaned_question
        if chat_history and len(chat_history) > 0:
            search_query = self._contextualize_query(cleaned_question, chat_history, selected_model)

        retrieved_results = self.vector_store.similarity_search_with_relevance_scores(
            search_query, 
            k=min(top_k, collection_count)
        )

        formatted_context_list = []
        sources: List[Dict[str, Any]] = []

        for i, (doc, score) in enumerate(retrieved_results):
            page_num = doc.metadata.get("page", 0) + 1
            doc_name = doc.metadata.get("doc_name") or os.path.basename(doc.metadata.get("source", "Document"))
            content = doc.page_content.strip()
            
            sources.append({
                "chunk_id": i + 1,
                "doc_name": doc_name,
                "page": page_num,
                "relevance_score": round(float(score), 3) if score is not None else 0.0,
                "snippet": content[:140]
            })
            
            formatted_context_list.append(f"[Chunk {i+1} | Document: {doc_name} | Page {page_num}]\n{content}")

        joined_context = "\n\n".join(formatted_context_list)

        messages = [
            SystemMessage(
                content=(
                    "You are DocPilot AI, an expert academic tutor and technical assistant.\n"
                    "Rules:\n"
                    "1. Explain the concepts thoroughly, clearly, and in plain language using the provided context.\n"
                    "2. When comparing items or listing registers/instructions, format them into neat Markdown tables.\n"
                    "3. Cite specific Document names and Page numbers (e.g. [Document.pdf, Page X]).\n"
                    "4. When information comes from different uploaded documents, synthesize or contrast them clearly.\n"
                    "5. If a specific detail is entirely absent from the text, state what is missing rather than giving a generic refusal."
                )
            )
        ]

        if chat_history:
            for turn in chat_history[-4:]:
                if turn.get("role") == "user":
                    messages.append(HumanMessage(content=turn.get("content", "")))
                else:
                    messages.append(AIMessage(content=turn.get("content", "")))

        messages.append(
            HumanMessage(content=f"Context:\n{joined_context}\n\nQuestion: {cleaned_question}")
        )

        try:
            llm = self._get_llm(model=selected_model, temperature=0.1, max_tokens=1500)
            resp = llm.invoke(messages)
            answer = resp.content
        except Exception as e:
            answer = f"Error communicating with Gemini API: {str(e)}"

        return {
            "answer": answer,
            "sources": sources,
            "standalone_query": search_query
        }

    def generate_study_guide(self, model: Optional[str] = None, progress_callback=None) -> str:
        """Map-reduce summarization with dynamic model selection and token safety across documents."""
        if not self.raw_documents:
            raise ValueError("No documents are currently loaded. Please upload at least one PDF first.")

        selected_model = model or self.default_model
        total_pages = len(self.raw_documents)
        page_summaries = []

        batch_size = 4
        batches = [self.raw_documents[i:i + batch_size] for i in range(0, total_pages, batch_size)]

        llm_map = self._get_llm(model=selected_model, temperature=0.1, max_tokens=350)

        for idx, batch in enumerate(batches):
            if progress_callback:
                first_doc = batch[0].metadata.get("doc_name", "Doc")
                first_page = batch[0].metadata.get("page", 0) + 1
                last_page = batch[-1].metadata.get("page", 0) + 1
                progress_callback(
                    (idx + 1) / (len(batches) + 1),
                    f"Analyzing {first_doc} (Pages {first_page} to {last_page})..."
                )

            batch_text = ""
            for doc in batch:
                p_num = doc.metadata.get("page", 0) + 1
                d_name = doc.metadata.get("doc_name") or os.path.basename(doc.metadata.get("source", "Document"))
                clean_text = doc.page_content.strip()[:1800]
                batch_text += f"\n--- [{d_name}] Page {p_num} ---\n{clean_text}"

            map_prompt = (
                "Extract and summarize key technical concepts, registers, memory layouts, "
                "hardware mechanics, or assembly instructions in concise bullet points with document name and page citations.\n\n"
                f"{batch_text}\n\n"
                "Technical Summary (max 200 words):"
            )

            try:
                resp = llm_map.invoke(map_prompt)
                page_summaries.append(resp.content.strip())
            except Exception as e:
                page_summaries.append(f"Section Summary: {str(e)}")

            time.sleep(0.3)

        if progress_callback:
            progress_callback(0.95, "Synthesizing master study guide...")

        combined_summaries = "\n\n".join(page_summaries)
        if len(combined_summaries) > 10000:
            combined_summaries = combined_summaries[:10000]

        reduce_prompt = (
            "You are an expert academic professor. Using the following section summaries from the provided technical documents, "
            "create a structured Master Study Guide synthesizing key concepts across all documents.\n\n"
            "Format your guide cleanly in Markdown with standard tables:\n"
            "# 📘 Comprehensive Study Guide & Exam Prep\n"
            "## 1. Executive Overview\n"
            "## 2. Core Technical Breakdown (with syntax, examples, and [Document, Page X] citations)\n"
            "## 3. Key Registers & Memory Map Tables\n"
            "## 4. High-Yield Exam Points\n\n"
            f"Source Section Summaries:\n{combined_summaries}"
        )

        try:
            llm_reduce = self._get_llm(model=selected_model, temperature=0.1, max_tokens=1800)
            final_resp = llm_reduce.invoke(reduce_prompt)
            if progress_callback:
                progress_callback(1.0, "Study Guide Ready!")
            return final_resp.content
        except Exception as e:
            return f"Error synthesizing final study guide: {str(e)}"