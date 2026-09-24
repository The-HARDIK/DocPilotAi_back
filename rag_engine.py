import os
import re
import time
from typing import Dict, Any, List, Optional
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings.fastembed import FastEmbedEmbeddings
try:
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    ChatGoogleGenerativeAI = None
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_chroma import Chroma

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

def _extract_text(content: Any) -> str:
    """Extracts clean string text from Gemini response blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        texts = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                texts.append(part["text"])
            elif isinstance(part, str):
                texts.append(part)
        return "".join(texts)
    return str(content)

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
        
        # High-speed local ONNX embeddings (Instantaneous indexing, 0 rate limits, ~50MB RAM)
        self.embeddings = FastEmbedEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        # ChromaDB Cloud or Local configuration
        self.chroma_api_key = os.getenv("CHROMA_API_KEY")
        self.chroma_tenant = os.getenv("CHROMA_TENANT")
        self.chroma_database = os.getenv("CHROMA_DATABASE", "osproject")
        self.chroma_collection = os.getenv("CHROMA_COLLECTION", "odpdf")
        self.cloud_client = None

        if self.chroma_api_key and self.chroma_tenant:
            try:
                import chromadb
                self.cloud_client = chromadb.CloudClient(
                    api_key=self.chroma_api_key,
                    tenant=self.chroma_tenant,
                    database=self.chroma_database
                )
                self.vector_store = Chroma(
                    client=self.cloud_client,
                    collection_name=self.chroma_collection,
                    embedding_function=self.embeddings
                )
                print(f"[DocPilot] Connected to Chroma Cloud (Tenant: {self.chroma_tenant[:8]}..., Database: {self.chroma_database}, Collection: {self.chroma_collection}).")
            except Exception as ce:
                print(f"[DocPilot] Warning: Could not connect to Chroma Cloud ({ce}). Falling back to local ChromaDB.")
                self.vector_store = self._init_local_chroma()
        else:
            self.vector_store = self._init_local_chroma()

        self.default_model = os.getenv("GEMINI_MODEL", "models/gemini-3.5-flash-lite")
        self.fallback_model = "models/gemini-3.6-flash"
        self._cached_study_guide: Dict[str, str] = {}

    def _init_local_chroma(self) -> Chroma:
        """Initializes a clean local Chroma vector store instance."""
        try:
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )
        except Exception:
            try:
                import shutil
                if os.path.exists(self.persist_directory):
                    shutil.rmtree(self.persist_directory)
            except Exception:
                pass
            return Chroma(
                collection_name=self.collection_name,
                embedding_function=self.embeddings,
                persist_directory=self.persist_directory
            )

    def ensure_default_os_knowledge(self) -> bool:
        """Loads and indexes the core Operating Systems knowledge guide into ChromaDB if not already present."""
        builtin_name = "Operating_Systems_Core_Guide.pdf"
        if builtin_name in self.doc_registry:
            return True

        default_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "default_docs")
        pdf_path = os.path.join(default_dir, "operating_systems_core_guide.pdf")

        if not os.path.isfile(pdf_path):
            try:
                from default_docs.generate_os_reference import create_os_guide
                create_os_guide(pdf_path)
            except Exception as e:
                print(f"[DocPilot] Could not generate OS guide PDF: {e}")
                return False

        if os.path.isfile(pdf_path):
            try:
                self.add_pdf(pdf_path, filename=builtin_name, is_builtin=True)
                print(f"[DocPilot] Default OS knowledge base indexed successfully ({self.doc_registry[builtin_name]['chunks']} chunks).")
                return True
            except Exception as e:
                print(f"[DocPilot] Error indexing default OS guide: {e}")
                return False
        return False

    def _get_llm(self, model: Optional[str] = None, max_tokens: Optional[int] = 4096) -> ChatGoogleGenerativeAI:
        chosen_model = model or self.default_model
        if not chosen_model.startswith("models/"):
            chosen_model = f"models/{chosen_model}" if "gemini" in chosen_model else self.default_model

        # Map experimental or legacy aliases gracefully to active production endpoints
        model_aliases = {
            "models/gemini-3.8-flash-latest": "models/gemini-3.6-flash",
            "models/gemini-3.6-flash": "models/gemini-3.6-flash",
            "models/gemini-3.5-flash": "models/gemini-3.5-flash-lite",
            "models/gemini-3.5-flash-lite": "models/gemini-3.5-flash-lite",
            "models/gemini-flash-latest": "models/gemini-3.5-flash-lite",
            "models/gemini-flash-lite-latest": "models/gemini-3.5-flash-lite",
            "models/gemini-2.5-flash": "models/gemini-3.5-flash-lite",
            "models/gemini-2.5-flash-lite": "models/gemini-3.5-flash-lite",
        }
        chosen_model = model_aliases.get(chosen_model, chosen_model)

        if ChatGoogleGenerativeAI is not None:
            return ChatGoogleGenerativeAI(
                model=chosen_model,
                google_api_key=self.api_key,
                max_output_tokens=max_tokens
            )
        class SimpleLLM:
            def __init__(self, m):
                self.model = m
        return SimpleLLM(chosen_model)

    def _invoke_llm_with_retry(self, llm_or_model: Any, prompt_or_messages: Any, max_tokens: int = 4096, max_retries: int = 3) -> str:
        """Invokes Gemini LLM directly using google-genai SDK for ultra-fast (2-5s) inference."""
        from google import genai
        from google.genai import types

        model_name = self.default_model
        if isinstance(llm_or_model, str):
            model_name = llm_or_model
        elif hasattr(llm_or_model, "model"):
            model_name = getattr(llm_or_model, "model", self.default_model)

        clean_model = model_name.replace("models/", "")

        system_text = ""
        contents = []
        if isinstance(prompt_or_messages, list):
            for msg in prompt_or_messages:
                if isinstance(msg, SystemMessage):
                    system_text += msg.content + "\n"
                elif isinstance(msg, HumanMessage):
                    contents.append(f"{msg.content}")
                elif isinstance(msg, AIMessage):
                    contents.append(f"Assistant: {msg.content}")
                elif isinstance(msg, dict):
                    role = msg.get("role", "user")
                    content = msg.get("content", "")
                    contents.append(f"{role.capitalize()}: {content}")
                else:
                    contents.append(str(msg))
            full_content = "\n\n".join(contents)
        else:
            full_content = str(prompt_or_messages)

        client = genai.Client(api_key=self.api_key)
        config = types.GenerateContentConfig(
            max_output_tokens=max_tokens,
            system_instruction=system_text.strip() if system_text else None,
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        )

        current_model = clean_model
        last_error = None

        for attempt in range(max_retries):
            try:
                resp = client.models.generate_content(
                    model=current_model,
                    contents=full_content,
                    config=config
                )
                text = resp.text or ""
                if text.strip():
                    return text
            except Exception as e:
                last_error = e
                err_str = str(e)
                if ("429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "404" in err_str or "503" in err_str or "UNAVAILABLE" in err_str or "high demand" in err_str.lower()):
                    fallback_clean = self.fallback_model.replace("models/", "")
                    if current_model != fallback_clean:
                        print(f"[DocPilot] Model {current_model} busy/unavailable ({err_str[:60]}). Switching to fallback: {fallback_clean}")
                        current_model = fallback_clean
                        continue
                    delay = 3.0
                    match = re.search(r"retry in ([\d\.]+)s", err_str)
                    if match:
                        delay = float(match.group(1)) + 1.0
                    print(f"[DocPilot] Gemini service busy. Waiting {delay:.1f}s before retry (attempt {attempt + 1}/{max_retries})...")
                    time.sleep(delay)
                else:
                    raise e
        if last_error:
            raise last_error
        return "No answer could be generated. Please try asking in a slightly different way."

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

    def get_documents(self, include_builtin: bool = False) -> List[Dict[str, Any]]:
        """Returns metadata for currently indexed documents (excluding built-in docs unless include_builtin is True)."""
        return [
            {
                "filename": info["filename"],
                "chunks": info["chunks"],
                "pages": info["pages"],
                "filepath": info["filepath"],
                "is_builtin": info.get("is_builtin", False)
            }
            for info in self.doc_registry.values()
            if include_builtin or not info.get("is_builtin", False)
        ]

    def has_builtin_knowledge(self) -> bool:
        """Returns True if the default Operating Systems guide is indexed."""
        return "Operating_Systems_Core_Guide.pdf" in self.doc_registry

    def get_builtin_document(self) -> Optional[Dict[str, Any]]:
        """Returns metadata for the default Operating Systems guide if indexed."""
        return self.doc_registry.get("Operating_Systems_Core_Guide.pdf")

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
        self._cached_study_guide.clear()
        return True

    def clear_all(self, preserve_builtin: bool = True):
        """Removes all indexed documents and resets vector store. Preserves default OS guide by default."""
        try:
            self.vector_store.delete_collection()
        except Exception:
            pass

        if self.cloud_client:
            try:
                self.vector_store = Chroma(
                    client=self.cloud_client,
                    collection_name=self.chroma_collection,
                    embedding_function=self.embeddings
                )
            except Exception:
                self.vector_store = self._init_local_chroma()
        else:
            self.vector_store = self._init_local_chroma()

        self.doc_registry.clear()
        self.raw_documents_map.clear()
        self._cached_study_guide.clear()

        if preserve_builtin:
            self.ensure_default_os_knowledge()

    def add_pdf(self, pdf_path: str, filename: Optional[str] = None, is_builtin: bool = False) -> int:
        """Loads, chunks, and indexes a PDF document into ChromaDB in seconds."""
        if not os.path.isfile(pdf_path):
            raise FileNotFoundError(f"File not found: {pdf_path}")

        doc_name = filename or os.path.basename(pdf_path)

        if not is_builtin:
            user_doc_count = len([d for d in self.doc_registry.values() if not d.get("is_builtin", False)])
            if doc_name not in self.doc_registry and user_doc_count >= self.MAX_DOCUMENTS:
                raise ValueError(f"Maximum {self.MAX_DOCUMENTS} documents allowed. Please remove a document first.")

        # If already exists, replace cleanly
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
            chunk_size=1200,
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

        # FastEmbed runs on local ONNX: embeds all chunks in 1-3 seconds with zero rate limits
        self.vector_store.add_documents(documents=chunks, ids=chunk_ids)

        self.doc_registry[doc_name] = {
            "filename": doc_name,
            "filepath": pdf_path,
            "chunks": len(chunks),
            "pages": len(pages),
            "chunk_ids": chunk_ids,
            "is_builtin": is_builtin
        }
        self.raw_documents_map[doc_name] = pages
        self._cached_study_guide.clear()

        return len(chunks)

    def ingest_pdf(self, pdf_path: str, filename: Optional[str] = None, clear_existing: bool = False) -> int:
        """Loads and indexes a PDF document. If clear_existing is True, resets existing docs."""
        if clear_existing:
            self.clear_all(preserve_builtin=True)
        return self.add_pdf(pdf_path, filename=filename)

    def _contextualize_query(self, question: str, chat_history: List[Dict[str, str]], model: str) -> str:
        """Reformulates follow-up questions into standalone search queries using past turns."""
        if not chat_history:
            return question

        # Skip extra LLM latency if the question is already detailed and self-contained
        words = question.lower().split()
        ambiguous_triggers = {"it", "this", "that", "these", "those", "why", "how", "more", "again", "explain", "what about"}
        is_short_or_ambiguous = len(words) < 6 or any(w.strip("?,.!") in ambiguous_triggers for w in words[:3])
        if not is_short_or_ambiguous:
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
            llm = self._get_llm(model=model, max_tokens=150)
            return self._invoke_llm_with_retry(llm, rephrase_prompt).strip()
        except Exception:
            return question

    def query(
        self, 
        question: str, 
        chat_history: Optional[List[Dict[str, str]]] = None,
        top_k: int = 4, 
        model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Performs contextual retrieval and multi-turn response generation."""
        cleaned_question = question.strip()
        if not cleaned_question:
            return {"answer": "Please provide a valid question.", "sources": []}

        selected_model = model or self.default_model
        collection_count = self.vector_store._collection.count()

        # If no documents exist in vector store, provide direct AI answer
        if collection_count == 0:
            direct_messages = [
                SystemMessage(
                    content=(
                        "You are DocPilot AI, an expert academic tutor and technical assistant specializing in Operating Systems and computer science.\n"
                        "Formatting Rules:\n"
                        "1. Explain the concepts thoroughly, clearly, and step-by-step.\n"
                        "2. TABLES: Format comparison matrices, process tables, and registers strictly as Markdown tables with header pipes.\n"
                        "3. GANTT CHARTS & DIAGRAMS: Wrap execution flows, ASCII timelines, or queues in fenced code blocks (```text ... ```).\n"
                        "4. MATH & FORMULAS: Present calculations step-by-step with clean arithmetic.\n"
                    )
                )
            ]
            if chat_history:
                for turn in chat_history[-4:]:
                    if turn.get("role") == "user":
                        direct_messages.append(HumanMessage(content=turn.get("content", "")))
                    else:
                        direct_messages.append(AIMessage(content=turn.get("content", "")))
            direct_messages.append(HumanMessage(content=cleaned_question))
            try:
                llm = self._get_llm(model=selected_model, max_tokens=4096)
                answer = self._invoke_llm_with_retry(llm, direct_messages)
            except Exception as e:
                answer = f"Error communicating with Gemini AI: {str(e)}"
            return {"answer": answer, "sources": [], "standalone_query": cleaned_question}

        search_query = cleaned_question
        if chat_history and len(chat_history) > 0:
            search_query = self._contextualize_query(cleaned_question, chat_history, selected_model)

        retrieved_results = self.vector_store.similarity_search_with_score(
            search_query, 
            k=min(top_k, collection_count)
        )

        formatted_context_list = []
        sources: List[Dict[str, Any]] = []

        for i, (doc, score) in enumerate(retrieved_results):
            page_num = doc.metadata.get("page", 0) + 1
            doc_name = doc.metadata.get("doc_name") or os.path.basename(doc.metadata.get("source", "Document"))
            content = doc.page_content.strip()
            # Normalize distance score to a 0.0 - 1.0 confidence value
            norm_score = round(max(0.0, min(1.0, 1.0 / (1.0 + float(score)))), 3) if score is not None else 0.0
            
            sources.append({
                "chunk_id": i + 1,
                "doc_name": doc_name,
                "page": page_num,
                "relevance_score": norm_score,
                "snippet": content[:140]
            })
            
            formatted_context_list.append(f"[Chunk {i+1} | Document: {doc_name} | Page {page_num}]\n{content}")

        joined_context = "\n\n".join(formatted_context_list)

        messages = [
            SystemMessage(
                content=(
                    "You are DocPilot AI, an expert academic tutor and technical assistant specializing in Operating Systems and computer science.\n"
                    "Formatting Rules:\n"
                    "1. Explain the concepts thoroughly, clearly, and step-by-step using the provided context whenever relevant.\n"
                    "2. CITATIONS: When citing information from the context, cite specific Document names and Page numbers (e.g. [Operating_Systems_Core_Guide.pdf, Page X] or [UploadedDoc.pdf, Page X]).\n"
                    "3. GENERAL & TECHNICAL QUESTIONS: If the question asks for explanations, comparisons, algorithms, or examples that go beyond the excerpted chunks, leverage your full academic technical expertise to provide a complete, rigorous, and accurate answer rather than refusing.\n"
                    "4. TABLES: Whenever presenting structured data, process tables, comparison matrices, or registers, ALWAYS format them strictly as standard GitHub-flavored Markdown tables with header pipes (e.g. | Algorithm | Preemptive | Waiting Time |). NEVER output raw tabs or whitespace-separated columns.\n"
                    "5. GANTT CHARTS & DIAGRAMS: Whenever illustrating Gantt charts, ASCII timelines, execution flows, or queue states (e.g. +---+---+ or [P1, P2]), ALWAYS wrap them inside fenced code blocks (```text ... ```) so spaces, alignment, and monospace formatting are preserved.\n"
                    "6. MATH & FORMULAS: Present calculations cleanly with arithmetic steps.\n"
                    "7. MULTI-DOC: When information comes from different loaded documents, synthesize or contrast them clearly."
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
            llm = self._get_llm(model=selected_model, max_tokens=4096)
            answer = self._invoke_llm_with_retry(llm, messages)
        except Exception as e:
            answer = f"Error communicating with Gemini API: {str(e)}"

        return {
            "answer": answer,
            "sources": sources,
            "standalone_query": search_query
        }

    def step_explain(self, question: str, answer: str, model: Optional[str] = None) -> Dict[str, Any]:
        """Generates a step-by-step walkthrough of a numerical problem or conceptual demonstration."""
        selected_model = model or self.default_model

        system_prompt = (
            "You are DocPilot AI Step-by-Step Tutor.\n"
            "The student has already received a summary answer. Now they clicked 'Step-by-Step Explain' and want a "
            "detailed, animated walkthrough.\n\n"
            "RULES:\n"
            "1. Break the solution into clear, numbered steps. Each step should be a complete, self-contained unit.\n"
            "2. Start each step with '### Step N: <Title>' on its own line.\n"
            "3. For NUMERICAL PROBLEMS (scheduling, calculations, algorithms):\n"
            "   - Show the input data as a Markdown table first.\n"
            "   - Walk through each iteration/time-unit showing intermediate state.\n"
            "   - Show Gantt charts or queue states in fenced ```text``` code blocks.\n"
            "   - Show all arithmetic explicitly (e.g. TAT = CT - AT = 9 - 0 = 9).\n"
            "   - End with a final results table.\n"
            "4. For CONCEPTUAL/PROCESS explanations:\n"
            "   - Walk through the mechanism phase by phase.\n"
            "   - Use concrete examples from the document context.\n"
            "   - Use diagrams in ```text``` code blocks where helpful.\n"
            "5. FORMAT: Use standard Markdown tables (| col | col |), fenced code blocks, "
            "and LaTeX math ($...$) throughout. NEVER use raw tabs.\n"
            "6. Each step should make sense on its own since they will be revealed one at a time.\n"
            "7. Be thorough — if the student asked for a numerical solution, solve it completely."
        )

        user_prompt = (
            f"Original Question: {question}\n\n"
            f"Summary Answer Already Given:\n{answer}\n\n"
            "Now produce the full step-by-step walkthrough as described in your rules."
        )

        try:
            llm = self._get_llm(model=selected_model, max_tokens=4096)
            full_response = self._invoke_llm_with_retry(llm, [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt)
            ])

            # Split the response into individual steps by "### Step" headers
            import re as _re
            step_blocks = _re.split(r'(?=^### Step )', full_response, flags=_re.MULTILINE)
            steps = [s.strip() for s in step_blocks if s.strip()]

            # If the model didn't follow the format, just split by double newlines
            if len(steps) <= 1:
                raw_steps = full_response.split("\n\n")
                steps = []
                for idx, chunk in enumerate(raw_steps):
                    chunk = chunk.strip()
                    if chunk:
                        if not chunk.startswith("###"):
                            chunk = f"### Step {idx + 1}\n{chunk}"
                        steps.append(chunk)

            return {
                "steps": steps,
                "total_steps": len(steps)
            }
        except Exception as e:
            return {
                "steps": [f"### Error\nCould not generate step-by-step explanation: {str(e)}"],
                "total_steps": 1
            }

    def generate_study_guide(self, model: Optional[str] = None, progress_callback=None) -> str:
        """Generates a comprehensive Master Study Guide using single-shot synthesis with caching to avoid rate limits."""
        # Check cache first
        cache_key = "_".join(sorted(self.doc_registry.keys())) or "default_os_guide"
        if cache_key in self._cached_study_guide:
            if progress_callback:
                progress_callback(1.0, "Study Guide Ready (Loaded from Cache)!")
            return self._cached_study_guide[cache_key]

        # Ensure documents are available; if none, index default OS guide
        if not self.raw_documents:
            self.ensure_default_os_knowledge()

        selected_model = model or self.default_model

        if progress_callback:
            progress_callback(0.25, "Extracting core concepts and structure...")

        # Aggregate excerpted content across active documents (up to 18,000 chars for optimal single-shot context)
        aggregated_snippets = []
        doc_count = len(self.raw_documents_map)

        for doc_name, pages in self.raw_documents_map.items():
            doc_text = ""
            for p in pages:
                p_num = p.metadata.get("page", 0) + 1
                content = p.page_content.strip()
                if content:
                    doc_text += f"\n[{doc_name} Page {p_num}]:\n{content}\n"
            # Cap per document to balance multi-doc representation
            char_budget = 18000 // max(1, doc_count)
            aggregated_snippets.append(doc_text[:char_budget])

        source_material = "\n\n".join(aggregated_snippets)
        if not source_material.strip():
            source_material = "Core Operating Systems Principles: Processes, Threads, CPU Scheduling, Synchronization, Memory Management, Paging, Virtual Memory, and File Systems."

        if progress_callback:
            progress_callback(0.65, "Synthesizing master study guide...")

        study_guide_prompt = (
            "You are a distinguished university professor in Computer Science and Operating Systems.\n"
            "Create a master study guide and comprehensive exam review based on the following material.\n\n"
            "Format your guide cleanly in Markdown with tables:\n"
            "# 📘 Master Study Guide & Exam Prep\n"
            "## 1. Executive Summary & Foundational Principles\n"
            "## 2. Core Technical Breakdown (with citations like [Document, Page X] when available)\n"
            "## 3. High-Yield Comparison Tables (e.g., Scheduling Algorithms, Paging vs Segmentation, Mutex vs Semaphore)\n"
            "## 4. Key Formulas, Numerical Solving Tips & Formulas (e.g., Turnaround Time, Waiting Time, Effective Access Time)\n"
            "## 5. Must-Know Exam & Viva Review Questions\n\n"
            f"Reference Material:\n{source_material}"
        )

        try:
            llm = self._get_llm(model=selected_model, max_tokens=4096)
            guide = self._invoke_llm_with_retry(llm, study_guide_prompt)
            if progress_callback:
                progress_callback(1.0, "Study Guide Ready!")
            if guide and not guide.startswith("Error"):
                self._cached_study_guide[cache_key] = guide
            return guide
        except Exception as e:
            return f"Error synthesizing study guide: {str(e)}"