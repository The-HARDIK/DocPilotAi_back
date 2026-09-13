import os
import shutil
import json
from typing import List, Optional
from fastapi import FastAPI, UploadFile, File, Query, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel
from google.oauth2 import id_token
from google.auth.transport import requests as grequests

from rag_engine import DocPilotEngine
from database import init_db, SessionLocal, User, Conversation, Message

# Initialize DB on startup
init_db()

GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "247202504530-chsa46107gb9ii7l9aob0spg945fg6pq.apps.googleusercontent.com")

app = FastAPI(title="DocPilot AI API", version="1.0.0")

# Configure CORS
allowed_origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
frontend_env = os.getenv("FRONTEND_URL")
if frontend_env:
    for url in frontend_env.split(","):
        cleaned = url.strip().rstrip("/")
        if cleaned and cleaned not in allowed_origins:
            allowed_origins.append(cleaned)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_origin_regex=r"https://.*\.vercel\.app|https://.*\.netlify\.app",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

engine = DocPilotEngine()

# Dedicated storage folder for active documents
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploaded_docs")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Tracks which document is currently active in the viewer
selected_document: Optional[str] = None

class ChatMessage(BaseModel):
    role: str
    content: str

class QueryRequest(BaseModel):
    question: str
    chat_history: Optional[List[ChatMessage]] = []
    conversation_id: Optional[int] = None

class SelectDocRequest(BaseModel):
    filename: str

class GoogleAuthRequest(BaseModel):
    credential: str

class CreateConversationRequest(BaseModel):
    user_id: str
    title: Optional[str] = "New Study Session"
    documents: Optional[List[str]] = []

class SaveMessageRequest(BaseModel):
    conversation_id: int
    role: str
    content: str
    sources: Optional[list] = []

def get_current_selected_doc() -> Optional[str]:
    global selected_document
    docs = engine.get_documents()
    if not docs:
        selected_document = None
        return None
    filenames = [d["filename"] for d in docs]
    if selected_document not in filenames:
        selected_document = filenames[0]
    return selected_document

@app.get("/api/status")
def get_status():
    docs = engine.get_documents()
    current_doc = get_current_selected_doc()
    total_chunks = sum(d["chunks"] for d in docs)
    return {
        "status": "ready",
        "documents": docs,
        "active_document": current_doc,
        "total_documents": len(docs),
        "max_documents": engine.MAX_DOCUMENTS,
        "total_chunks": total_chunks
    }

@app.post("/api/upload")
async def upload_documents(
    files: Optional[List[UploadFile]] = File(None),
    file: Optional[UploadFile] = File(None)
):
    global selected_document
    # Collect all provided files (supporting both single and multiple uploads)
    all_files: List[UploadFile] = []
    if files:
        all_files.extend(files)
    if file:
        all_files.append(file)

    if not all_files:
        raise HTTPException(status_code=400, detail="No files provided for upload.")

    for f in all_files:
        if not f.filename.lower().endswith(".pdf"):
            raise HTTPException(status_code=400, detail=f"File '{f.filename}' is not a PDF. Only PDF files are supported.")

    existing_docs = {d["filename"] for d in engine.get_documents()}
    # Count how many new distinct documents are being added
    new_filenames = {f.filename for f in all_files if f.filename not in existing_docs}
    if len(existing_docs) + len(new_filenames) > engine.MAX_DOCUMENTS:
        raise HTTPException(
            status_code=400,
            detail=f"Limit exceeded: doc library can hold at most {engine.MAX_DOCUMENTS} PDFs. You currently have {len(existing_docs)} loaded."
        )

    uploaded_results = []
    for f in all_files:
        saved_path = os.path.join(UPLOAD_DIR, f.filename)
        with open(saved_path, "wb") as buffer:
            shutil.copyfileobj(f.file, buffer)

        try:
            chunks = engine.add_pdf(saved_path, filename=f.filename)
            uploaded_results.append({"filename": f.filename, "chunks": chunks})
            if selected_document is None:
                selected_document = f.filename
        except Exception as e:
            if os.path.exists(saved_path):
                os.remove(saved_path)
            raise HTTPException(status_code=500, detail=f"Error indexing '{f.filename}': {str(e)}")

    docs = engine.get_documents()
    current_doc = get_current_selected_doc()
    return {
        "message": f"{len(uploaded_results)} document(s) processed successfully.",
        "documents": docs,
        "uploaded": uploaded_results,
        "active_document": current_doc,
        "total_chunks": sum(d["chunks"] for d in docs)
    }

@app.get("/api/document")
def get_document(filename: Optional[str] = Query(None)):
    """Serves an uploaded PDF directly for the split-screen viewer."""
    target_filename = filename or get_current_selected_doc()
    if not target_filename:
        raise HTTPException(status_code=404, detail="No active document found.")

    file_path = os.path.join(UPLOAD_DIR, target_filename)
    if not os.path.isfile(file_path):
        raise HTTPException(status_code=404, detail=f"Document '{target_filename}' not found.")

    return FileResponse(
        path=file_path, 
        media_type="application/pdf",
        filename=target_filename,
        headers={
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Expose-Headers": "Content-Disposition"
        }
    )

@app.post("/api/document/select")
def select_active_document(payload: SelectDocRequest):
    global selected_document
    docs = {d["filename"] for d in engine.get_documents()}
    if payload.filename not in docs:
        raise HTTPException(status_code=404, detail=f"Document '{payload.filename}' not found in loaded documents.")
    selected_document = payload.filename
    return {"message": f"Active preview set to {selected_document}", "active_document": selected_document}

@app.delete("/api/document/{filename}")
def delete_document(filename: str):
    global selected_document
    removed = engine.remove_pdf(filename)
    if not removed:
        raise HTTPException(status_code=404, detail=f"Document '{filename}' not found.")

    file_path = os.path.join(UPLOAD_DIR, filename)
    if os.path.isfile(file_path):
        try:
            os.remove(file_path)
        except Exception:
            pass

    if selected_document == filename:
        selected_document = None
    current_doc = get_current_selected_doc()

    docs = engine.get_documents()
    return {
        "message": f"Document '{filename}' removed successfully.",
        "documents": docs,
        "active_document": current_doc,
        "total_chunks": sum(d["chunks"] for d in docs)
    }

@app.delete("/api/documents/clear")
def clear_all_documents():
    global selected_document
    engine.clear_all()
    selected_document = None
    # Remove files in UPLOAD_DIR
    for fname in os.listdir(UPLOAD_DIR):
        fpath = os.path.join(UPLOAD_DIR, fname)
        if os.path.isfile(fpath):
            try:
                os.remove(fpath)
            except Exception:
                pass
    return {"message": "All documents cleared successfully.", "documents": [], "active_document": None}

# ==========================================
# Google Auth & Chat History Endpoints
# ==========================================

@app.post("/api/auth/google")
def google_auth(payload: GoogleAuthRequest):
    try:
        # Accommodate minor system clock drift with clock_skew_in_seconds=10
        idinfo = id_token.verify_oauth2_token(
            payload.credential,
            grequests.Request(),
            GOOGLE_CLIENT_ID,
            clock_skew_in_seconds=10
        )
        user_id = idinfo["sub"]
        email = idinfo["email"]
        name = idinfo.get("name", "")
        avatar = idinfo.get("picture", "")

        db = SessionLocal()
        user = db.query(User).filter(User.id == user_id).first()
        if not user:
            user = User(id=user_id, email=email, name=name, avatar=avatar)
            db.add(user)
            db.commit()
            db.refresh(user)
        else:
            user.name = name
            user.avatar = avatar
            db.commit()
            db.refresh(user)

        user_data = {"id": user.id, "email": user.email, "name": user.name, "avatar": user.avatar}
        db.close()

        return {
            "status": "success",
            "user": user_data
        }
    except ValueError as ve:
        raise HTTPException(status_code=400, detail=f"Google authentication token invalid or expired: {str(ve)}")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Authentication failed: {str(e)}")

@app.get("/api/chats/{user_id}")
def get_user_chats(user_id: str):
    db = SessionLocal()
    chats = db.query(Conversation).filter(Conversation.user_id == user_id).order_by(Conversation.created_at.desc()).all()
    result = [
        {
            "id": c.id,
            "title": c.title,
            "documents": json.loads(c.documents) if c.documents else [],
            "created_at": c.created_at.isoformat() if hasattr(c.created_at, "isoformat") else str(c.created_at)
        }
        for c in chats
    ]
    db.close()
    return result

@app.post("/api/chats")
def create_chat(payload: CreateConversationRequest):
    db = SessionLocal()
    # Snapshot active documents from engine if not explicitly provided
    docs_to_record = payload.documents
    if not docs_to_record:
        docs_to_record = [d["filename"] for d in engine.get_documents()]

    chat = Conversation(
        user_id=payload.user_id,
        title=payload.title or "New Study Session",
        documents=json.dumps(docs_to_record) if docs_to_record else None
    )
    db.add(chat)
    db.commit()
    db.refresh(chat)
    chat_id = chat.id
    chat_title = chat.title
    db.close()
    return {"id": chat_id, "title": chat_title, "documents": docs_to_record or []}

@app.delete("/api/chats/{conversation_id}")
def delete_chat(conversation_id: int):
    db = SessionLocal()
    chat = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not chat:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")
    db.delete(chat)
    db.commit()
    db.close()
    return {"status": "success", "message": "Conversation deleted"}

@app.get("/api/chats/messages/{conversation_id}")
def get_chat_messages(conversation_id: int):
    db = SessionLocal()
    chat = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not chat:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    conv_docs = json.loads(chat.documents) if chat.documents else []
    msgs = db.query(Message).filter(Message.conversation_id == conversation_id).order_by(Message.timestamp.asc()).all()
    result = [
        {
            "role": m.role,
            "content": m.content,
            "sources": json.loads(m.sources) if m.sources else []
        }
        for m in msgs
    ]
    db.close()
    return {
        "conversation_id": conversation_id,
        "title": chat.title,
        "documents": conv_docs,
        "messages": result
    }

@app.post("/api/chats/{conversation_id}/restore")
def restore_chat_documents(conversation_id: int):
    """Aligns active vector database with documents associated with this conversation."""
    global selected_document
    db = SessionLocal()
    chat = db.query(Conversation).filter(Conversation.id == conversation_id).first()
    if not chat:
        db.close()
        raise HTTPException(status_code=404, detail="Conversation not found")

    target_docs = json.loads(chat.documents) if chat.documents else []
    db.close()

    current_loaded = [d["filename"] for d in engine.get_documents()]

    # If the active documents match already, no need to reindex
    if sorted(current_loaded) == sorted(target_docs) and len(target_docs) > 0:
        return {
            "status": "already_aligned",
            "restored": False,
            "documents": engine.get_documents(),
            "active_document": get_current_selected_doc()
        }

    # Re-index PDFs that are in the session and present in uploads folder
    if target_docs:
        engine.clear_all()
        selected_document = None
        restored = []
        for doc_name in target_docs:
            saved_path = os.path.join(UPLOAD_DIR, doc_name)
            if os.path.exists(saved_path):
                try:
                    engine.add_pdf(saved_path, filename=doc_name)
                    restored.append(doc_name)
                    if selected_document is None:
                        selected_document = doc_name
                except Exception as e:
                    print(f"Warning: could not restore {doc_name}: {e}")

        return {
            "status": "restored",
            "restored": True,
            "restored_documents": restored,
            "documents": engine.get_documents(),
            "active_document": get_current_selected_doc()
        }

    return {
        "status": "no_documents",
        "restored": False,
        "documents": engine.get_documents(),
        "active_document": get_current_selected_doc()
    }

@app.post("/api/chats/messages")
def save_chat_message(payload: SaveMessageRequest):
    db = SessionLocal()
    msg = Message(
        conversation_id=payload.conversation_id,
        role=payload.role,
        content=payload.content,
        sources=json.dumps(payload.sources) if payload.sources else None
    )
    db.add(msg)
    db.commit()
    db.refresh(msg)
    msg_id = msg.id
    db.close()
    return {"status": "success", "message_id": msg_id}

@app.post("/api/chat")
async def chat_query(payload: QueryRequest):
    docs = engine.get_documents()
    if not docs:
        raise HTTPException(status_code=400, detail="No documents currently loaded. Upload up to 5 PDFs first.")

    history_dict = [{"role": msg.role, "content": msg.content} for msg in payload.chat_history]
    result = engine.query(question=payload.question, chat_history=history_dict)

    # Automatically persist turn if associated with an active conversation
    if payload.conversation_id:
        try:
            db = SessionLocal()
            chat = db.query(Conversation).filter(Conversation.id == payload.conversation_id).first()
            if chat and not chat.documents:
                # Snapshot active document names into the conversation
                current_doc_names = [d["filename"] for d in docs]
                chat.documents = json.dumps(current_doc_names)

            user_msg = Message(
                conversation_id=payload.conversation_id,
                role="user",
                content=payload.question
            )
            asst_msg = Message(
                conversation_id=payload.conversation_id,
                role="assistant",
                content=result["answer"],
                sources=json.dumps(result.get("sources", [])) if result.get("sources") else None
            )
            db.add(user_msg)
            db.add(asst_msg)
            db.commit()
            db.close()
        except Exception:
            pass

    return result

@app.post("/api/study-guide")
async def generate_guide():
    docs = engine.get_documents()
    if not docs:
        raise HTTPException(status_code=400, detail="No documents currently loaded. Upload up to 5 PDFs first.")

    try:
        guide_md = engine.generate_study_guide()
        return {"study_guide": guide_md}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port, reload=True)