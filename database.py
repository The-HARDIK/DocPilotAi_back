import os
import socket
from urllib.parse import urlparse, parse_qs, urlencode, urlunparse
from sqlalchemy import create_engine, Column, String, Integer, Text, ForeignKey, DateTime, LargeBinary, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from datetime import datetime
from dotenv import load_dotenv

env_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
if os.path.exists(env_path):
    load_dotenv(env_path)
else:
    load_dotenv()

raw_db_url = os.getenv("DATABASE_URL")

def get_engine():
    if raw_db_url and ("postgres" in raw_db_url or "neon.tech" in raw_db_url):
        url = raw_db_url.replace("postgres://", "postgresql://", 1)
        
        # Test if DNS resolves for host; if campus network refuses DNS, inject hostaddr
        try:
            parsed = urlparse(url)
            host = parsed.hostname
            if host:
                try:
                    socket.gethostbyname(host)
                except socket.gaierror:
                    # DNS resolution failed locally (e.g. campus Wi-Fi blocking .tech TLD)
                    fallback_ip = "52.76.246.190"
                    query_params = parse_qs(parsed.query)
                    if "hostaddr" not in query_params:
                        query_params["hostaddr"] = [fallback_ip]
                        new_query = urlencode(query_params, doseq=True)
                        url = urlunparse((parsed.scheme, parsed.netloc, parsed.path, parsed.params, new_query, parsed.fragment))
        except Exception:
            pass

        try:
            pg_engine = create_engine(
                url,
                pool_pre_ping=True,
                pool_recycle=300
            )
            print("[DocPilot] Initialized Neon PostgreSQL cloud database engine.")
            return pg_engine
        except Exception as e:
            print(f"[DocPilot] Engine creation for Neon PostgreSQL failed: {e}")
            raise ConnectionError(f"Failed to initialize Neon PostgreSQL database engine: {e}")

    raise ValueError("DATABASE_URL is not configured. Please supply a valid Neon PostgreSQL connection string in .env.")

engine = get_engine()
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(String, primary_key=True)  # Google sub / email
    email = Column(String, unique=True, index=True)
    name = Column(String)
    avatar = Column(String)
    
    conversations = relationship("Conversation", back_populates="user", cascade="all, delete-orphan")

class Conversation(Base):
    __tablename__ = "conversations"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    user_id = Column(String, ForeignKey("users.id"))
    title = Column(String, default="New Chat")
    documents = Column(Text, nullable=True)  # JSON list of document filenames associated with this session
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="conversations")
    messages = relationship("Message", back_populates="conversation", cascade="all, delete-orphan")

class Message(Base):
    __tablename__ = "messages"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    conversation_id = Column(Integer, ForeignKey("conversations.id"))
    role = Column(String)  # 'user' or 'assistant'
    content = Column(Text)
    sources = Column(Text, nullable=True)  # JSON string of sources if needed
    timestamp = Column(DateTime, default=datetime.utcnow)

    conversation = relationship("Conversation", back_populates="messages")

class DocumentRecord(Base):
    """Stores uploaded PDF documents directly in Neon PostgreSQL."""
    __tablename__ = "documents"
    id = Column(Integer, primary_key=True, index=True, autoincrement=True)
    filename = Column(String, unique=True, index=True, nullable=False)
    user_id = Column(String, ForeignKey("users.id"), nullable=True)
    file_data = Column(LargeBinary, nullable=False)  # Binary PDF content in Neon PostgreSQL
    file_size = Column(Integer, default=0)
    page_count = Column(Integer, default=0)
    chunk_count = Column(Integer, default=0)
    created_at = Column(DateTime, default=datetime.utcnow)

def init_db():
    Base.metadata.create_all(bind=engine)
    with engine.connect() as conn:
        try:
            dialect = engine.dialect.name
            if dialect == "postgresql":
                # Ensure PostgreSQL sequences are correctly synced
                conn.execute(text("SELECT setval(pg_get_serial_sequence('conversations', 'id'), COALESCE((SELECT MAX(id) FROM conversations), 1))"))
                conn.execute(text("SELECT setval(pg_get_serial_sequence('messages', 'id'), COALESCE((SELECT MAX(id) FROM messages), 1))"))
                conn.execute(text("SELECT setval(pg_get_serial_sequence('documents', 'id'), COALESCE((SELECT MAX(id) FROM documents), 1))"))
                conn.commit()
        except Exception as e:
            print(f"[DocPilot] Note on init_db sequence sync: {e}")