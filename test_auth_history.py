import sys
import json
from database import init_db, SessionLocal, User, Conversation, Message

# Reconfigure stdout for UTF-8 on Windows
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

def run_test():
    print("=== Testing Database & Chat History Backend Integration ===")
    init_db()
    print("[1] Database initialized successfully.")

    db = SessionLocal()
    test_user_id = "test_google_user_123"

    # Clean up prior test user if exists
    existing = db.query(User).filter(User.id == test_user_id).first()
    if existing:
        db.delete(existing)
        db.commit()

    # Create user
    user = User(
        id=test_user_id,
        email="student@university.edu",
        name="Alex Researcher",
        avatar="https://lh3.googleusercontent.com/a/test"
    )
    db.add(user)
    db.commit()
    print("[2] User created in database.")

    # Create conversation
    conv = Conversation(user_id=test_user_id, title="Physics Midterm Review")
    db.add(conv)
    db.commit()
    db.refresh(conv)
    print(f"[3] Created conversation ID: {conv.id} ('{conv.title}')")

    # Add messages
    m1 = Message(
        conversation_id=conv.id,
        role="user",
        content="What is Newton's second law?"
    )
    m2 = Message(
        conversation_id=conv.id,
        role="assistant",
        content="Newton's second law states that F = m * a.",
        sources=json.dumps([{"doc_name": "physics.pdf", "page": 42}])
    )
    db.add(m1)
    db.add(m2)
    db.commit()
    print("[4] Added user and assistant messages with sources.")

    # Query back
    saved_msgs = db.query(Message).filter(Message.conversation_id == conv.id).order_by(Message.timestamp.asc()).all()
    assert len(saved_msgs) == 2, f"Expected 2 messages, got {len(saved_msgs)}"
    assert saved_msgs[0].role == "user"
    assert saved_msgs[1].role == "assistant"
    parsed_sources = json.loads(saved_msgs[1].sources)
    assert parsed_sources[0]["doc_name"] == "physics.pdf"
    assert parsed_sources[0]["page"] == 42
    print(f"[5] Verified retrieved messages and parsed sources: {parsed_sources}")

    # Query user conversations
    user_convs = db.query(Conversation).filter(Conversation.user_id == test_user_id).all()
    assert len(user_convs) == 1
    print(f"[6] Verified user conversations query returned {len(user_convs)} chat.")

    # Clean up test user & cascade delete
    db.delete(user)
    db.commit()
    remaining = db.query(Conversation).filter(Conversation.user_id == test_user_id).all()
    assert len(remaining) == 0, "Expected cascade delete on conversation"
    db.close()
    print("[7] Cascade delete verified on cleanup.")

    print("\n=== ALL DATABASE & CHAT HISTORY TESTS PASSED! ===")

if __name__ == "__main__":
    run_test()
