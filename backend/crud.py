"""
CRUD helpers for database operations.
"""

from __future__ import annotations

import uuid
from typing import Sequence

from sqlalchemy.orm import Session
from sqlalchemy import func, text

from models import User, Document, DocumentChunk, AuditLog, ChatSession, ChatMessage


# ── Users ─────────────────────────────────────────────────────────────────────

def create_user(db: Session, *, username: str, email: str) -> User:
    user = User(username=username, email=email)
    db.add(user)
    db.commit()
    db.refresh(user)
    return user


def get_user(db: Session, user_id: uuid.UUID) -> User | None:
    return db.query(User).filter(User.id == user_id).first()


def list_users(db: Session) -> Sequence[User]:
    return db.query(User).order_by(User.created_at.desc()).all()


# ── Documents ─────────────────────────────────────────────────────────────────

def create_document(
    db: Session,
    *,
    filename: str,
    file_size: int,
    metadata_json: dict | None = None,
) -> Document:
    doc = Document(
        filename=filename,
        file_size=file_size,
        metadata_json=metadata_json or {},
    )
    db.add(doc)
    db.commit()
    db.refresh(doc)
    return doc


def add_chunks(
    db: Session,
    document_id: uuid.UUID,
    chunks: list[dict],
) -> list[DocumentChunk]:
    """Bulk-insert chunks with optional embeddings for a document."""
    objects = [
        DocumentChunk(
            document_id=document_id,
            chunk_index=idx,
            text=c["text"],
            embedding=c.get("embedding"),
        )
        for idx, c in enumerate(chunks)
    ]
    db.add_all(objects)
    db.commit()
    return objects


def list_documents(db: Session) -> Sequence[Document]:
    return db.query(Document).order_by(Document.uploaded_at.desc()).all()


def get_document(db: Session, doc_id: uuid.UUID) -> Document | None:
    return db.query(Document).filter(Document.id == doc_id).first()


def delete_document(db: Session, doc_id: uuid.UUID) -> None:
    """Delete a document and its chunks (CASCADE via FK)."""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if doc:
        db.delete(doc)
        db.commit()


# ── Similarity search (pgvector) ─────────────────────────────────────────────

def similarity_search(
    db: Session,
    *,
    query_embedding: list[float],
    top_k: int = 5,
    threshold: float = 0.0,
) -> list[tuple[DocumentChunk, float, Document]]:
    """
    Perform cosine similarity search over document_chunks using pgvector.

    Returns a list of (chunk, similarity_score, document) tuples ordered
    by descending similarity.

    Uses the ``<=>`` cosine distance operator which leverages the IVFFlat
    index defined on ``document_chunks.embedding``.
    """
    # pgvector's <=> returns cosine *distance*, so similarity = 1 - distance
    distance = DocumentChunk.embedding.cosine_distance(query_embedding)

    results = (
        db.query(
            DocumentChunk,
            (1 - distance).label("similarity"),
            Document,
        )
        .join(Document, Document.id == DocumentChunk.document_id)
        .filter(DocumentChunk.embedding.isnot(None))
        .filter((1 - distance) >= threshold)
        .order_by(distance.asc())
        .limit(top_k)
        .all()
    )

    return [(chunk, sim, doc) for chunk, sim, doc in results]


# ── Audit Logs ────────────────────────────────────────────────────────────────

def create_audit_log(
    db: Session,
    *,
    user_id: uuid.UUID | None = None,
    action: str,
    token_usage: int | None = None,
    latency_ms: float | None = None,
) -> AuditLog:
    log = AuditLog(
        user_id=user_id,
        action=action,
        token_usage=token_usage,
        latency_ms=latency_ms,
    )
    db.add(log)
    db.commit()
    db.refresh(log)
    return log


def list_audit_logs(
    db: Session,
    limit: int = 100,
    action_filter: str | None = None,
) -> Sequence[AuditLog]:
    """Return recent audit logs, optionally filtered by action type."""
    query = db.query(AuditLog)
    if action_filter:
        query = query.filter(AuditLog.action == action_filter)
    return query.order_by(AuditLog.created_at.desc()).limit(limit).all()


# ── Chat sessions & messages ─────────────────────────────────────────────────

def create_session(db: Session, title: str = "New Chat") -> ChatSession:
    session = ChatSession(title=title)
    db.add(session)
    db.commit()
    db.refresh(session)
    return session


def get_session(db: Session, session_id: uuid.UUID) -> ChatSession | None:
    return db.query(ChatSession).filter(ChatSession.id == session_id).first()


def list_sessions(db: Session) -> Sequence[ChatSession]:
    return db.query(ChatSession).order_by(ChatSession.created_at.desc()).all()


def add_message(
    db: Session,
    session_id: uuid.UUID,
    role: str,
    content: str,
) -> ChatMessage:
    msg = ChatMessage(session_id=session_id, role=role, content=content)
    db.add(msg)
    db.commit()
    db.refresh(msg)
    return msg


# ── Analytics ─────────────────────────────────────────────────────────────────

def get_analytics(db: Session) -> dict:
    return {
        "total_users": db.query(func.count(User.id)).scalar() or 0,
        "total_documents": db.query(func.count(Document.id)).scalar() or 0,
        "total_chunks": db.query(func.count(DocumentChunk.id)).scalar() or 0,
        "total_sessions": db.query(func.count(ChatSession.id)).scalar() or 0,
        "total_messages": db.query(func.count(ChatMessage.id)).scalar() or 0,
        "total_audit_logs": db.query(func.count(AuditLog.id)).scalar() or 0,
    }
