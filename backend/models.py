"""
SQLAlchemy ORM models for the Enterprise RAG & SQL Copilot.

Tables
------
- **users**            – application users
- **documents**        – uploaded source documents (PDF, CSV, …)
- **document_chunks**  – chunked text with pgvector embeddings
- **audit_logs**       – per-request telemetry (action, tokens, latency)
- **chat_sessions**    – conversation sessions
- **chat_messages**    – individual messages within a session
"""

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    String,
    Text,
    BigInteger,
    Integer,
    Float,
    DateTime,
    ForeignKey,
    Index,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID, JSONB
from sqlalchemy.orm import relationship
from pgvector.sqlalchemy import Vector

from database import Base


# ── Utility ───────────────────────────────────────────────────────────────────

def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# ═════════════════════════════════════════════════════════════════════════════
#  User
# ═════════════════════════════════════════════════════════════════════════════

class User(Base):
    """An application user who can upload documents and start chats."""

    __tablename__ = "users"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    username = Column(String(128), unique=True, nullable=False, index=True)
    email = Column(String(256), unique=True, nullable=False, index=True)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    audit_logs = relationship(
        "AuditLog",
        back_populates="user",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<User {self.username!r}>"


# ═════════════════════════════════════════════════════════════════════════════
#  Document  +  DocumentChunk (pgvector)
# ═════════════════════════════════════════════════════════════════════════════

class Document(Base):
    """An uploaded source document (PDF, TXT, CSV, DOCX, …)."""

    __tablename__ = "documents"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    filename = Column(String(512), nullable=False)
    file_size = Column(BigInteger, nullable=False, doc="Size in bytes")
    uploaded_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)
    metadata_json = Column(
        JSONB,
        nullable=True,
        default=dict,
        doc="Arbitrary key-value metadata (e.g. source, author, page count)",
    )

    # Relationships
    chunks = relationship(
        "DocumentChunk",
        back_populates="document",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<Document {self.filename!r} ({self.file_size} B)>"


class DocumentChunk(Base):
    """
    A chunked fragment of a :class:`Document` with its vector embedding.

    The ``embedding`` column uses pgvector's ``vector(384)`` type
    (compatible with ``all-MiniLM-L6-v2`` and similar sentence-transformer
    models).  Adjust the dimension constant if you switch models.
    """

    __tablename__ = "document_chunks"

    EMBEDDING_DIM: int = 384  # matches all-MiniLM-L6-v2

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    document_id = Column(
        UUID(as_uuid=True),
        ForeignKey("documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    chunk_index = Column(Integer, nullable=False, doc="0-based position within the document")
    text = Column(Text, nullable=False)
    embedding = Column(Vector(384), doc="pgvector embedding for similarity search")

    # Relationships
    document = relationship("Document", back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_per_doc"),
        Index(
            "ix_chunk_embedding_cosine",
            embedding,
            postgresql_using="ivfflat",
            postgresql_with={"lists": 100},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
    )

    def __repr__(self) -> str:
        return f"<DocumentChunk doc={self.document_id} idx={self.chunk_index}>"


# ═════════════════════════════════════════════════════════════════════════════
#  AuditLog
# ═════════════════════════════════════════════════════════════════════════════

class AuditLog(Base):
    """
    Per-request audit trail capturing who did what and how expensive it was.

    Tracks LLM token consumption and response latency so that operators can
    monitor cost and performance over time.
    """

    __tablename__ = "audit_logs"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    user_id = Column(
        UUID(as_uuid=True),
        ForeignKey("users.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
    )
    action = Column(
        String(128),
        nullable=False,
        index=True,
        doc="Semantic label, e.g. 'chat_query', 'document_upload', 'sql_generation'",
    )
    token_usage = Column(
        Integer,
        nullable=True,
        doc="Total LLM tokens consumed (prompt + completion)",
    )
    latency_ms = Column(
        Float,
        nullable=True,
        doc="End-to-end request latency in milliseconds",
    )
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    # Relationships
    user = relationship("User", back_populates="audit_logs")

    __table_args__ = (
        Index("ix_audit_created", "created_at"),
    )

    def __repr__(self) -> str:
        return f"<AuditLog {self.action!r} user={self.user_id}>"


# ═════════════════════════════════════════════════════════════════════════════
#  Chat Session  +  Message  (retained from initial scaffold)
# ═════════════════════════════════════════════════════════════════════════════

class ChatSession(Base):
    """A conversation session between a user and the copilot."""

    __tablename__ = "chat_sessions"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    title = Column(String(256), default="New Chat")
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    messages = relationship(
        "ChatMessage",
        back_populates="session",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def __repr__(self) -> str:
        return f"<ChatSession {self.title!r}>"


class ChatMessage(Base):
    """Single message within a :class:`ChatSession`."""

    __tablename__ = "chat_messages"

    id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    session_id = Column(
        UUID(as_uuid=True),
        ForeignKey("chat_sessions.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    role = Column(String(16), nullable=False, doc='"user" | "assistant"')
    content = Column(Text, nullable=False)
    created_at = Column(DateTime(timezone=True), default=_utcnow, nullable=False)

    session = relationship("ChatSession", back_populates="messages")

    def __repr__(self) -> str:
        return f"<ChatMessage {self.role!r} session={self.session_id}>"
