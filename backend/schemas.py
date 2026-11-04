"""
Pydantic schemas for request / response validation.
"""

from __future__ import annotations

import uuid
from datetime import date, datetime
from typing import Optional, Any

from pydantic import BaseModel, Field, model_validator


# ── Users ─────────────────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    username: str = Field(..., min_length=1, max_length=128)
    email: str = Field(..., max_length=256)


class UserOut(BaseModel):
    id: uuid.UUID
    username: str
    email: str
    created_at: datetime

    model_config = {"from_attributes": True}


# ── Documents ─────────────────────────────────────────────────────────────────

class DocumentChunkOut(BaseModel):
    """A single chunk that was created during ingestion."""
    id: uuid.UUID
    chunk_index: int
    text: str = Field(description="First 200 chars of the chunk text")

    model_config = {"from_attributes": True}


class DocumentOut(BaseModel):
    """Lightweight document metadata (returned in list endpoints)."""
    id: uuid.UUID
    filename: str
    file_size: int
    uploaded_at: datetime
    metadata_json: Optional[dict[str, Any]] = None

    model_config = {"from_attributes": True}


class DocumentUploadResponse(BaseModel):
    """Rich response returned after a successful upload + ingestion."""
    id: uuid.UUID
    filename: str
    file_size: int
    uploaded_at: datetime
    metadata_json: Optional[dict[str, Any]] = None
    total_chunks: int = Field(description="Number of chunks created")
    chunks_preview: list[DocumentChunkOut] = Field(
        default_factory=list,
        description="Preview of the first 5 chunks (text truncated)",
    )

    model_config = {"from_attributes": True}


class DocumentDetailOut(BaseModel):
    """Full document detail including all chunk texts."""
    id: uuid.UUID
    filename: str
    file_size: int
    uploaded_at: datetime
    metadata_json: Optional[dict[str, Any]] = None
    chunks: list[DocumentChunkOut] = []

    model_config = {"from_attributes": True}


# ── Chat ──────────────────────────────────────────────────────────────────────

class ChatQueryRequest(BaseModel):
    """Payload for the RAG chat endpoint."""
    query: Optional[str] = Field(None, min_length=1, max_length=4096, description="User question")
    content: Optional[str] = Field(None, min_length=1, max_length=4096, description="Alias for query")
    session_id: Optional[uuid.UUID] = Field(
        None,
        description="Existing session ID; omit to auto-create a new session",
    )
    top_k: Optional[int] = Field(
        None, ge=1, le=20,
        description="Number of chunks to retrieve (default: 5)",
    )

    @model_validator(mode="after")
    def check_query_or_content(self):
        if not self.query and not self.content:
            raise ValueError("Either 'query' or 'content' must be provided.")
        if not self.query:
            self.query = self.content
        return self


class SourceChunk(BaseModel):
    """A retrieved document chunk included as a citation source."""
    chunk_id: uuid.UUID
    document_id: uuid.UUID
    filename: str
    chunk_index: int
    text: str = Field(description="Chunk text (truncated to 300 chars)")
    similarity: float = Field(description="Cosine similarity score")


class ChatQueryResponse(BaseModel):
    """Response from the RAG chat endpoint."""
    session_id: uuid.UUID
    answer: str
    content: Optional[str] = None
    sources: list[SourceChunk] = Field(
        default_factory=list,
        description="Document chunks used as context for the answer",
    )
    model: str = Field(description="LLM model used for generation")
    token_usage: Optional[int] = Field(None, description="Total tokens consumed")
    latency_ms: float = Field(description="End-to-end latency in milliseconds")

    @model_validator(mode="after")
    def sync_content(self):
        if not self.content:
            self.content = self.answer
        return self


class ChatMessageCreate(BaseModel):
    content: str = Field(..., min_length=1, max_length=4096)


class ChatMessageOut(BaseModel):
    id: uuid.UUID
    role: str
    content: str
    created_at: datetime

    model_config = {"from_attributes": True}


class ChatSessionCreate(BaseModel):
    title: Optional[str] = "New Chat"


class ChatSessionOut(BaseModel):
    id: uuid.UUID
    title: str
    created_at: datetime
    messages: list[ChatMessageOut] = []

    model_config = {"from_attributes": True}


# ── Audit Logs ────────────────────────────────────────────────────────────────

class AuditLogOut(BaseModel):
    id: uuid.UUID
    user_id: Optional[uuid.UUID] = None
    action: str
    token_usage: Optional[int] = None
    latency_ms: Optional[float] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class RecentAuditLogOut(BaseModel):
    """Paginated list of recent audit log entries."""
    logs: list[AuditLogOut]
    total: int


# ── Analytics ─────────────────────────────────────────────────────────────────

class AnalyticsSummary(BaseModel):
    total_users: int = 0
    total_documents: int = 0
    total_chunks: int = 0
    total_sessions: int = 0
    total_messages: int = 0
    total_audit_logs: int = 0
    user_token_analytics: list[UserTokenRow] = Field(
        default_factory=list,
        description="Per-user token usage and latency rankings computed via complex SQL CTE + Window functions",
    )


class UserTokenRow(BaseModel):
    """Single row from the CTE + window-function query."""
    usage_rank: int
    username: str
    activity_date: date
    request_count: int
    daily_tokens: int
    cumulative_tokens: int
    token_delta: int
    avg_latency_ms: float
    total_tokens: int
    latency_rank: int


class UserTokenAnalytics(BaseModel):
    """Response for the per-user token analysis endpoint."""
    rows: list[UserTokenRow]
    total_rows: int


class DocumentWeekRow(BaseModel):
    """Single row from the document upload trends query."""
    week_start: date
    documents_uploaded: int
    total_chunks: int
    avg_file_size_kb: float
    min_file_bytes: int
    max_file_bytes: int
    distinct_sources: int


class DocumentUploadTrend(BaseModel):
    """Response for the document upload trends endpoint."""
    weeks: list[DocumentWeekRow]
    total_weeks: int


# ── Error ─────────────────────────────────────────────────────────────────────

class ErrorResponse(BaseModel):
    """Standard error envelope returned by exception handlers."""
    detail: str
    error_type: Optional[str] = None
