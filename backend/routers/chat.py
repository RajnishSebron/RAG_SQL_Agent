"""
/api/chat — RAG-powered conversational copilot.

Pipeline per query:
  1. Embed the user question with the same model used at ingestion time.
  2. Run pgvector cosine similarity search to retrieve top-k relevant chunks.
  3. Assemble a grounded prompt (system + context + conversation history).
  4. Call Groq's Llama 3.1 8B Instant via LangChain ``ChatGroq``.
  5. Persist both messages, log audit metrics, and return the response
     with source citations.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from dotenv import load_dotenv

from database import get_db
from schemas import (
    ChatQueryRequest,
    ChatQueryResponse,
    SourceChunk,
    ChatSessionCreate,
    ChatSessionOut,
    ChatMessageOut,
)
import crud

# ── LangChain / Groq ─────────────────────────────────────────────────────────
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage

load_dotenv()
logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/chat", tags=["chat"])

# ── Constants ─────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.1-8b-instant"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"
TOP_K = 5                       # number of chunks to retrieve
SIMILARITY_THRESHOLD = 0.25     # minimum cosine similarity to include a chunk
MAX_CONTEXT_CHARS = 6000        # cap total context length fed to the LLM
MAX_HISTORY_MESSAGES = 10       # conversation turns to include in the prompt

SYSTEM_PROMPT = """\
You are an Enterprise RAG & SQL Copilot — a helpful, accurate AI assistant.

RULES:
1. Answer the user's question using ONLY the provided context chunks.
2. If the context does not contain enough information, say so honestly.
3. Cite sources by referencing the document filename and chunk index
   (e.g. "[report.pdf #3]").
4. Be concise but thorough. Use markdown formatting where helpful.
5. If the user asks you to generate SQL, produce standard PostgreSQL syntax
   and explain what the query does.
"""

# ── Lazy singletons ──────────────────────────────────────────────────────────

_llm: ChatGroq | None = None
_embed_model: HuggingFaceEmbeddings | None = None


def get_llm() -> ChatGroq:
    """Return a cached ChatGroq client (created once)."""
    global _llm
    if _llm is None:
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=503,
                detail="GROQ_API_KEY is not set. Add it to your .env file.",
            )
        _llm = ChatGroq(
            model=GROQ_MODEL,
            api_key=api_key,
            temperature=0.3,
            max_tokens=2048,
        )
        logger.info("Initialised ChatGroq model: %s", GROQ_MODEL)
    return _llm


def get_embed_model() -> HuggingFaceEmbeddings:
    """Return a cached embedding model (same one used at upload time)."""
    global _embed_model
    if _embed_model is None:
        _embed_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.info("Loaded embedding model: %s", EMBEDDING_MODEL)
    return _embed_model


# ── Helpers ───────────────────────────────────────────────────────────────────

def _build_context_block(sources: list[dict[str, Any]]) -> str:
    """Format retrieved chunks into a numbered context block for the prompt."""
    if not sources:
        return "(No relevant documents found in the knowledge base.)"

    parts: list[str] = []
    total = 0
    for s in sources:
        entry = (
            f"[{s['filename']} #{s['chunk_index']}]  "
            f"(similarity: {s['similarity']:.2f})\n"
            f"{s['text']}"
        )
        if total + len(entry) > MAX_CONTEXT_CHARS:
            break
        parts.append(entry)
        total += len(entry)

    return "\n\n---\n\n".join(parts)


def _build_messages(
    context_block: str,
    history: list[dict[str, str]],
    user_query: str,
) -> list:
    """Assemble the full LangChain message list."""
    messages: list = [SystemMessage(content=SYSTEM_PROMPT)]

    # Conversation history (capped)
    for msg in history[-MAX_HISTORY_MESSAGES:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        else:
            messages.append(AIMessage(content=msg["content"]))

    # Current turn: inject context then user question
    augmented = (
        f"### Retrieved Context\n\n{context_block}\n\n"
        f"### User Question\n\n{user_query}"
    )
    messages.append(HumanMessage(content=augmented))
    return messages


# ── Endpoints ─────────────────────────────────────────────────────────────────

# ── Session management (unchanged) ───────────────────────────────────────────

@router.post("/sessions", response_model=ChatSessionOut, status_code=201)
def create_session(
    body: ChatSessionCreate,
    db: Session = Depends(get_db),
):
    """Create a new chat session."""
    return crud.create_session(db, title=body.title or "New Chat")


@router.get("/sessions", response_model=list[ChatSessionOut])
def list_sessions(db: Session = Depends(get_db)):
    """List all chat sessions (newest first)."""
    return crud.list_sessions(db)


@router.get("/sessions/{session_id}", response_model=ChatSessionOut)
def get_session(session_id: uuid.UUID, db: Session = Depends(get_db)):
    """Get a single chat session with its messages."""
    session = crud.get_session(db, session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session


# ── RAG chat endpoint ────────────────────────────────────────────────────────

@router.post(
    "",
    response_model=ChatQueryResponse,
    status_code=200,
    include_in_schema=False,
)
@router.post(
    "/",
    response_model=ChatQueryResponse,
    status_code=200,
    summary="Ask a question (RAG)",
)
def rag_chat(
    body: ChatQueryRequest,
    db: Session = Depends(get_db),
):
    """
    Accept a user query, retrieve relevant document chunks via pgvector
    similarity search, generate an answer with Groq LLM, and return
    the response with source citations and usage metrics.
    """
    t_start = time.perf_counter()

    # ── 0. Resolve or create session ─────────────────────────────────────
    if body.session_id:
        session = crud.get_session(db, body.session_id)
        if not session:
            raise HTTPException(status_code=404, detail="Session not found")
    else:
        session = crud.create_session(db, title=body.query[:60])

    # ── 1. Embed the query ───────────────────────────────────────────────
    embed_model = get_embed_model()
    query_embedding: list[float] = embed_model.embed_query(body.query)

    # ── 2. Similarity search ─────────────────────────────────────────────
    top_k = body.top_k or TOP_K
    raw_results = crud.similarity_search(
        db,
        query_embedding=query_embedding,
        top_k=top_k,
        threshold=SIMILARITY_THRESHOLD,
    )

    sources: list[dict[str, Any]] = []
    for chunk, similarity, doc in raw_results:
        sources.append({
            "chunk_id": str(chunk.id),
            "document_id": str(doc.id),
            "filename": doc.filename,
            "chunk_index": chunk.chunk_index,
            "text": chunk.text,
            "similarity": similarity,
        })

    # ── 3. Build conversation history ────────────────────────────────────
    history: list[dict[str, str]] = [
        {"role": m.role, "content": m.content}
        for m in sorted(session.messages, key=lambda m: m.created_at)
    ]

    # ── 4. Assemble prompt & call Groq ───────────────────────────────────
    context_block = _build_context_block(sources)
    messages = _build_messages(context_block, history, body.query)

    llm = get_llm()
    ai_response = llm.invoke(messages)

    answer_text: str = ai_response.content
    token_usage: int | None = None

    # Extract token usage from the response metadata if available
    if hasattr(ai_response, "response_metadata"):
        meta = ai_response.response_metadata
        usage = meta.get("token_usage") or meta.get("usage", {})
        if usage:
            token_usage = usage.get("total_tokens")

    # ── 5. Persist messages ──────────────────────────────────────────────
    crud.add_message(db, session.id, role="user", content=body.query)
    crud.add_message(db, session.id, role="assistant", content=answer_text)

    # ── 6. Audit log ─────────────────────────────────────────────────────
    latency_ms = (time.perf_counter() - t_start) * 1000
    crud.create_audit_log(
        db,
        action="chat_query",
        token_usage=token_usage,
        latency_ms=round(latency_ms, 2),
    )

    logger.info(
        "RAG query completed in %.0f ms | %d sources | %s tokens",
        latency_ms, len(sources), token_usage or "N/A",
    )

    # ── 7. Build response ────────────────────────────────────────────────
    source_chunks = [
        SourceChunk(
            chunk_id=uuid.UUID(s["chunk_id"]),
            document_id=uuid.UUID(s["document_id"]),
            filename=s["filename"],
            chunk_index=s["chunk_index"],
            text=s["text"][:300],
            similarity=round(s["similarity"], 4),
        )
        for s in sources
    ]

    return ChatQueryResponse(
        session_id=session.id,
        answer=answer_text,
        sources=source_chunks,
        model=GROQ_MODEL,
        token_usage=token_usage,
        latency_ms=round(latency_ms, 2),
    )


# ── Session message send (legacy endpoint, now backed by RAG) ────────────────

@router.post(
    "/sessions/{session_id}/messages",
    response_model=ChatQueryResponse,
    status_code=200,
    summary="Send a message within a session (RAG)",
)
def send_message_in_session(
    session_id: uuid.UUID,
    body: ChatQueryRequest,
    db: Session = Depends(get_db),
):
    """Send a message in an existing session — delegates to the RAG pipeline."""
    body.session_id = session_id
    return rag_chat(body=body, db=db)
