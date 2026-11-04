"""
/api/upload — Document ingestion pipeline.

Full lifecycle:
  1. Accept a multipart file upload (PDF, TXT, CSV, DOCX).
  2. Extract raw text (PyPDF2 for PDFs, plain decode for text formats).
  3. Split the text into overlapping chunks (LangChain RecursiveCharacterTextSplitter).
  4. Generate vector embeddings for each chunk (HuggingFace sentence-transformers).
  5. Persist the document row + chunk rows (with embeddings) via pgvector.
  6. Log the operation in the audit_logs table.
"""

from __future__ import annotations

import io
import logging
import time
import uuid
from typing import Any

from fastapi import APIRouter, Depends, UploadFile, File, HTTPException, Query
from sqlalchemy.orm import Session

from database import get_db
from schemas import DocumentOut, DocumentUploadResponse, DocumentChunkOut, DocumentDetailOut
import crud

# ── LangChain text splitter ──────────────────────────────────────────────────
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── Embedding model (lazy-loaded singleton) ──────────────────────────────────
from langchain_community.embeddings import HuggingFaceEmbeddings

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/upload", tags=["upload"])

# ── Constants ─────────────────────────────────────────────────────────────────

ALLOWED_CONTENT_TYPES: dict[str, str] = {
    "text/plain": "txt",
    "text/csv": "csv",
    "application/pdf": "pdf",
    "application/vnd.openxmlformats-officedocument.wordprocessingml.document": "docx",
}

CHUNK_SIZE = 1000          # characters per chunk
CHUNK_OVERLAP = 200        # overlap between consecutive chunks
EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # 384-dim, fast & free
EMBEDDING_DIM = 384        # must match the model output dimension

MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB

# ── Lazy singletons ──────────────────────────────────────────────────────────

_embeddings_model: HuggingFaceEmbeddings | None = None
_text_splitter: RecursiveCharacterTextSplitter | None = None


def get_embeddings_model() -> HuggingFaceEmbeddings:
    """Return a cached embedding model instance (loaded once on first call)."""
    global _embeddings_model
    if _embeddings_model is None:
        logger.info("Loading embedding model '%s' …", EMBEDDING_MODEL)
        _embeddings_model = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _embeddings_model


def get_text_splitter() -> RecursiveCharacterTextSplitter:
    """Return a cached text splitter instance."""
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
    return _text_splitter


# ── Text extraction helpers ──────────────────────────────────────────────────

def _extract_text_from_pdf(raw_bytes: bytes) -> str:
    """Extract text from a PDF using PyPDF2."""
    try:
        from PyPDF2 import PdfReader
    except ImportError:
        raise HTTPException(
            status_code=501,
            detail="PyPDF2 is not installed. Run: pip install PyPDF2",
        )

    reader = PdfReader(io.BytesIO(raw_bytes))
    pages: list[str] = []
    for page_num, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            pages.append(text)

    if not pages:
        raise HTTPException(
            status_code=422,
            detail="Could not extract any text from the PDF. "
                   "The file may be image-based (scanned). "
                   "Consider using OCR before uploading.",
        )
    return "\n\n".join(pages)


def _extract_text_from_bytes(raw_bytes: bytes, content_type: str) -> str:
    """Route to the correct text extractor based on MIME type."""
    if content_type == "application/pdf":
        return _extract_text_from_pdf(raw_bytes)

    # TXT, CSV, DOCX-as-text fallback
    return raw_bytes.decode("utf-8", errors="replace")


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post(
    "/",
    response_model=DocumentUploadResponse,
    status_code=201,
    summary="Upload & ingest a document",
)
async def upload_document(
    file: UploadFile = File(..., description="PDF, TXT, CSV, or DOCX file"),
    chunk_size: int = Query(CHUNK_SIZE, ge=100, le=8000, description="Characters per chunk"),
    chunk_overlap: int = Query(CHUNK_OVERLAP, ge=0, le=2000, description="Overlap between chunks"),
    db: Session = Depends(get_db),
):
    """
    Upload a document and run the full ingestion pipeline:

    1. **Validate** file type and size.
    2. **Extract** text content (PDF → PyPDF2, others → UTF-8 decode).
    3. **Chunk** the text with LangChain's `RecursiveCharacterTextSplitter`.
    4. **Embed** each chunk using a HuggingFace sentence-transformer model.
    5. **Store** the document metadata + chunk embeddings in PostgreSQL / pgvector.
    6. **Audit-log** the operation with timing information.
    """
    t_start = time.perf_counter()

    # ── 1. Validate ──────────────────────────────────────────────────────
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail=(
                f"Unsupported content type '{file.content_type}'. "
                f"Allowed: {', '.join(sorted(ALLOWED_CONTENT_TYPES.values()))}"
            ),
        )

    raw_bytes = await file.read()

    if len(raw_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    if len(raw_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds the {MAX_FILE_SIZE // (1024*1024)} MB limit.",
        )

    # ── 2. Extract text ──────────────────────────────────────────────────
    raw_text = _extract_text_from_bytes(raw_bytes, file.content_type)

    if not raw_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No extractable text found in the uploaded file.",
        )

    # ── 3. Chunk ─────────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunk_texts: list[str] = splitter.split_text(raw_text)

    if not chunk_texts:
        raise HTTPException(
            status_code=422,
            detail="Text splitting produced zero chunks.",
        )

    logger.info(
        "Document '%s': extracted %d chars → %d chunks (size=%d, overlap=%d)",
        file.filename, len(raw_text), len(chunk_texts), chunk_size, chunk_overlap,
    )

    # ── 4. Embed ─────────────────────────────────────────────────────────
    model = get_embeddings_model()
    embeddings: list[list[float]] = model.embed_documents(chunk_texts)

    # ── 5. Persist ───────────────────────────────────────────────────────
    file_type = ALLOWED_CONTENT_TYPES.get(file.content_type, "unknown")
    metadata: dict[str, Any] = {
        "content_type": file.content_type,
        "file_type": file_type,
        "total_chars": len(raw_text),
        "total_chunks": len(chunk_texts),
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        "embedding_model": EMBEDDING_MODEL,
        "embedding_dim": EMBEDDING_DIM,
    }

    doc = crud.create_document(
        db,
        filename=file.filename or "unnamed",
        file_size=len(raw_bytes),
        metadata_json=metadata,
    )

    chunk_records = [
        {"text": text, "embedding": emb}
        for text, emb in zip(chunk_texts, embeddings)
    ]
    saved_chunks = crud.add_chunks(db, doc.id, chunk_records)

    # ── 6. Audit log ─────────────────────────────────────────────────────
    latency_ms = (time.perf_counter() - t_start) * 1000
    crud.create_audit_log(
        db,
        action="document_upload",
        latency_ms=round(latency_ms, 2),
    )

    logger.info(
        "Ingested '%s' → %d chunks in %.0f ms",
        file.filename, len(saved_chunks), latency_ms,
    )

    # ── Build response ───────────────────────────────────────────────────
    preview_chunks = [
        DocumentChunkOut(
            id=c.id,
            chunk_index=c.chunk_index,
            text=c.text[:200],   # truncate for preview
        )
        for c in saved_chunks[:5]
    ]

    return DocumentUploadResponse(
        id=doc.id,
        filename=doc.filename,
        file_size=doc.file_size,
        uploaded_at=doc.uploaded_at,
        metadata_json=doc.metadata_json,
        total_chunks=len(saved_chunks),
        chunks_preview=preview_chunks,
    )


@router.get(
    "/",
    response_model=list[DocumentOut],
    summary="List all documents",
)
def list_documents(db: Session = Depends(get_db)):
    """Return all uploaded documents (newest first)."""
    return crud.list_documents(db)


@router.get(
    "/{doc_id}",
    response_model=DocumentDetailOut,
    summary="Get document with chunks",
)
def get_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    """Return full document metadata and all its chunk texts."""
    doc = crud.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    chunk_previews = [
        DocumentChunkOut(
            id=c.id,
            chunk_index=c.chunk_index,
            text=c.text[:200],
        )
        for c in sorted(doc.chunks, key=lambda c: c.chunk_index)
    ]

    return DocumentDetailOut(
        id=doc.id,
        filename=doc.filename,
        file_size=doc.file_size,
        uploaded_at=doc.uploaded_at,
        metadata_json=doc.metadata_json,
        chunks=chunk_previews,
    )


@router.delete(
    "/{doc_id}",
    status_code=204,
    summary="Delete a document and its chunks",
)
def delete_document(doc_id: uuid.UUID, db: Session = Depends(get_db)):
    """Delete a document and all its associated chunks (CASCADE)."""
    doc = crud.get_document(db, doc_id)
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")
    crud.delete_document(db, doc_id)
