"""
Enterprise RAG & SQL Copilot — FastAPI Application Entry Point.

Boots the API server, registers routers, configures CORS, and wires up
global exception handlers for consistent error responses.
"""

import logging
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from database import init_db, engine
from routers import upload, chat, analytics
from schemas import ErrorResponse

import models  # noqa: F401  — ensure all models are registered on Base

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s │ %(levelname)-8s │ %(name)s │ %(message)s",
)
logger = logging.getLogger(__name__)


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Create pgvector extension and all tables on startup."""
    logger.info("Starting up — initialising database …")
    init_db()
    logger.info("Database ready.")
    yield
    logger.info("Shutting down.")


# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Enterprise RAG & SQL Copilot",
    description=(
        "Upload documents, ask natural-language questions, "
        "and get AI-powered answers grounded in your data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)


# ── CORS Middleware ───────────────────────────────────────────────────────────
# In production, replace ["*"] with your actual frontend origins.

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request Timing Middleware ─────────────────────────────────────────────────

@app.middleware("http")
async def add_request_timing(request: Request, call_next):
    """Inject an X-Process-Time header on every response."""
    start = time.perf_counter()
    response = await call_next(request)
    elapsed_ms = (time.perf_counter() - start) * 1000
    response.headers["X-Process-Time-Ms"] = f"{elapsed_ms:.2f}"
    return response


# ── Exception Handlers ───────────────────────────────────────────────────────

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_exception_handler(request: Request, exc: SQLAlchemyError):
    """Catch unhandled database errors and return a clean 500."""
    logger.error("Database error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="A database error occurred. Please try again later.",
            error_type="database_error",
        ).model_dump(),
    )


@app.exception_handler(ValueError)
async def value_error_handler(request: Request, exc: ValueError):
    """Catch ValueErrors raised in business logic and return 422."""
    logger.warning("Validation error on %s %s: %s", request.method, request.url.path, exc)
    return JSONResponse(
        status_code=422,
        content=ErrorResponse(
            detail=str(exc),
            error_type="validation_error",
        ).model_dump(),
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    """Last-resort handler — keeps the API from leaking stack traces."""
    logger.exception("Unhandled exception on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="An unexpected error occurred.",
            error_type="internal_error",
        ).model_dump(),
    )


# ── Routers ──────────────────────────────────────────────────────────────────

app.include_router(upload.router)
app.include_router(chat.router)
app.include_router(analytics.router)


# ── Health Check ─────────────────────────────────────────────────────────────

@app.get("/", tags=["health"])
async def root():
    return {"status": "ok", "service": "Enterprise RAG & SQL Copilot"}


@app.get("/health", tags=["health"])
async def health_check():
    """
    Deep health check — verifies the database connection is alive.
    """
    try:
        from sqlalchemy import text as sa_text
        with engine.connect() as conn:
            conn.execute(sa_text("SELECT 1"))
        db_status = "connected"
    except Exception as e:
        logger.error("Health check DB failure: %s", e)
        db_status = "unavailable"

    return {
        "status": "ok" if db_status == "connected" else "degraded",
        "service": "Enterprise RAG & SQL Copilot",
        "database": db_status,
    }
