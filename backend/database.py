"""
Database connection, session management, and pgvector extension setup.

Configures the SQLAlchemy engine, scoped session factory, and declarative
Base class.  On first connection the ``vector`` extension is created if it
does not already exist so that pgvector types are available to all models.
"""

import os
from dotenv import load_dotenv

from sqlalchemy import create_engine, event, text
from sqlalchemy.orm import sessionmaker, declarative_base

load_dotenv()

# ── Connection URL ────────────────────────────────────────────────────────────
DATABASE_URL: str = os.getenv(
    "DATABASE_URL",
    "postgresql://postgres:postgres@localhost:5432/rag_copilot",
)

# ── Engine & Session ──────────────────────────────────────────────────────────
engine = create_engine(
    DATABASE_URL,
    pool_pre_ping=True,       # reconnect stale connections automatically
    pool_size=10,              # max persistent connections in the pool
    max_overflow=20,           # extra connections allowed beyond pool_size
    echo=False,                # set True for SQL debug logging
)

SessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=engine,
)

Base = declarative_base()


# ── pgvector extension bootstrap ─────────────────────────────────────────────
@event.listens_for(engine, "connect")
def _create_pgvector_extension(dbapi_connection, connection_record):
    """Ensure the ``vector`` extension exists on every new raw connection."""
    cursor = dbapi_connection.cursor()
    cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cursor.execute("COMMIT")
    cursor.close()


# ── FastAPI dependency ────────────────────────────────────────────────────────
def get_db():
    """
    Yield a database session for a single request lifecycle.

    Usage::

        @router.get("/items")
        def read_items(db: Session = Depends(get_db)):
            ...
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db() -> None:
    """
    Create all tables defined on ``Base.metadata``.

    Safe to call repeatedly — SQLAlchemy's ``create_all`` is a no-op for
    tables that already exist.
    """
    Base.metadata.create_all(bind=engine)
