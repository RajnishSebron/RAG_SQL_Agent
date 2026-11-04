"""
/api/analytics — System usage metrics and advanced analytics.

Serves the frontend dashboard with aggregate statistics and runs
the complex SQL queries from ``complex_queries.sql`` to surface
per-user token analysis and document upload trends.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import APIRouter, Depends, Query
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import get_db
from schemas import (
    AnalyticsSummary,
    UserTokenAnalytics,
    UserTokenRow,
    DocumentUploadTrend,
    DocumentWeekRow,
    RecentAuditLogOut,
    AuditLogOut,
)
import crud

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/analytics", tags=["analytics"])


# ── 1. Complex Query: Per-user token analysis (CTE + Window Functions) ──────

USER_TOKEN_QUERY = text("""
WITH daily_user_stats AS (
    SELECT
        u.id                                          AS user_id,
        u.username,
        DATE(a.created_at)                            AS activity_date,
        COUNT(a.id)                                   AS request_count,
        COALESCE(SUM(a.token_usage), 0)               AS daily_tokens,
        ROUND(AVG(a.latency_ms)::numeric, 2)          AS avg_latency_ms
    FROM users u
    INNER JOIN audit_logs a ON a.user_id = u.id
    GROUP BY u.id, u.username, DATE(a.created_at)
),
ranked AS (
    SELECT
        *,
        SUM(daily_tokens) OVER (PARTITION BY user_id)        AS total_tokens,
        SUM(daily_tokens) OVER (
            PARTITION BY user_id
            ORDER BY activity_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                     AS cumulative_tokens,
        daily_tokens - LAG(daily_tokens, 1, 0) OVER (
            PARTITION BY user_id ORDER BY activity_date
        )                                                     AS token_delta,
        ROW_NUMBER() OVER (
            ORDER BY SUM(daily_tokens) OVER (PARTITION BY user_id) DESC
        )                                                     AS usage_rank,
        RANK() OVER (
            ORDER BY AVG(avg_latency_ms) OVER (PARTITION BY user_id) DESC
        )                                                     AS latency_rank
    FROM daily_user_stats
)
SELECT
    usage_rank,
    username,
    activity_date,
    request_count,
    daily_tokens,
    cumulative_tokens,
    token_delta,
    avg_latency_ms,
    total_tokens,
    latency_rank
FROM ranked
ORDER BY usage_rank, activity_date
LIMIT :row_limit
""")


# ── 2. Primary Analytics Endpoint: GET /api/analytics ────────────────────────

@router.get(
    "",
    response_model=AnalyticsSummary,
    summary="Dashboard metrics & complex SQL token analytics",
    include_in_schema=False,
)
@router.get(
    "/",
    response_model=AnalyticsSummary,
    summary="Dashboard metrics & complex SQL token analytics",
)
def get_analytics(
    limit: int = Query(25, ge=1, le=500, description="Max user ranking rows to include"),
    db: Session = Depends(get_db),
):
    """
    Executes the Step 2 complex SQL query (CTE + Window Functions) to analyze
    per-user token consumption and returns complete system usage metrics for
    the frontend dashboard.
    """
    # 1. Base counts
    stats = crud.get_analytics(db)

    # 2. Execute complex SQL CTE + Window functions
    result = db.execute(USER_TOKEN_QUERY, {"row_limit": limit})
    rows = [
        UserTokenRow(
            usage_rank=r.usage_rank,
            username=r.username,
            activity_date=r.activity_date,
            request_count=r.request_count,
            daily_tokens=r.daily_tokens,
            cumulative_tokens=r.cumulative_tokens,
            token_delta=r.token_delta,
            avg_latency_ms=float(r.avg_latency_ms) if r.avg_latency_ms else 0.0,
            total_tokens=r.total_tokens,
            latency_rank=r.latency_rank,
        )
        for r in result
    ]

    return AnalyticsSummary(
        total_users=stats.get("total_users", 0),
        total_documents=stats.get("total_documents", 0),
        total_chunks=stats.get("total_chunks", 0),
        total_sessions=stats.get("total_sessions", 0),
        total_messages=stats.get("total_messages", 0),
        total_audit_logs=stats.get("total_audit_logs", 0),
        user_token_analytics=rows,
    )


# ── 3. Dedicated Per-user token analysis (CTE + Window Functions) ────────────


@router.get(
    "/user-tokens",
    response_model=UserTokenAnalytics,
    summary="Per-user token usage (CTE + window functions)",
)
def user_token_analytics(
    limit: int = Query(100, ge=1, le=1000, description="Max rows to return"),
    db: Session = Depends(get_db),
):
    """
    Execute the CTE + window-function query from ``complex_queries.sql``
    to analyse per-user daily token consumption with rankings.
    """
    result = db.execute(USER_TOKEN_QUERY, {"row_limit": limit})
    rows = [
        UserTokenRow(
            usage_rank=r.usage_rank,
            username=r.username,
            activity_date=r.activity_date,
            request_count=r.request_count,
            daily_tokens=r.daily_tokens,
            cumulative_tokens=r.cumulative_tokens,
            token_delta=r.token_delta,
            avg_latency_ms=float(r.avg_latency_ms) if r.avg_latency_ms else 0.0,
            total_tokens=r.total_tokens,
            latency_rank=r.latency_rank,
        )
        for r in result
    ]
    return UserTokenAnalytics(rows=rows, total_rows=len(rows))


# ── 3. Document upload trends (JOINs + GROUP BY / HAVING) ────────────────────

DOC_TREND_QUERY = text("""
SELECT
    DATE_TRUNC('week', d.uploaded_at)::date                  AS week_start,
    COUNT(DISTINCT d.id)                                     AS documents_uploaded,
    COALESCE(SUM(dc.chunk_count), 0)                         AS total_chunks,
    ROUND(AVG(d.file_size)::numeric / 1024, 2)               AS avg_file_size_kb,
    MIN(d.file_size)                                         AS min_file_bytes,
    MAX(d.file_size)                                         AS max_file_bytes,
    COUNT(DISTINCT d.metadata_json ->> 'source')             AS distinct_sources
FROM documents d
LEFT JOIN (
    SELECT document_id, COUNT(*) AS chunk_count
    FROM document_chunks
    GROUP BY document_id
) dc ON dc.document_id = d.id
GROUP BY DATE_TRUNC('week', d.uploaded_at)
HAVING COUNT(DISTINCT d.id) >= :min_docs
ORDER BY week_start DESC
LIMIT :row_limit
""")


@router.get(
    "/doc-trends",
    response_model=DocumentUploadTrend,
    summary="Weekly document upload trends (JOINs + aggregations)",
)
def document_upload_trends(
    min_docs: int = Query(1, ge=1, description="Minimum documents per week to include"),
    limit: int = Query(52, ge=1, le=200, description="Max weeks to return"),
    db: Session = Depends(get_db),
):
    """
    Execute the complex JOIN + GROUP BY / HAVING query to summarise
    document uploads by week with chunk counts and file-size stats.
    """
    result = db.execute(DOC_TREND_QUERY, {"min_docs": min_docs, "row_limit": limit})
    rows = [
        DocumentWeekRow(
            week_start=r.week_start,
            documents_uploaded=r.documents_uploaded,
            total_chunks=r.total_chunks,
            avg_file_size_kb=float(r.avg_file_size_kb) if r.avg_file_size_kb else 0.0,
            min_file_bytes=r.min_file_bytes,
            max_file_bytes=r.max_file_bytes,
            distinct_sources=r.distinct_sources,
        )
        for r in result
    ]
    return DocumentUploadTrend(weeks=rows, total_weeks=len(rows))


# ── 4. Recent audit logs ─────────────────────────────────────────────────────

@router.get(
    "/recent-activity",
    response_model=RecentAuditLogOut,
    summary="Recent audit log entries",
)
def recent_activity(
    limit: int = Query(50, ge=1, le=500, description="Number of entries"),
    action: Optional[str] = Query(None, description="Filter by action type"),
    db: Session = Depends(get_db),
):
    """Return the most recent audit log entries, optionally filtered by action."""
    logs = crud.list_audit_logs(db, limit=limit, action_filter=action)
    items = [
        AuditLogOut(
            id=l.id,
            user_id=l.user_id,
            action=l.action,
            token_usage=l.token_usage,
            latency_ms=l.latency_ms,
            created_at=l.created_at,
        )
        for l in logs
    ]
    return RecentAuditLogOut(logs=items, total=len(items))
