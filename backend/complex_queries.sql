-- ============================================================================
--  Enterprise RAG & SQL Copilot — Advanced PostgreSQL Queries
-- ============================================================================
--  This file contains three production-grade queries that demonstrate
--  CTEs, window functions, complex JOINs with aggregations, and a stored
--  procedure with supporting indexes for high-performance retrieval.
-- ============================================================================


-- ────────────────────────────────────────────────────────────────────────────
--  QUERY 1 — CTE + Window Functions
--  Analyse per-user token consumption and rank users by total usage.
-- ────────────────────────────────────────────────────────────────────────────
--  Uses:
--    • CTE (WITH)          — pre-aggregate per-user daily stats
--    • ROW_NUMBER()        — rank users by total tokens (1 = heaviest)
--    • RANK()              — detect ties in average latency
--    • LAG()               — show day-over-day token change
--    • SUM() OVER          — running cumulative token total per user
-- ────────────────────────────────────────────────────────────────────────────

WITH daily_user_stats AS (
    -- Aggregate token usage and latency per user per day
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
        -- Total tokens across all days for this user
        SUM(daily_tokens) OVER (PARTITION BY user_id)        AS total_tokens,
        -- Running cumulative tokens per user ordered by date
        SUM(daily_tokens) OVER (
            PARTITION BY user_id
            ORDER BY activity_date
            ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW
        )                                                     AS cumulative_tokens,
        -- Day-over-day change in token usage
        daily_tokens - LAG(daily_tokens, 1, 0) OVER (
            PARTITION BY user_id ORDER BY activity_date
        )                                                     AS token_delta,
        -- Global ranking by total tokens (1 = heaviest consumer)
        ROW_NUMBER() OVER (
            ORDER BY SUM(daily_tokens) OVER (PARTITION BY user_id) DESC
        )                                                     AS usage_rank,
        -- Rank by average latency (ties get same rank)
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
ORDER BY usage_rank, activity_date;


-- ────────────────────────────────────────────────────────────────────────────
--  QUERY 2 — Complex JOINs + GROUP BY / HAVING
--  Summarise document uploads by weekly buckets, including chunk counts
--  and average file size, filtering to only "interesting" weeks.
-- ────────────────────────────────────────────────────────────────────────────
--  Uses:
--    • LEFT JOIN            — include documents even if no chunks yet
--    • DATE_TRUNC           — bucket uploads into ISO weeks
--    • GROUP BY + HAVING    — keep only weeks with ≥ 5 documents
--    • Aggregate functions  — COUNT, SUM, AVG, MIN, MAX
--    • JSONB operator       — extract a metadata field for grouping
-- ────────────────────────────────────────────────────────────────────────────

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
    -- Pre-aggregate chunk counts per document to avoid double-counting
    SELECT
        document_id,
        COUNT(*) AS chunk_count
    FROM document_chunks
    GROUP BY document_id
) dc ON dc.document_id = d.id
GROUP BY DATE_TRUNC('week', d.uploaded_at)
HAVING
    COUNT(DISTINCT d.id) >= 5              -- at least 5 documents that week
    AND COALESCE(SUM(dc.chunk_count), 0) > 0  -- at least some chunks generated
ORDER BY week_start DESC;


-- ────────────────────────────────────────────────────────────────────────────
--  QUERY 3 — Stored Procedure + Optimised Indexes
--  High-performance semantic search: find the top-k most similar document
--  chunks for a given query embedding, enriched with document metadata.
-- ────────────────────────────────────────────────────────────────────────────

-- 3a) Supporting indexes (idempotent — safe to re-run) ────────────────────

-- IVFFlat index for fast approximate cosine similarity on chunk embeddings.
-- Tune `lists` based on dataset size: lists ≈ sqrt(row_count).
CREATE INDEX IF NOT EXISTS ix_chunk_embedding_cosine
    ON document_chunks
    USING ivfflat (embedding vector_cosine_ops)
    WITH (lists = 100);

-- B-tree index on document FK for fast JOIN lookups from chunks → documents.
CREATE INDEX IF NOT EXISTS ix_chunk_document_id
    ON document_chunks (document_id);

-- Covering index on audit_logs for the CTE query above.
CREATE INDEX IF NOT EXISTS ix_audit_user_created
    ON audit_logs (user_id, created_at)
    INCLUDE (token_usage, latency_ms);

-- Partial index: only index audit rows that consumed tokens (skip nulls).
CREATE INDEX IF NOT EXISTS ix_audit_token_nonnull
    ON audit_logs (token_usage)
    WHERE token_usage IS NOT NULL;

-- B-tree index on documents.uploaded_at for date-range queries.
CREATE INDEX IF NOT EXISTS ix_doc_uploaded
    ON documents (uploaded_at DESC);


-- 3b) Stored procedure: semantic_search ───────────────────────────────────

CREATE OR REPLACE FUNCTION semantic_search(
    query_embedding  vector(384),    -- the embedding of the user's question (all-MiniLM-L6-v2)
    match_count      int DEFAULT 5,  -- how many results to return
    similarity_floor float DEFAULT 0.0  -- minimum cosine similarity threshold
)
RETURNS TABLE (
    chunk_id         uuid,
    document_id      uuid,
    chunk_index      int,
    chunk_text       text,
    filename         text,
    file_size        bigint,
    metadata         jsonb,
    cosine_similarity float
)
LANGUAGE sql
STABLE              -- safe for read-only replicas
PARALLEL SAFE       -- PostgreSQL may parallelise this
AS $$
    SELECT
        dc.id                                         AS chunk_id,
        d.id                                          AS document_id,
        dc.chunk_index,
        dc.text                                       AS chunk_text,
        d.filename,
        d.file_size,
        d.metadata_json                               AS metadata,
        1 - (dc.embedding <=> query_embedding)        AS cosine_similarity
    FROM document_chunks dc
    INNER JOIN documents d ON d.id = dc.document_id
    WHERE dc.embedding IS NOT NULL
      AND 1 - (dc.embedding <=> query_embedding) >= similarity_floor
    ORDER BY dc.embedding <=> query_embedding   -- uses the IVFFlat index
    LIMIT match_count;
$$;

-- Example call (replace the placeholder with an actual 1536-dim vector):
--
--   SELECT * FROM semantic_search(
--       query_embedding  := '[0.012, -0.034, …]'::vector(384),
--       match_count      := 10,
--       similarity_floor := 0.72
--   );

