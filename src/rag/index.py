"""
pgvector index build (ROADMAP.md Section 3.2).

Creates the rag_chunks table (if missing) and upserts Chunk rows + their
embeddings. Idempotent: re-running over the same chunks.jsonl updates existing
rows by chunk_id rather than duplicating them, so re-indexing after a
re-chunk or re-embed is always safe to just run again.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, List

from psycopg2.extensions import connection as PGConnection
from psycopg2.extras import execute_values

from src.rag.chunker import DEFAULT_OUTPUT as DEFAULT_CHUNKS_PATH, iter_chunks
from src.rag.config import RagConfig, VectorStoreConfig
from src.rag.db import get_connection
from src.rag.embedder import Embedder
from src.rag.schemas import Chunk

logger = logging.getLogger(__name__)

_COLUMNS = (
    "chunk_id",
    "doc_id",
    "chunk_index",
    "text",
    "token_count",
    "start_char",
    "end_char",
    "chunking_version",
    "embedding_model",
    "embedding_version",
    "title",
    "source_url",
    "credibility",
    "quality_score",
    "seed_category",
    "embedding",
)


def create_schema(conn: PGConnection, config: VectorStoreConfig, dimensions: int) -> None:
    """Create the pgvector extension, table, and ANN index if they don't exist yet."""
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {config.table} (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                text TEXT NOT NULL,
                token_count INTEGER NOT NULL,
                start_char INTEGER NOT NULL,
                end_char INTEGER NOT NULL,
                chunking_version TEXT NOT NULL,
                embedding_model TEXT,
                embedding_version TEXT,
                title TEXT,
                source_url TEXT,
                credibility TEXT,
                quality_score REAL,
                seed_category TEXT,
                embedding vector({dimensions}) NOT NULL
            );
            """
        )
        cur.execute(
            f"""
            CREATE INDEX IF NOT EXISTS {config.table}_embedding_idx
            ON {config.table} USING hnsw (embedding vector_cosine_ops);
            """
        )
        cur.execute(
            f"CREATE INDEX IF NOT EXISTS {config.table}_doc_id_idx ON {config.table} (doc_id);"
        )
    conn.commit()


def _chunk_row(chunk: Chunk, embedding: List[float]) -> tuple[Any, ...]:
    return (
        chunk.chunk_id,
        chunk.doc_id,
        chunk.chunk_index,
        chunk.text,
        chunk.token_count,
        chunk.start_char,
        chunk.end_char,
        chunk.chunking_version,
        chunk.embedding_model,
        chunk.embedding_version,
        chunk.title,
        chunk.source_url,
        chunk.credibility,
        chunk.quality_score,
        chunk.seed_category,
        embedding,
    )


def upsert_chunks(conn: PGConnection, config: VectorStoreConfig, rows: Iterable[tuple[Any, ...]]) -> int:
    rows = list(rows)
    if not rows:
        return 0
    columns_sql = ", ".join(_COLUMNS)
    update_sql = ", ".join(f"{c} = EXCLUDED.{c}" for c in _COLUMNS if c != "chunk_id")
    with conn.cursor() as cur:
        execute_values(
            cur,
            f"""
            INSERT INTO {config.table} ({columns_sql}) VALUES %s
            ON CONFLICT (chunk_id) DO UPDATE SET {update_sql};
            """,
            rows,
        )
    conn.commit()
    return len(rows)


def build_index(
    *,
    chunks_path: Path = DEFAULT_CHUNKS_PATH,
    config: RagConfig | None = None,
    embedder: Embedder | None = None,
) -> dict[str, Any]:
    """Embed every chunk in chunks_path and upsert it into the pgvector table."""
    if not chunks_path.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_path}. Run scripts/chunk_corpus.py first.")

    config = config or RagConfig.load()
    embedder = embedder or Embedder(config.embedding)

    conn = get_connection()
    try:
        create_schema(conn, config.vector_store, config.embedding.dimensions)

        total = 0
        batch: List[Chunk] = []
        for chunk in iter_chunks(chunks_path):
            batch.append(chunk)
            if len(batch) >= config.embedding.batch_size:
                total += _embed_and_upsert(conn, config, embedder, batch)
                batch = []
        if batch:
            total += _embed_and_upsert(conn, config, embedder, batch)

        return {
            "chunks_path": str(chunks_path),
            "table": config.vector_store.table,
            "embedding_model": config.embedding.model,
            "chunks_indexed": total,
        }
    finally:
        conn.close()


def _embed_and_upsert(
    conn: PGConnection, config: RagConfig, embedder: Embedder, batch: List[Chunk]
) -> int:
    vectors = embedder.embed_documents([c.text for c in batch])
    rows = []
    for chunk, vector in zip(batch, vectors):
        stamped = chunk.model_copy(
            update={
                "embedding_model": config.embedding.model,
                "embedding_version": config.embedding.embedding_version,
            }
        )
        rows.append(_chunk_row(stamped, vector))
    return upsert_chunks(conn, config.vector_store, rows)
