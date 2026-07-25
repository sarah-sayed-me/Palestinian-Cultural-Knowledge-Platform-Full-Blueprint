"""
Integration tests against a real Postgres+pgvector instance (docker-compose.yml).

Skipped automatically if Postgres isn't reachable, so `uv run pytest` stays
green without Docker running. Use a small hand-picked-vector fake embedder
(not the real Qwen3-Embedding model) so these tests are fast and deterministic
— they check the DB round-trip and ranking logic, not embedding quality.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.rag.config import RagConfig, VectorStoreConfig
from src.rag.db import get_connection
from src.rag.index import build_index
from src.rag.retriever import Retriever
from src.rag.schemas import Chunk


def _postgres_available() -> bool:
    try:
        conn = get_connection()
        conn.close()
        return True
    except Exception:
        return False


pytestmark = pytest.mark.skipif(
    not _postgres_available(), reason="Postgres not reachable — run `docker compose up -d`"
)


class _FakeEmbedder:
    """Fixed, hand-picked unit vectors for a small set of known strings —
    deterministic and instant, so DB/ranking logic is testable without the
    real embedding model."""

    _VECTORS = {
        "query near A": [1.0, 0.0, 0.0],
        "doc A": [0.99, 0.14, 0.0],
        "doc B": [0.0, 0.99, 0.14],
    }

    def __init__(self):
        self.dimensions = 3

    def embed_documents(self, texts):
        return [self._VECTORS[t] for t in texts]

    def embed_query(self, text):
        return self._VECTORS[text]


def _chunk(chunk_id: str, doc_id: str, text: str, title: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=doc_id,
        chunk_index=0,
        text=text,
        token_count=1,
        start_char=0,
        end_char=len(text),
        chunking_version="test-v1",
        title=title,
    )


@pytest.fixture
def test_config():
    config = replace(
        RagConfig.default(),
        vector_store=VectorStoreConfig(backend="pgvector", table="rag_chunks_pytest", distance="cosine"),
        embedding=replace(RagConfig.default().embedding, dimensions=3),
    )
    yield config
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"DROP TABLE IF EXISTS {config.vector_store.table};")
        conn.commit()
    finally:
        conn.close()


def test_build_index_upsert_is_idempotent(tmp_path, test_config):
    chunks = [_chunk("c1", "d1", "doc A", "A"), _chunk("c2", "d2", "doc B", "B")]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("\n".join(c.model_dump_json() for c in chunks), encoding="utf-8")
    embedder = _FakeEmbedder()

    first = build_index(chunks_path=chunks_path, config=test_config, embedder=embedder)
    second = build_index(chunks_path=chunks_path, config=test_config, embedder=embedder)

    assert first["chunks_indexed"] == 2
    assert second["chunks_indexed"] == 2  # re-run doesn't error

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(f"SELECT count(*) FROM {test_config.vector_store.table};")
            row_count = cur.fetchone()[0]
    finally:
        conn.close()
    assert row_count == 2  # upsert, not duplicate rows


def test_retriever_ranks_by_similarity(tmp_path, test_config):
    chunks = [_chunk("c1", "d1", "doc A", "A"), _chunk("c2", "d2", "doc B", "B")]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("\n".join(c.model_dump_json() for c in chunks), encoding="utf-8")
    embedder = _FakeEmbedder()
    build_index(chunks_path=chunks_path, config=test_config, embedder=embedder)

    conn = get_connection()
    try:
        retriever = Retriever(conn, embedder, test_config)
        results = retriever.retrieve("query near A", top_k=2)

        assert len(results) == 2
        assert results[0].chunk.doc_id == "d1"  # "doc A" vector is closer to the query
        assert results[0].score > results[1].score
    finally:
        conn.close()


def test_retriever_top_k_limits_results(tmp_path, test_config):
    chunks = [_chunk("c1", "d1", "doc A", "A"), _chunk("c2", "d2", "doc B", "B")]
    chunks_path = tmp_path / "chunks.jsonl"
    chunks_path.write_text("\n".join(c.model_dump_json() for c in chunks), encoding="utf-8")
    embedder = _FakeEmbedder()
    build_index(chunks_path=chunks_path, config=test_config, embedder=embedder)

    conn = get_connection()
    try:
        retriever = Retriever(conn, embedder, test_config)
        results = retriever.retrieve("query near A", top_k=1)
        assert len(results) == 1
    finally:
        conn.close()
