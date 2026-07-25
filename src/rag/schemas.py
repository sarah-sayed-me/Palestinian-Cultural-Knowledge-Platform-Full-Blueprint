"""
Chunk schema for the RAG pipeline.

A Chunk is a retrieval-sized slice of one DocumentMetadata.text, carrying enough
denormalized source metadata (title, url, credibility, quality, seed_category) that
the retriever can filter and cite without a join back to the document store.

DocumentMetadata (src/ingestion/schemas.py) is the single source of truth for a
document and is not extended further for RAG — Chunk is a sibling model that
references it by doc_id. See ROADMAP.md Section 2.
"""

from __future__ import annotations

import hashlib
from typing import Optional

from pydantic import BaseModel, Field

from src.ingestion.schemas import CredibilityTier


class Chunk(BaseModel):
    chunk_id: str = Field(
        description="SHA-256 of (doc_id + chunk_index + chunking_version). "
        "Stable for the same document/chunker; changes if either does."
    )
    doc_id: str = Field(description="DocumentMetadata.doc_id this chunk was cut from")
    chunk_index: int = Field(description="0-based position of this chunk within its document")

    text: str
    token_count: int
    start_char: int = Field(description="Offset of this chunk's start in the source document text")
    end_char: int = Field(description="Offset of this chunk's end in the source document text")

    chunking_version: str = Field(
        description="Identifies the chunking strategy/parameters used, e.g. 'recursive-500-v1'. "
        "Bump this on any strategy change instead of mutating existing chunks — "
        "re-chunking becomes a new row batch, not a migration."
    )
    embedding_model: Optional[str] = Field(
        default=None, description="Model name used to embed this chunk, e.g. 'Qwen3-Embedding-0.6B'"
    )
    embedding_version: Optional[str] = None

    # Denormalized from DocumentMetadata so retrieval can filter/cite without a join.
    title: Optional[str] = None
    source_url: Optional[str] = None
    credibility: Optional[CredibilityTier] = None
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    seed_category: Optional[str] = None

    model_config = {"use_enum_values": True}


def make_chunk_id(doc_id: str, chunk_index: int, chunking_version: str) -> str:
    """Stable chunk id: same document + same chunker version always yields the same id."""
    fingerprint = f"{doc_id}:{chunk_index}:{chunking_version}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


class RetrievedChunk(BaseModel):
    """A Chunk returned by the retriever, carrying its similarity score for this query."""

    chunk: Chunk
    score: float = Field(description="Similarity score for the query that produced this result, higher is better")
