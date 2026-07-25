"""
RAG pipeline configuration.

Mirrors the QualityConfig pattern in src/ingestion/quality_checker.py: one
frozen dataclass per configs/rag.yaml section, a zero-argument `default()` for
filesystem-free use (tests, ad-hoc scripts), and `load()` for the real YAML.
See ROADMAP.md Section 3 for the reasoning behind each default value.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import yaml

DEFAULT_RAG_CONFIG_PATH = Path("configs/rag.yaml")


@dataclass(frozen=True)
class ChunkingConfig:
    strategy: str = "recursive-sentence-aware"
    chunking_version: str = "recursive-500-v1"
    target_tokens: int = 500
    overlap_tokens: int = 75


@dataclass(frozen=True)
class EmbeddingConfig:
    provider: str = "sentence-transformers"
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    embedding_version: str = "qwen3-embedding-0.6b-v1"
    dimensions: int = 1024
    batch_size: int = 32
    device: str = "auto"  # "auto" | "cpu" | "cuda"


@dataclass(frozen=True)
class VectorStoreConfig:
    backend: str = "pgvector"
    table: str = "rag_chunks"
    distance: str = "cosine"


@dataclass(frozen=True)
class RetrievalConfig:
    top_k: int = 5
    min_credibility_tier: Optional[str] = None


@dataclass(frozen=True)
class GenerationConfig:
    provider: str = "ollama"
    model: str = "qwen3:14b"
    host_env_var: str = "OLLAMA_HOST"
    temperature: float = 0.2
    max_output_tokens: int = 1024
    fallback_provider: Optional[str] = None
    fallback_model: Optional[str] = None
    fallback_note: Optional[str] = None


@dataclass(frozen=True)
class RagConfig:
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    vector_store: VectorStoreConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

    @classmethod
    def default(cls) -> "RagConfig":
        return cls(
            chunking=ChunkingConfig(),
            embedding=EmbeddingConfig(),
            vector_store=VectorStoreConfig(),
            retrieval=RetrievalConfig(),
            generation=GenerationConfig(),
        )

    @classmethod
    def load(cls, path: Path = DEFAULT_RAG_CONFIG_PATH) -> "RagConfig":
        """Load configs/rag.yaml, falling back to defaults for any missing key or section."""
        if not path.exists():
            return cls.default()
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        d = cls.default()
        chunking = raw.get("chunking", {})
        embedding = raw.get("embedding", {})
        vector_store = raw.get("vector_store", {})
        retrieval = raw.get("retrieval", {})
        generation = raw.get("generation", {})
        fallback = generation.get("fallback", {})

        return cls(
            chunking=ChunkingConfig(
                strategy=chunking.get("strategy", d.chunking.strategy),
                chunking_version=chunking.get("chunking_version", d.chunking.chunking_version),
                target_tokens=chunking.get("target_tokens", d.chunking.target_tokens),
                overlap_tokens=chunking.get("overlap_tokens", d.chunking.overlap_tokens),
            ),
            embedding=EmbeddingConfig(
                provider=embedding.get("provider", d.embedding.provider),
                model=embedding.get("model", d.embedding.model),
                embedding_version=embedding.get("embedding_version", d.embedding.embedding_version),
                dimensions=embedding.get("dimensions", d.embedding.dimensions),
                batch_size=embedding.get("batch_size", d.embedding.batch_size),
                device=embedding.get("device", d.embedding.device),
            ),
            vector_store=VectorStoreConfig(
                backend=vector_store.get("backend", d.vector_store.backend),
                table=vector_store.get("table", d.vector_store.table),
                distance=vector_store.get("distance", d.vector_store.distance),
            ),
            retrieval=RetrievalConfig(
                top_k=retrieval.get("top_k", d.retrieval.top_k),
                min_credibility_tier=retrieval.get(
                    "min_credibility_tier", d.retrieval.min_credibility_tier
                ),
            ),
            generation=GenerationConfig(
                provider=generation.get("provider", d.generation.provider),
                model=generation.get("model", d.generation.model),
                host_env_var=generation.get("host_env_var", d.generation.host_env_var),
                temperature=generation.get("temperature", d.generation.temperature),
                max_output_tokens=generation.get("max_output_tokens", d.generation.max_output_tokens),
                fallback_provider=fallback.get("provider", d.generation.fallback_provider),
                fallback_model=fallback.get("model", d.generation.fallback_model),
                fallback_note=fallback.get("note", d.generation.fallback_note),
            ),
        )
