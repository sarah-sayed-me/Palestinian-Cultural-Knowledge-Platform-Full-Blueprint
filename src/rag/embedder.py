"""
Embedding pipeline (ROADMAP.md Section 3.1 — Qwen3-Embedding-0.6B, self-hosted).

Qwen3-Embedding is an asymmetric retrieval model: queries should be encoded with
its "query" instruction prompt, documents should not. Embedder exposes separate
embed_documents()/embed_query() methods so that distinction can never be
accidentally dropped at a call site — mixing them up is a silent quality bug,
not an error, so the API is shaped to make it hard to do by accident.
"""

from __future__ import annotations

import logging
from typing import List

from src.rag.config import EmbeddingConfig

logger = logging.getLogger(__name__)


def _resolve_device(device: str) -> str:
    """Resolve configs/rag.yaml's embedding.device to a concrete torch device.

    "auto" picks cuda if available else cpu. An explicit "cuda" is respected
    but fails loudly (not silently falls back to cpu) if this machine's torch
    build can't actually see a GPU — a silent fallback would hide a real
    performance regression from whoever set device: cuda on purpose.
    """
    import torch

    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "configs/rag.yaml sets embedding.device: cuda, but torch.cuda.is_available() "
            "is False in this environment (often a CPU-only torch build despite a real "
            "GPU being present — check `python -c \"import torch; print(torch.version.cuda)\"`). "
            "Install a CUDA-enabled torch build, or set embedding.device to 'auto' or 'cpu' "
            "in configs/rag.yaml."
        )
    return device


class Embedder:
    def __init__(self, config: EmbeddingConfig):
        from sentence_transformers import SentenceTransformer

        self.config = config
        device = _resolve_device(config.device)
        logger.info("Loading embedding model %s on device=%s", config.model, device)
        self.model = SentenceTransformer(config.model, device=device)

        # sentence-transformers renamed this method; support both without a version pin.
        dim_fn = getattr(self.model, "get_embedding_dimension", None) or self.model.get_sentence_embedding_dimension
        actual_dim = dim_fn()
        if actual_dim != config.dimensions:
            raise RuntimeError(
                f"configs/rag.yaml declares embedding.dimensions: {config.dimensions}, but "
                f"{config.model} actually produces {actual_dim}-dimensional vectors. Fix the "
                f"config — a silent mismatch here would corrupt the pgvector column."
            )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed chunk/document text. No task instruction prefix — this is the
        asymmetric-retrieval "document side"."""
        if not texts:
            return []
        vectors = self.model.encode(
            texts,
            batch_size=self.config.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return vectors.tolist()

    def embed_query(self, text: str) -> List[float]:
        """Embed a search query using the model's query instruction prompt when
        available (Qwen3-Embedding ships one); falls back to plain encoding for
        models that don't define one."""
        try:
            vector = self.model.encode(
                text, prompt_name="query", normalize_embeddings=True, show_progress_bar=False
            )
        except ValueError:
            logger.debug("%s has no 'query' prompt defined; encoding without one.", self.config.model)
            vector = self.model.encode(text, normalize_embeddings=True, show_progress_bar=False)
        return vector.tolist()
