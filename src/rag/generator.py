"""
Generator (ROADMAP.md Section 3.4 — local Ollama + Qwen3 by default).

Generator is a Protocol so the RAGPipeline (pipeline.py) never depends on a
concrete provider — swapping OllamaGenerator for a future GeminiGenerator (the
documented free-tier fallback, not yet implemented — see configs/rag.yaml's
generation.fallback block) is a config change, not a pipeline change.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List, Optional, Protocol

from src.rag.config import GenerationConfig
from src.rag.schemas import RetrievedChunk

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """You are a cultural knowledge assistant for the Palestinian Cultural \
Knowledge Platform. Answer strictly and only using the numbered source passages provided \
below — never use outside knowledge, and never invent facts or sources.

Rules:
1. If the passages do not contain enough information to answer, say so plainly instead of \
guessing.
2. Respond in the same language as the question (an Arabic question gets an Arabic answer; \
an English question gets an English answer).
3. After every claim, cite the source(s) it came from using bracketed numbers, e.g. [1] or \
[1][2], matching the numbered sources below. Do not cite a source number that wasn't given \
to you.
4. Be concise and factual."""

_ARABIC_RE = re.compile(r"[؀-ۿ]")


def _looks_arabic(text: str) -> bool:
    return bool(_ARABIC_RE.search(text))


def _format_context(context_chunks: List[RetrievedChunk]) -> str:
    parts = []
    for i, retrieved in enumerate(context_chunks, start=1):
        chunk = retrieved.chunk
        title = chunk.title or "(untitled)"
        url = chunk.source_url or "(no url)"
        parts.append(f"[{i}] Title: {title}\nSource: {url}\nText: {chunk.text}")
    return "\n\n".join(parts)


class Generator(Protocol):
    def generate(self, question: str, context_chunks: List[RetrievedChunk]) -> str: ...


def insufficient_context_message(question: str) -> str:
    """Used by RAGPipeline when retrieval returns nothing at all — no generator
    call needed for an empty context."""
    if _looks_arabic(question):
        return "لا تتوفر لدي معلومات كافية في المصادر المتاحة للإجابة على هذا السؤال."
    return "I don't have enough information in the available sources to answer this question."


class OllamaGenerator:
    """Default generator: a local Ollama server (see ROADMAP.md Section 3.4 for
    why local-first was chosen over Gemini/OpenRouter/Grok)."""

    def __init__(self, config: GenerationConfig, host: Optional[str] = None):
        import ollama

        self.config = config
        resolved_host = host or os.environ.get(config.host_env_var, "http://localhost:11434")
        self.client = ollama.Client(host=resolved_host)
        self._host = resolved_host

    def generate(self, question: str, context_chunks: List[RetrievedChunk]) -> str:
        import ollama as ollama_module

        context_block = _format_context(context_chunks)
        user_message = f"Sources:\n{context_block}\n\nQuestion: {question}"
        try:
            response = self.client.chat(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_message},
                ],
                options={
                    "temperature": self.config.temperature,
                    "num_predict": self.config.max_output_tokens,
                },
            )
        except ollama_module.ResponseError as exc:
            if exc.status_code == 404:
                raise RuntimeError(
                    f"Ollama model '{self.config.model}' is not pulled on {self._host}. "
                    f"Run `ollama pull {self.config.model}` first."
                ) from exc
            raise
        except ConnectionError as exc:
            raise RuntimeError(
                f"Could not reach Ollama at {self._host}. Is it running? "
                f"Install from https://ollama.com and run `ollama serve`, "
                f"or start the app if already installed."
            ) from exc
        return response.message.content or ""
