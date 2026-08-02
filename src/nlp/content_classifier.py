"""
Cultural/content classification (Track F2).

ROADMAP.md's original design: fine-tune AraBERT on ~1,000-1,500 human-reviewed,
LLM-assisted-labeled examples for the ContentCategory enum. That's a real
training pipeline (labeling UI or workflow, a training run, a held-out eval
set) that doesn't fit inside this session — flagged here explicitly as a
deliberate deviation, not a silent scope cut.

What's built instead: the same local LLM (qwen3 via Ollama) already used for
RAG generation (Track B) and relation extraction (Track E3) as a zero-shot
classifier — one prompt per document, asked to pick exactly one
ContentCategory and a confidence. This is cheaper to stand up, needs no
labeled training set, and reuses infrastructure the project already has
running. It is NOT a replacement for the fine-tuned-AraBERT plan long-term:
a zero-shot LLM call per document doesn't scale as cheaply as a small
fine-tuned classifier once the corpus is large, and its accuracy hasn't been
benchmarked against a real held-out set the way NER/retrieval/RAG were in
Track C — see eval/gold/content_classification_gold.json once built. Revisit
AraBERT fine-tuning if/when there's a labeled set to train on (this
classifier's own high-confidence outputs are a plausible bootstrap source
for one) or the corpus grows enough that per-document LLM calls stop being
the cheap option.

Reuses the same think=False fix Track E3 needed for qwen3: without it, the
model's hidden <think> block can consume the whole token budget and return
empty content — confirmed directly while building relation extraction.
"""

from __future__ import annotations

import json
import re
from typing import List, Optional, Tuple

from src.ingestion.schemas import ContentCategory
from src.rag.config import GenerationConfig

_CATEGORY_VALUES = [c.value for c in ContentCategory if c != ContentCategory.UNCATEGORIZED]

SYSTEM_PROMPT = f"""You are a content classifier for a Palestinian cultural knowledge corpus. \
Given a document's title and text, choose the SINGLE category that best describes its \
PRIMARY subject. Respond with ONLY a JSON object, no other text:
{{"category": "<one_of_the_categories_below>", "confidence": <0.0-1.0>}}

Categories: {", ".join(_CATEGORY_VALUES)}

Rules:
- Pick exactly one category — the one the document is MOSTLY about, even if it touches others.
- "conflict" is for documents primarily about war, violence, or military/political confrontation.
- "politics" is for governance, elections, diplomacy — distinct from "conflict".
- "culture", "heritage", and "folklore" overlap; prefer "heritage" for named traditional \
practices/objects, "folklore" for oral tradition/stories, "culture" as the general fallback \
among the three.
- If truly nothing fits, use "uncategorized"."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_CATEGORIES = {c.value for c in ContentCategory}


def parse_classification_response(raw: str) -> Optional[Tuple[str, float]]:
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    category = payload.get("category")
    if not isinstance(category, str) or category not in _VALID_CATEGORIES:
        return None
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return category, confidence


class OllamaContentClassifier:
    def __init__(self, config: Optional[GenerationConfig] = None, host: Optional[str] = None):
        import os

        import ollama

        self.config = config or GenerationConfig()
        resolved_host = host or os.environ.get(self.config.host_env_var, "http://localhost:11434")
        self.client = ollama.Client(host=resolved_host)

    def _call_llm(self, title: str, text: str, max_chars: int = 2000) -> str:
        user_message = f'Title: "{title}"\nText: "{text[:max_chars]}"'
        response = self.client.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            think=False,  # see module docstring — avoids qwen3's empty-response failure mode
            options={"temperature": 0.0, "num_predict": 60},
        )
        return response.message.content or ""

    def classify(self, title: str, text: str) -> Optional[Tuple[str, float]]:
        raw = self._call_llm(title, text)
        return parse_classification_response(raw)
