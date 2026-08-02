"""
Bias measurement (Track F3).

Three independent, complementary signals, per ROADMAP.md:

1. Category-distribution comparison across sources — does WAFA frame
   documents differently than GDELT or Wikipedia (more "conflict", less
   "culture")? Uses ContentCategory labels (Track F2) rather than requiring
   BERTopic topics specifically, so this doesn't depend on pgvector being up.
2. A WEAT-style embedding association test — adapted from Caliskan et al.
   2017's Word Embedding Association Test. Classic WEAT uses word2vec-style
   single-word embeddings; this project's embedding model
   (Qwen3-Embedding, sentence-transformers) is a passage encoder, so single
   words/short terms are embedded individually as a practical adaptation
   (a documented simplification, not the original word-embedding setup).
   Runs entirely through the existing Embedder — no pgvector dependency,
   everything computed in-memory.
3. An LLM framing probe (Track B/C's generator, i.e. qwen3 via Ollama) —
   asks directly whether a passage frames its subject through conflict or
   through culture/daily life, aggregated by source.

WEAT effect size follows the standard formula (Caliskan et al. 2017):
for target word w, association(w) = mean_cos_sim(w, A) - mean_cos_sim(w, B)
over attribute sets A, B. Effect size = (mean(assoc(X)) - mean(assoc(Y))) /
std(assoc(X union Y)) for target sets X, Y. Positive effect size means set X
associates more with attribute set A (and Y more with B); magnitude near 0
means no differential association was found.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from src.rag.config import GenerationConfig


def category_distribution_by_source(documents: Iterable[dict]) -> Dict[str, Dict[str, float]]:
    """Normalized category distribution (fractions summing to 1) per source_id,
    over documents that carry a `category` field."""
    counts: Dict[str, Counter] = {}
    for doc in documents:
        category = doc.get("category")
        source_id = doc.get("source_id")
        if not category or not source_id:
            continue
        counts.setdefault(source_id, Counter())[category] += 1

    distributions: Dict[str, Dict[str, float]] = {}
    for source_id, counter in counts.items():
        total = sum(counter.values())
        distributions[source_id] = {cat: round(n / total, 4) for cat, n in counter.items()}
    return distributions


def total_variation_distance(dist_a: Dict[str, float], dist_b: Dict[str, float]) -> float:
    """0 = identical distributions, 1 = disjoint support — a simple,
    interpretable divergence metric (half the L1 distance between the two
    probability distributions) rather than KL divergence, which is undefined
    when one source never uses a category the other does.
    """
    categories = set(dist_a) | set(dist_b)
    return round(0.5 * sum(abs(dist_a.get(c, 0.0) - dist_b.get(c, 0.0)) for c in categories), 4)


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def weat_effect_size(
    target_set_x_embeddings: Sequence[np.ndarray],
    target_set_y_embeddings: Sequence[np.ndarray],
    attribute_set_a_embeddings: Sequence[np.ndarray],
    attribute_set_b_embeddings: Sequence[np.ndarray],
) -> Tuple[float, Dict[str, List[float]]]:
    """Standard WEAT effect size (Caliskan et al. 2017) — see module docstring."""

    def association(word_emb: np.ndarray) -> float:
        mean_a = float(np.mean([_cosine_similarity(word_emb, a) for a in attribute_set_a_embeddings]))
        mean_b = float(np.mean([_cosine_similarity(word_emb, b) for b in attribute_set_b_embeddings]))
        return mean_a - mean_b

    assoc_x = [association(w) for w in target_set_x_embeddings]
    assoc_y = [association(w) for w in target_set_y_embeddings]
    all_assoc = assoc_x + assoc_y

    std = float(np.std(all_assoc))
    if std == 0:
        effect_size = 0.0
    else:
        effect_size = round((float(np.mean(assoc_x)) - float(np.mean(assoc_y))) / std, 4)

    return effect_size, {"target_x_associations": assoc_x, "target_y_associations": assoc_y}


_FRAMING_SYSTEM_PROMPT = """You are a media-framing analyst. Given a passage, judge whether it \
frames its subject PRIMARILY through conflict/violence/political confrontation, or PRIMARILY \
through culture/daily life/heritage/history-as-narrative (non-conflict). Respond with ONLY a \
JSON object, no other text:
{"framing": "conflict" | "non_conflict" | "mixed", "confidence": <0.0-1.0>}

"mixed" is a legitimate answer when the passage genuinely balances both — don't force a side."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_VALID_FRAMINGS = {"conflict", "non_conflict", "mixed"}


def parse_framing_response(raw: str) -> Optional[Tuple[str, float]]:
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    framing = payload.get("framing")
    if framing not in _VALID_FRAMINGS:
        return None
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return framing, confidence


class OllamaFramingProbe:
    """LLM framing probe — reuses the same Ollama model/config as RAG
    generation (Track B) and relation extraction (Track E3), including the
    think=False fix qwen3 needed in both of those (see relations.py's
    module docstring for how that was found)."""

    def __init__(self, config: Optional[GenerationConfig] = None, host: Optional[str] = None):
        import os

        import ollama

        self.config = config or GenerationConfig()
        resolved_host = host or os.environ.get(self.config.host_env_var, "http://localhost:11434")
        self.client = ollama.Client(host=resolved_host)

    def _call_llm(self, passage: str, max_chars: int = 1500) -> str:
        response = self.client.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": _FRAMING_SYSTEM_PROMPT},
                {"role": "user", "content": f'Passage: "{passage[:max_chars]}"'},
            ],
            think=False,
            options={"temperature": 0.0, "num_predict": 60},
        )
        return response.message.content or ""

    def probe(self, passage: str) -> Optional[Tuple[str, float]]:
        return parse_framing_response(self._call_llm(passage))
