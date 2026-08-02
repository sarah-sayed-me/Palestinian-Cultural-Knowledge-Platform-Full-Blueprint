"""
Temporal analysis (Track F4).

ROADMAP.md's original design: bucket documents by `DocumentMetadata.decade`
(already computed from `date_published`) and measure term drift across
decades. Checked against the real corpus before building on that assumption
(882 documents, all sources): `decade` is null for all 581 Wikipedia
documents (articles are continuously edited — there's no meaningful
"published" date to derive it from) and, for the 204 documents that do have
it, is 2020 for all but two (WAFA/GDELT are current news, Semantic Scholar
papers are mostly recent) — one degenerate bucket, no real cross-decade
spread from metadata alone.

This module therefore ALSO extracts a content-based signal: 4-digit years
actually mentioned in a document's own text (a Wikipedia article about the
1948 Nakba discusses 1948 extensively regardless of when the page was last
edited). `bucket_documents_by_decade` prefers a real metadata `decade` when
present and falls back to this content-based estimate otherwise — this is
what actually produces a usable temporal spread on this corpus; the
metadata path is kept, not removed, for sources that do carry a genuine
publish date (e.g. real archival material, if/when added).
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Dict, Iterable, List, Optional

from src.preprocessing.arabic_normalizer import normalize_arabic

# Plausible calendar years only (1800-2029) — bounded so this doesn't match
# arbitrary 4-digit numbers (populations, phone numbers, IDs) that happen to
# fall in a year-shaped pattern.
_YEAR_RE = re.compile(r"\b(1[89]\d{2}|20[0-2]\d)\b")


def extract_year_mentions(text: str) -> Counter:
    return Counter(int(y) for y in _YEAR_RE.findall(text))


def document_decade_from_text(text: str, min_mentions: int = 1) -> Optional[int]:
    """The decade of the most frequently mentioned year in this document's
    own text, or None if no plausible year appears at all."""
    counts = extract_year_mentions(text)
    if not counts:
        return None
    year, count = counts.most_common(1)[0]
    if count < min_mentions:
        return None
    return (year // 10) * 10


def bucket_documents_by_decade(documents: Iterable[dict]) -> Dict[int, List[dict]]:
    """Group documents by decade — metadata `decade` if present, else a
    content-based estimate from the document's own text (see module
    docstring for why the fallback is necessary on this corpus)."""
    buckets: Dict[int, List[dict]] = {}
    for doc in documents:
        decade = doc.get("decade")
        if decade is None:
            decade = document_decade_from_text(doc.get("text", ""))
        if decade is None:
            continue
        buckets.setdefault(decade, []).append(doc)
    return buckets


def term_frequency_by_decade(buckets: Dict[int, List[dict]], terms: List[str]) -> Dict[str, Dict[int, float]]:
    """Relative frequency (mentions per 1,000 words) of each term per decade
    bucket — a simple, interpretable term-count signal rather than an
    embedding-space drift metric, requiring no additional embedding pass
    over what the corpus already has.
    """
    result: Dict[str, Dict[int, float]] = {term: {} for term in terms}
    for decade, docs in buckets.items():
        total_words = sum(d.get("word_count") or len(d.get("text", "").split()) for d in docs)
        if total_words == 0:
            continue
        for term in terms:
            term_normalized = normalize_arabic(term)
            count = sum(normalize_arabic(d.get("text", "")).count(term_normalized) for d in docs)
            result[term][decade] = round(count / total_words * 1000, 4)
    return result


def decade_summary(buckets: Dict[int, List[dict]]) -> Dict[int, int]:
    return {decade: len(docs) for decade, docs in sorted(buckets.items())}
