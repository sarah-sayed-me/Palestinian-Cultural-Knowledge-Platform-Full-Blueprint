"""
Topic modeling (Track F1) — BERTopic over the chunk embeddings already
computed in Track B (the pgvector `rag_chunks` table), so no new embedding
pass is needed, per ROADMAP.md.

Scope note: this only covers whatever has actually been chunked + embedded +
indexed (scripts/chunk_corpus.py + scripts/build_index.py) — if that hasn't
been re-run since the corpus grew or new sources were added, topic modeling
silently only sees what's in `rag_chunks` today, not the full JSONL corpus.
The CLI script reports how many chunks/documents it actually saw so this
is visible, not a silent gap.

Topics are computed at chunk granularity (BERTopic's natural unit) and then
aggregated to document scope by majority vote across a document's chunks —
a document already has multiple topic-relevant units (its chunks), so this
avoids a second, document-level embedding pass just to get one topic per doc.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from psycopg2.extensions import connection as PGConnection


def fetch_chunk_embeddings(
    conn: PGConnection, table: str = "rag_chunks"
) -> Tuple[List[str], List[str], List[str], List[List[float]]]:
    """Return (chunk_ids, doc_ids, texts, embeddings) for every indexed chunk."""
    with conn.cursor() as cur:
        cur.execute(f"SELECT chunk_id, doc_id, text, embedding FROM {table};")
        rows = cur.fetchall()
    chunk_ids = [r[0] for r in rows]
    doc_ids = [r[1] for r in rows]
    texts = [r[2] for r in rows]
    embeddings = [list(r[3]) for r in rows]
    return chunk_ids, doc_ids, texts, embeddings


def fit_topic_model(
    texts: List[str],
    embeddings: List[List[float]],
    *,
    min_topic_size: int = 10,
    nr_topics: Optional[int] = None,
):
    """Fit BERTopic over precomputed embeddings (no re-embedding)."""
    import numpy as np
    from bertopic import BERTopic

    topic_model = BERTopic(
        min_topic_size=min_topic_size,
        nr_topics=nr_topics,
        calculate_probabilities=False,
        verbose=False,
    )
    topics, _ = topic_model.fit_transform(texts, embeddings=np.array(embeddings, dtype="float32"))
    return topic_model, topics


def topic_label(topic_model: Any, topic_id: int, top_n: int = 4) -> str:
    """A short human-readable label from BERTopic's own top-n keyword representation.

    topic_id == -1 is BERTopic/HDBSCAN's reserved "outlier" cluster (a chunk
    that didn't fit any dense topic cluster) — labeled explicitly rather than
    treated as topic "-1" with a keyword label that doesn't mean anything.
    """
    if topic_id == -1:
        return "outlier"
    words = topic_model.get_topic(topic_id)
    if not words:
        return f"topic_{topic_id}"
    return " / ".join(word for word, _ in words[:top_n])


def aggregate_document_topics(
    chunk_doc_ids: List[str], chunk_topics: List[int], topic_model: Any
) -> Dict[str, Dict[str, Any]]:
    """Majority-vote one topic per document across all of its chunks."""
    by_doc: Dict[str, Counter] = {}
    for doc_id, topic in zip(chunk_doc_ids, chunk_topics):
        by_doc.setdefault(doc_id, Counter())[topic] += 1

    result: Dict[str, Dict[str, Any]] = {}
    for doc_id, counter in by_doc.items():
        topic_id, _ = counter.most_common(1)[0]
        result[doc_id] = {
            "topic_id": int(topic_id),
            "topic_label": topic_label(topic_model, topic_id),
        }
    return result
