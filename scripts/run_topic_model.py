"""Topic model the corpus using chunk embeddings already in pgvector (Track F1).

Reads every chunk from the `rag_chunks` table (populated by scripts/build_index.py),
fits BERTopic over the existing embeddings (no re-embedding), aggregates a
majority-vote topic per document across its chunks, then writes topic_id/topic_label
back onto each given input file as a sibling `*.topics.jsonl` output.

Only documents that were actually chunked+indexed will get a topic — if you've
collected new documents since the last `scripts/build_index.py` run, re-run
chunk_corpus.py + build_index.py first or they won't show up here.

Usage:
    docker compose up -d          # if not already running
    uv run python scripts/run_topic_model.py
    uv run python scripts/run_topic_model.py --input data/processed/wikipedia_ar_documents.ner.jsonl \
        --min-topic-size 15
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.nlp.topic_model import aggregate_document_topics, fetch_chunk_embeddings, fit_topic_model
from src.rag.config import RagConfig
from src.rag.db import get_connection

DEFAULT_INPUTS = [Path("data/processed/wikipedia_ar_documents.ner.jsonl")]


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _output_path_for(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + ".topics.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fit BERTopic over indexed chunks and label documents.")
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--min-topic-size", type=int, default=10)
    parser.add_argument("--nr-topics", type=int, default=None, help="Cap the number of topics (optional).")
    args = parser.parse_args()

    missing = [p for p in args.input if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {missing}")

    config = RagConfig.load()
    conn = get_connection()
    try:
        chunk_ids, doc_ids, texts, embeddings = fetch_chunk_embeddings(conn, config.vector_store.table)
    finally:
        conn.close()

    if not texts:
        raise RuntimeError(
            f"No chunks found in '{config.vector_store.table}'. Run scripts/chunk_corpus.py "
            "and scripts/build_index.py first."
        )

    topic_model, chunk_topics = fit_topic_model(
        texts, embeddings, min_topic_size=args.min_topic_size, nr_topics=args.nr_topics
    )
    doc_topics = aggregate_document_topics(doc_ids, chunk_topics, topic_model)

    outputs = []
    total_docs_labeled = 0
    for input_path in args.input:
        output_path = _output_path_for(input_path)
        labeled = 0
        with output_path.open("w", encoding="utf-8") as out_handle:
            for doc in _read_jsonl(input_path):
                assignment = doc_topics.get(doc.get("doc_id"))
                if assignment:
                    doc["topic_id"] = assignment["topic_id"]
                    doc["topic_label"] = assignment["topic_label"]
                    labeled += 1
                out_handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
        outputs.append(str(output_path))
        total_docs_labeled += labeled

    topic_sizes = Counter(chunk_topics)
    top_topics = [
        {"topic_id": int(tid), "size": count, "label": None}
        for tid, count in topic_sizes.most_common(15)
        if tid != -1
    ]
    for entry in top_topics:
        entry["label"] = next(
            (a["topic_label"] for a in doc_topics.values() if a["topic_id"] == entry["topic_id"]), None
        )

    summary = {
        "chunks_seen": len(texts),
        "documents_with_chunks": len(set(doc_ids)),
        "documents_labeled": total_docs_labeled,
        "num_topics_found": len([t for t in topic_sizes if t != -1]),
        "outlier_chunks": topic_sizes.get(-1, 0),
        "top_topics": top_topics,
        "outputs": outputs,
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
