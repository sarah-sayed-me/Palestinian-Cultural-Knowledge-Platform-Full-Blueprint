"""Temporal analysis over the corpus (Track F4) — decade bucketing + term-frequency drift.

See src/nlp/temporal_analysis.py's module docstring for why this falls back
to content-mentioned years rather than relying solely on the mostly-null
`decade` metadata field on this corpus.

Usage:
    uv run python scripts/run_temporal_analysis.py
    uv run python scripts/run_temporal_analysis.py \
        --input data/processed/wikipedia_ar_documents.jsonl data/processed/wafa_documents.jsonl \
        --terms النكبة الاحتلال المقاومة التراث الثقافة
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.nlp.temporal_analysis import bucket_documents_by_decade, decade_summary, term_frequency_by_decade

DEFAULT_INPUTS = [
    Path("data/processed/wikipedia_ar_documents.jsonl"),
    Path("data/processed/wafa_documents.jsonl"),
    Path("data/processed/gdelt_documents.jsonl"),
]
# Culturally/historically salient Arabic terms — a starting set, not exhaustive.
DEFAULT_TERMS = ["النكبة", "الاحتلال", "المقاومة", "التراث", "الثقافة", "اللاجئين"]
DEFAULT_OUTPUT = Path("reports/temporal_analysis.json")


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket the corpus by decade and measure term drift.")
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--terms", type=str, nargs="+", default=DEFAULT_TERMS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    existing_inputs = [p for p in args.input if p.exists()]
    if not existing_inputs:
        raise FileNotFoundError(f"None of the input files exist: {args.input}")

    documents = [doc for path in existing_inputs for doc in _read_jsonl(path)]
    buckets = bucket_documents_by_decade(documents)
    decade_counts = decade_summary(buckets)
    metadata_decade_docs = sum(1 for d in documents if d.get("decade") is not None)
    content_estimated_docs = sum(len(v) for v in buckets.values()) - metadata_decade_docs
    frequencies = term_frequency_by_decade(buckets, args.terms)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "inputs": [str(p) for p in existing_inputs],
        "total_documents": len(documents),
        "documents_bucketed": sum(decade_counts.values()),
        "documents_with_metadata_decade": metadata_decade_docs,
        "documents_with_content_estimated_decade": max(content_estimated_docs, 0),
        "documents_unbucketed_no_year_found": len(documents) - sum(decade_counts.values()),
        "decade_distribution": decade_counts,
        "term_frequency_per_1000_words_by_decade": frequencies,
    }
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
