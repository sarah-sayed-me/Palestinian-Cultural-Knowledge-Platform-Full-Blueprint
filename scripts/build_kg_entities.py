"""Canonicalize per-document NER entities into corpus-scope KGEntity records (Track E1).

Usage:
    uv run python scripts/run_ner.py                          # produces the *.ner.jsonl input
    uv run python scripts/build_kg_entities.py
    uv run python scripts/build_kg_entities.py --input data/processed/wikipedia_ar_documents.ner.jsonl \
                                                 --output data/entities/kg_entities.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.knowlegde_graph.canonicalize import canonicalize_entities, iter_ner_documents, write_kg_entities

DEFAULT_INPUTS = [Path("data/processed/wikipedia_ar_documents.ner.jsonl")]
DEFAULT_OUTPUT = Path("data/entities/kg_entities.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build corpus-scope KGEntity records from NER output.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more *.ner.jsonl files (documents without an 'entities' field are skipped).",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    missing = [p for p in args.input if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {missing}. Run scripts/run_ner.py first.")

    documents = list(iter_ner_documents(args.input))
    entities = canonicalize_entities(documents)
    write_kg_entities(entities, args.output)

    type_counts: dict[str, int] = {}
    for entity in entities:
        type_counts[entity.type] = type_counts.get(entity.type, 0) + 1

    summary = {
        "inputs": [str(p) for p in args.input],
        "documents_with_entities": len(documents),
        "unique_entities": len(entities),
        "type_distribution": dict(sorted(type_counts.items(), key=lambda kv: -kv[1])),
        "top_entities": [
            {"canonical_name": e.canonical_name, "type": e.type, "mention_count": e.mention_count}
            for e in entities[:15]
        ],
        "output": str(args.output),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
