"""Extract entity relations via LLM prompting over sentence-level co-occurring entity pairs (Track E3).

Requires Ollama running locally with the model configured in configs/rag.yaml's
generation.model already pulled (same requirement as scripts/ask.py — run
`ollama pull <model>` first if you haven't).

This makes one LLM call per candidate entity pair, so runtime scales with
--max-docs * --max-pairs-per-doc. Start small (the defaults below) and scale
up once you've checked a sample of the output — same iterative-corpus
philosophy as the rest of this project (see README.md's Development
Philosophy section).

Accepts multiple --input files (one *.ner.jsonl per source, same convention
as scripts/build_kg_entities.py) so the KG isn't limited to whichever source
happened to get NER run on it first. --max-docs applies PER FILE, not to the
combined total — otherwise a single large source listed first would starve
every other source of any documents at all.

Usage:
    uv run python scripts/extract_kg_relations.py --max-docs 20
    uv run python scripts/extract_kg_relations.py \
        --input data/processed/wikipedia_ar_documents.ner.jsonl data/processed/wafa_documents.ner.jsonl \
        --max-docs 15 --max-pairs-per-doc 15 --model qwen3:14b
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.knowlegde_graph.relations import OllamaRelationExtractor
from src.rag.config import RagConfig

DEFAULT_INPUTS = [Path("data/processed/wikipedia_ar_documents.ner.jsonl")]
DEFAULT_OUTPUT = Path("data/graph/kg_relations.jsonl")


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract KG relations from NER-enriched documents via LLM prompting."
    )
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        default=DEFAULT_INPUTS,
        help="One or more *.ner.jsonl files. --max-docs applies to EACH file, not the combined total.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--max-docs", type=int, default=20, help="Documents considered PER input file.")
    parser.add_argument("--max-pairs-per-doc", type=int, default=30, help="Caps LLM calls per document.")
    parser.add_argument("--model", type=str, default=None, help="Override configs/rag.yaml's generation.model.")
    args = parser.parse_args()

    missing = [p for p in args.input if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {missing}. Run scripts/run_ner.py first.")

    generation_config = RagConfig.load().generation
    if args.model:
        generation_config = replace(generation_config, model=args.model)

    extractor = OllamaRelationExtractor(config=generation_config)

    started = time.time()
    documents_processed = 0
    relations_kept = 0
    predicate_counts: Dict[str, int] = {}
    per_source_counts: Dict[str, Dict[str, int]] = {}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as out_handle:
        for input_path in args.input:
            source_label = input_path.name
            source_stats = per_source_counts.setdefault(
                source_label, {"documents_processed": 0, "relations_extracted": 0}
            )
            docs_from_this_file = 0
            for doc in _read_jsonl(input_path):
                if docs_from_this_file >= args.max_docs:
                    break
                if not doc.get("entities"):
                    continue
                relations = extractor.extract_document(doc, max_pairs=args.max_pairs_per_doc)
                docs_from_this_file += 1
                documents_processed += 1
                source_stats["documents_processed"] += 1
                for relation in relations:
                    relations_kept += 1
                    source_stats["relations_extracted"] += 1
                    predicate_counts[relation.predicate] = predicate_counts.get(relation.predicate, 0) + 1
                    out_handle.write(json.dumps(relation.model_dump(), ensure_ascii=False) + "\n")

    summary = {
        "inputs": [str(p) for p in args.input],
        "output": str(args.output),
        "documents_processed": documents_processed,
        "relations_extracted": relations_kept,
        "per_source": per_source_counts,
        "predicate_distribution": dict(sorted(predicate_counts.items(), key=lambda kv: -kv[1])),
        "model": generation_config.model,
        "duration_seconds": round(time.time() - started, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
