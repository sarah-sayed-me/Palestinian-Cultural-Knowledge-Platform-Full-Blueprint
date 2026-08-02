"""Classify documents into ContentCategory via a local LLM (Track F2).

See src/nlp/content_classifier.py's module docstring for why this uses
zero-shot LLM classification instead of the originally-planned fine-tuned
AraBERT model, and what that trade-off means.

Requires Ollama running locally (same model as configs/rag.yaml's
generation.model). Accepts multiple --input files; --max-docs applies PER
FILE, same convention as scripts/extract_kg_relations.py.

Usage:
    uv run python scripts/run_content_classification.py --max-docs 30
    uv run python scripts/run_content_classification.py \
        --input data/processed/wikipedia_ar_documents.jsonl data/processed/wafa_documents.jsonl \
        --max-docs 20
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, Iterator

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.nlp.content_classifier import OllamaContentClassifier
from src.rag.config import RagConfig

DEFAULT_INPUTS = [Path("data/processed/wikipedia_ar_documents.jsonl")]


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _output_path_for(input_path: Path) -> Path:
    return input_path.with_name(input_path.stem + ".categorized.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Classify documents into ContentCategory via a local LLM.")
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--max-docs", type=int, default=30, help="Documents classified PER input file.")
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    missing = [p for p in args.input if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Input file(s) not found: {missing}")

    generation_config = RagConfig.load().generation
    if args.model:
        generation_config = replace(generation_config, model=args.model)

    classifier = OllamaContentClassifier(config=generation_config)

    started = time.time()
    category_counts: Counter[str] = Counter()
    total_classified = 0
    total_unparseable = 0
    outputs = []

    for input_path in args.input:
        output_path = _output_path_for(input_path)
        docs_from_this_file = 0
        with output_path.open("w", encoding="utf-8") as out_handle:
            for doc in _read_jsonl(input_path):
                if docs_from_this_file >= args.max_docs:
                    out_handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
                    continue
                result = classifier.classify(doc.get("title", ""), doc.get("text", ""))
                docs_from_this_file += 1
                if result is not None:
                    category, confidence = result
                    doc["category"] = category
                    doc["category_confidence"] = confidence
                    category_counts[category] += 1
                    total_classified += 1
                else:
                    total_unparseable += 1
                out_handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
        outputs.append(str(output_path))

    summary = {
        "inputs": [str(p) for p in args.input],
        "outputs": outputs,
        "documents_classified": total_classified,
        "documents_unparseable_response": total_unparseable,
        "category_distribution": dict(category_counts.most_common()),
        "model": generation_config.model,
        "duration_seconds": round(time.time() - started, 2),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
