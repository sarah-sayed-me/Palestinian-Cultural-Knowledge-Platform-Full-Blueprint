"""Populate the `entities` field of the processed corpus using EntityExtractor.

Reads data/processed/wikipedia_ar_documents.jsonl (or --input), runs the
combined CAMeL Tools NER + heritage dictionary pipeline in document batches
(CAMeL inference is batched across ALL sentences in a batch — not one call
per document), and writes the enriched records back out as JSONL.

Usage:
    uv run python scripts/run_ner.py
    uv run python scripts/run_ner.py --batch-size 64 --no-gpu
    uv run python scripts/run_ner.py --input data/processed/wikipedia_ar_documents.jsonl \
                                       --output data/processed/wikipedia_ar_documents.ner.jsonl
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Iterator

from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.ingestion.entity_extractor import CamelNERBackend, EntityExtractor, HeritageMatcher

DEFAULT_INPUT = Path("data/processed/wikipedia_ar_documents.jsonl")
DEFAULT_OUTPUT = Path("data/processed/wikipedia_ar_documents.ner.jsonl")
DEFAULT_HERITAGE_CONFIG = Path("configs/heritage_entities.yaml")


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _batched(iterable: Iterator[dict[str, Any]], batch_size: int) -> Iterator[list[dict[str, Any]]]:
    batch: list[dict[str, Any]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def run(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    heritage_config: Path = DEFAULT_HERITAGE_CONFIG,
    batch_size: int = 32,
    use_gpu: bool = True,
) -> dict[str, Any]:
    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    started = time.time()
    heritage_matcher = HeritageMatcher.from_yaml(heritage_config)
    camel_backend = CamelNERBackend.get(prefer_gpu=use_gpu)
    extractor = EntityExtractor(heritage_matcher=heritage_matcher, camel_backend=camel_backend)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    total_docs = 0
    total_entities = 0

    with output_path.open("w", encoding="utf-8") as out_handle:
        for batch in _batched(_read_jsonl(input_path), batch_size):
            texts = [doc.get("text", "") for doc in batch]
            entities_per_doc = extractor.extract_many(texts)
            for doc, entities in zip(batch, entities_per_doc):
                doc["entities"] = entities
                total_entities += len(entities)
                out_handle.write(json.dumps(doc, ensure_ascii=False) + "\n")
            total_docs += len(batch)

    summary = {
        "input": str(input_path),
        "output": str(output_path),
        "documents_processed": total_docs,
        "total_entities": total_entities,
        "average_entities_per_doc": round(total_entities / total_docs, 2) if total_docs else 0.0,
        "camel_available": camel_backend.available,
        "camel_device": camel_backend.device,
        "duration_seconds": round(time.time() - started, 2),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NER over the processed corpus.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--heritage-config", type=Path, default=DEFAULT_HERITAGE_CONFIG)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--no-gpu", action="store_true", help="Force CPU even if CUDA is available.")
    args = parser.parse_args()

    summary = run(
        input_path=args.input,
        output_path=args.output,
        heritage_config=args.heritage_config,
        batch_size=args.batch_size,
        use_gpu=not args.no_gpu,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()