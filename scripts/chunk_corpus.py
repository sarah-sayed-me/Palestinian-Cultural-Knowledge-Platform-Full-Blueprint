"""Chunk the accepted corpus into overlapping, sentence-aligned passages for RAG.

Usage:
    uv run python scripts/chunk_corpus.py
    uv run python scripts/chunk_corpus.py --input data/processed/wikipedia_ar_documents.jsonl \
                                            --output data/processed/chunks.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.chunker import DEFAULT_INPUT, DEFAULT_OUTPUT, chunk_corpus
from src.rag.config import RagConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk the accepted corpus for RAG.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    args = parser.parse_args()

    config = RagConfig.load(args.config)
    summary = chunk_corpus(input_path=args.input, output_path=args.output, config=config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
