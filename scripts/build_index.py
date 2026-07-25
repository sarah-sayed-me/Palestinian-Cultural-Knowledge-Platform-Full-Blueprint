"""Embed chunks.jsonl and load it into the pgvector index.

Usage:
    docker compose up -d
    uv run python scripts/chunk_corpus.py
    uv run python scripts/build_index.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.rag.chunker import DEFAULT_OUTPUT as DEFAULT_CHUNKS_PATH
from src.rag.config import RagConfig
from src.rag.index import build_index


def main() -> None:
    parser = argparse.ArgumentParser(description="Embed and index chunks.jsonl into pgvector.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS_PATH)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    args = parser.parse_args()

    config = RagConfig.load(args.config)
    summary = build_index(chunks_path=args.chunks, config=config)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
