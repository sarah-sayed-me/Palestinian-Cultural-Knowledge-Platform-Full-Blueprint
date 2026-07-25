"""RAG query CLI — the MVP entrypoint (ROADMAP.md Track B).

Usage:
    docker compose up -d
    uv run python scripts/chunk_corpus.py
    uv run python scripts/build_index.py
    uv run python scripts/ask.py "ما هي الكنافة النابلسية؟"

Requires Ollama running locally (see configs/rag.yaml `generation.model` for
which model to `ollama pull` first).
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Answers are frequently Arabic. Some Windows terminals default stdout to a
# legacy codepage (e.g. cp1252) that can't encode Arabic script and crashes
# on print() — force UTF-8 regardless of the console's configured codepage.
sys.stdout.reconfigure(encoding="utf-8")

from src.rag.config import RagConfig
from src.rag.db import get_connection
from src.rag.embedder import Embedder
from src.rag.generator import OllamaGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import Retriever


def main() -> None:
    parser = argparse.ArgumentParser(description="Ask a question over the RAG corpus.")
    parser.add_argument("question")
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument(
        "--model",
        help="Override configs/rag.yaml's generation.model for this run only "
        "(e.g. --model llama3.1:8b to use a model you already have pulled).",
    )
    args = parser.parse_args()

    config = RagConfig.load(args.config)
    if args.model:
        config = replace(config, generation=replace(config.generation, model=args.model))
    embedder = Embedder(config.embedding)
    conn = get_connection()
    try:
        retriever = Retriever(conn, embedder, config)
        generator = OllamaGenerator(config.generation)
        pipeline = RAGPipeline(retriever, generator, config)

        answer = pipeline.ask(args.question)

        print(answer.text)
        if answer.citations:
            print()
            for citation in answer.citations:
                print(f"[{citation.index}] {citation.title or '(untitled)'} — {citation.source_url or 'n/a'}")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
