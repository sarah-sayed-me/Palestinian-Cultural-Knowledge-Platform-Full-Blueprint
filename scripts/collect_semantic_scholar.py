"""Collect Palestine/Arabic-heritage papers from Semantic Scholar.

Fetches full open-access PDF text when Semantic Scholar reports one and it
extracts cleanly (see docs/licensing_checklist.md); falls back to the
abstract otherwise. The public API is unauthenticated and tightly
rate-limited; expect this to run slower than the Wikipedia collector.

Usage:
    docker compose up -d          # optional but recommended — enables
                                   # persistent cross-run dedup (Track D6)
    uv run python scripts/collect_semantic_scholar.py --max-docs 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.ingestion.pipeline import run_semantic_scholar_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect Semantic Scholar paper abstracts.")
    parser.add_argument("--max-docs", type=int, default=50)
    args = parser.parse_args()
    stats = run_semantic_scholar_pipeline(max_docs=args.max_docs)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
