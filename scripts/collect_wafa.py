"""Collect articles from WAFA (Palestinian news agency) via their sitemap.

Usage:
    docker compose up -d          # optional but recommended — enables
                                   # persistent cross-run dedup (Track D6)
    uv run python scripts/collect_wafa.py --max-docs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.ingestion.pipeline import run_wafa_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect WAFA news articles.")
    parser.add_argument("--max-docs", type=int, default=50)
    args = parser.parse_args()
    stats = run_wafa_pipeline(max_docs=args.max_docs)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
