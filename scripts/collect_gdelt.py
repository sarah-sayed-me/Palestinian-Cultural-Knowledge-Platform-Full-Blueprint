"""Collect GDELT-indexed news articles (full text fetched from source, only
where robots.txt permits — see src/ingestion/collectors/gdelt_collector.py).

Usage:
    docker compose up -d          # optional but recommended — enables
                                   # persistent cross-run dedup (Track D6)
    uv run python scripts/collect_gdelt.py --max-docs 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.ingestion.pipeline import run_gdelt_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect GDELT-indexed news articles.")
    parser.add_argument("--max-docs", type=int, default=30)
    args = parser.parse_args()
    stats = run_gdelt_pipeline(max_docs=args.max_docs)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
