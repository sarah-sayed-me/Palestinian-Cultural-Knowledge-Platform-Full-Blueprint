"""Fetch and cache a Palestine-related Wikidata alias dump (Track E2).

This is a one-off (or occasionally-refreshed) cache-build step, not something
run at link time — scripts/link_kg_entities.py reads the cached JSONL this
produces.

Usage:
    uv run python scripts/fetch_wikidata_aliases.py
    uv run python scripts/fetch_wikidata_aliases.py --limit 3000 \
        --output data/entities/wikidata_palestine_aliases.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.knowlegde_graph.wikidata_aliases import fetch_palestine_aliases, write_alias_dump

DEFAULT_OUTPUT = Path("data/entities/wikidata_palestine_aliases.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch a Palestine-related Wikidata alias dump.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=5000, help="Per-query SPARQL LIMIT.")
    parser.add_argument("--timeout", type=float, default=60.0)
    args = parser.parse_args()

    entries = fetch_palestine_aliases(limit=args.limit, timeout=args.timeout)
    write_alias_dump(entries, args.output)

    with_aliases = sum(1 for e in entries if e["aliases"])
    summary = {
        "output": str(args.output),
        "total_items": len(entries),
        "items_with_aliases": with_aliases,
        "sample": entries[:5],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
