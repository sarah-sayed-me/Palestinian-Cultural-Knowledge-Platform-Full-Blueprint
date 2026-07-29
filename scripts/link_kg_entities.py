"""Link corpus-scope KGEntity records to Wikidata QIDs via the alias-table linker (Track E2).

Usage:
    uv run python scripts/fetch_wikidata_aliases.py    # once, or whenever refreshing the dump
    uv run python scripts/build_kg_entities.py          # produces data/entities/kg_entities.jsonl
    uv run python scripts/link_kg_entities.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.knowlegde_graph.canonicalize import read_kg_entities, write_kg_entities
from src.knowlegde_graph.entity_linking import WikidataAliasLinker

DEFAULT_ENTITIES = Path("data/entities/kg_entities.jsonl")
DEFAULT_ALIASES = Path("data/entities/wikidata_palestine_aliases.jsonl")
DEFAULT_OUTPUT = Path("data/entities/kg_entities.linked.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Link KGEntity records to Wikidata QIDs.")
    parser.add_argument("--entities", type=Path, default=DEFAULT_ENTITIES)
    parser.add_argument("--aliases", type=Path, default=DEFAULT_ALIASES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    if not args.entities.exists():
        raise FileNotFoundError(f"{args.entities} not found. Run scripts/build_kg_entities.py first.")
    if not args.aliases.exists():
        raise FileNotFoundError(f"{args.aliases} not found. Run scripts/fetch_wikidata_aliases.py first.")

    entities = read_kg_entities(args.entities)
    linker = WikidataAliasLinker.from_dump(args.aliases)
    linked_pairs = linker.link_entities(entities)

    linked_entities = [entity for entity, _ in linked_pairs]
    write_kg_entities(linked_entities, args.output)

    method_counts = {"exact": 0, "fuzzy": 0, "none": 0}
    for _, result in linked_pairs:
        method_counts[result.method] += 1

    summary = {
        "total_entities": len(entities),
        "linked": method_counts["exact"] + method_counts["fuzzy"],
        "link_rate": round((method_counts["exact"] + method_counts["fuzzy"]) / len(entities), 4)
        if entities
        else 0.0,
        "method_breakdown": method_counts,
        "output": str(args.output),
        "sample_linked": [
            {
                "canonical_name": entity.canonical_name,
                "type": entity.type,
                "wikidata_qid": entity.wikidata_qid,
                "match_method": result.method,
                "match_score": round(result.score, 3),
            }
            for entity, result in linked_pairs
            if entity.wikidata_qid
        ][:15],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
