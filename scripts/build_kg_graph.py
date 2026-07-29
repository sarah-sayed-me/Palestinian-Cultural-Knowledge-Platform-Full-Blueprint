"""Build and persist the NetworkX knowledge graph from linked entities + extracted relations (Track E4).

Usage:
    uv run python scripts/build_kg_graph.py
    uv run python scripts/build_kg_graph.py --entities data/entities/kg_entities.linked.jsonl \
        --relations data/graph/kg_relations.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

import networkx as nx

from src.knowlegde_graph.canonicalize import read_kg_entities
from src.knowlegde_graph.graph_store import build_graph, save_graph
from src.knowlegde_graph.schemas import KGRelation

DEFAULT_ENTITIES = Path("data/entities/kg_entities.linked.jsonl")
DEFAULT_ENTITIES_FALLBACK = Path("data/entities/kg_entities.jsonl")
DEFAULT_RELATIONS = Path("data/graph/kg_relations.jsonl")
DEFAULT_OUTPUT = Path("data/graph/kg_graph.graphml")


def _read_relations(path: Path) -> list[KGRelation]:
    with path.open("r", encoding="utf-8") as handle:
        return [KGRelation.model_validate_json(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the NetworkX knowledge graph.")
    parser.add_argument(
        "--entities",
        type=Path,
        default=DEFAULT_ENTITIES,
        help=f"Falls back to {DEFAULT_ENTITIES_FALLBACK} (un-linked) if this doesn't exist.",
    )
    parser.add_argument("--relations", type=Path, default=DEFAULT_RELATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    entities_path = args.entities if args.entities.exists() else DEFAULT_ENTITIES_FALLBACK
    if not entities_path.exists():
        raise FileNotFoundError("No KGEntity file found. Run scripts/build_kg_entities.py first.")
    if not args.relations.exists():
        raise FileNotFoundError(f"{args.relations} not found. Run scripts/extract_kg_relations.py first.")

    entities = read_kg_entities(entities_path)
    relations = _read_relations(args.relations)
    graph = build_graph(entities, relations)
    save_graph(graph, args.output)

    type_counts = Counter(data.get("type") for _, data in graph.nodes(data=True))
    predicate_counts = Counter(data.get("predicate") for _, _, data in graph.edges(data=True))
    degrees = dict(graph.degree())
    top_by_degree = sorted(degrees.items(), key=lambda kv: -kv[1])[:10]

    summary = {
        "entities_input": str(entities_path),
        "relations_input": str(args.relations),
        "output": str(args.output),
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "skipped_relations": graph.graph.get("skipped_relations", 0),
        "node_type_distribution": dict(type_counts.most_common(15)),
        "predicate_distribution": dict(predicate_counts.most_common(15)),
        "weakly_connected_components": nx.number_weakly_connected_components(graph),
        "top_entities_by_degree": [
            {"entity_id": eid, "canonical_name": graph.nodes[eid].get("canonical_name"), "degree": deg}
            for eid, deg in top_by_degree
            if deg > 0
        ],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
