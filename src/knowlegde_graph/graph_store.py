"""
Knowledge graph store — NetworkX in-process prototype (Track E4).

ROADMAP.md's explicit call: prove the graph is useful with NetworkX before
standing up Neo4j Community — "matches the project's own iterative-validation
practice: prove the graph is useful before standing up infrastructure for
it." Migrate to Neo4j only once E1-E3 are validated and the dashboard's KG
explorer (Track G) actually needs multi-hop Cypher queries at scale.

GraphML is the persistence format: a standard, tool-readable interchange
format (Gephi, yEd, and networkx itself all read/write it) that handles this
project's scalar node/edge attributes cleanly. Every node/edge attribute set
here (canonical_name, type, wikidata_qid, mention_count / predicate,
confidence, source_doc_id, evidence_sentence) is a plain string, float, or
int — no lists/dicts — specifically so GraphML round-trips it without lossy
serialization. This also means nothing here is a one-way door: a future
Neo4j migration reads the same GraphML file directly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List

import networkx as nx

from src.knowlegde_graph.schemas import KGEntity, KGRelation


def build_graph(entities: Iterable[KGEntity], relations: Iterable[KGRelation]) -> nx.MultiDiGraph:
    """Build a directed multigraph: one node per entity, one edge per relation.

    MultiDiGraph (not DiGraph) because two entities can legitimately have more
    than one relation between them (different predicates, or the same
    predicate re-observed with different evidence in different documents) —
    collapsing those into a single edge would silently drop information.
    """
    graph = nx.MultiDiGraph()
    for entity in entities:
        graph.add_node(
            entity.entity_id,
            canonical_name=entity.canonical_name,
            type=entity.type,
            wikidata_qid=entity.wikidata_qid or "",
            mention_count=entity.mention_count,
        )

    skipped = 0
    for relation in relations:
        if relation.subject_entity_id not in graph or relation.object_entity_id not in graph:
            # A relation whose entities aren't in the E1/E2 entity set (e.g. a
            # stale/mismatched run against different inputs) — skip rather
            # than silently creating a placeholder node with no metadata.
            skipped += 1
            continue
        graph.add_edge(
            relation.subject_entity_id,
            relation.object_entity_id,
            key=relation.relation_id,
            predicate=relation.predicate,
            confidence=relation.confidence,
            source_doc_id=relation.source_doc_id,
            evidence_sentence=relation.evidence_sentence or "",
        )
    graph.graph["skipped_relations"] = skipped
    return graph


def save_graph(graph: nx.MultiDiGraph, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx.write_graphml(graph, path)


def load_graph(path: Path) -> nx.MultiDiGraph:
    return nx.read_graphml(path)


def find_entities_by_name(graph: nx.MultiDiGraph, query: str) -> List[str]:
    """Substring match against canonical_name — a cheap lookup helper for
    poking at the graph without Cypher, per ROADMAP.md's E4 note. Returns
    matching entity_ids."""
    query_lower = query.lower()
    return [
        node for node, data in graph.nodes(data=True) if query_lower in str(data.get("canonical_name", "")).lower()
    ]


def neighbors_of(graph: nx.MultiDiGraph, entity_id: str) -> List[dict]:
    """All outgoing relations from entity_id, with the target entity's own
    attributes attached so callers don't need a second lookup."""
    if entity_id not in graph:
        return []
    results = []
    for _, target, data in graph.out_edges(entity_id, data=True):
        results.append(
            {
                "predicate": data.get("predicate"),
                "confidence": data.get("confidence"),
                "target_entity_id": target,
                "target_canonical_name": graph.nodes[target].get("canonical_name"),
                "target_type": graph.nodes[target].get("type"),
            }
        )
    return results
