"""
Pure data-loading functions for the dashboard (Track G2).

Deliberately no `streamlit`/`plotly` imports here — every function returns
plain Python data structures (dicts/lists) so they're unit-testable without
a Streamlit runtime, and reusable from the API or a notebook too. Everything
reads from files already produced by earlier tracks (corpus JSONL, Track E's
KG GraphML, Track F's report JSON files) — nothing here requires a live
Postgres connection, so the dashboard's static panels work even when the
vector store is unreachable. Only the live "Ask" panel (built directly into
dashboard.py, not here) needs Postgres+Ollama, and is expected to degrade
gracefully on its own when they aren't reachable.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

DEFAULT_CORPUS_FILES = [
    Path("data/processed/wikipedia_ar_documents.jsonl"),
    Path("data/processed/wikipedia_en_documents.jsonl"),
    Path("data/processed/wafa_documents.jsonl"),
    Path("data/processed/gdelt_documents.jsonl"),
    Path("data/processed/semantic_scholar_documents.jsonl"),
]
DEFAULT_KG_GRAPH_PATH = Path("data/graph/kg_graph.graphml")
DEFAULT_TEMPORAL_REPORT_PATH = Path("reports/temporal_analysis.json")
DEFAULT_BIAS_REPORT_PATH = Path("reports/bias_measurement.json")


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_corpus_overview(paths: Optional[List[Path]] = None) -> Dict[str, Any]:
    """Per-source, per-language document counts — reads directly from the
    corpus JSONL files, always available regardless of DB state."""
    paths = paths if paths is not None else DEFAULT_CORPUS_FILES
    source_counts: Counter = Counter()
    language_counts: Counter = Counter()
    total = 0
    for path in paths:
        if not path.exists():
            continue
        for doc in _read_jsonl(path):
            total += 1
            source_counts[doc.get("source_id", "unknown")] += 1
            language_counts[doc.get("language", "unknown")] += 1
    return {
        "total_documents": total,
        "by_source": dict(source_counts.most_common()),
        "by_language": dict(language_counts.most_common()),
    }


def load_json_report(path: Path) -> Optional[Dict[str, Any]]:
    """Generic loader for a Track F report file — returns None (not an
    error) if the report hasn't been generated yet, so callers can render
    a "run this script first" message instead of crashing."""
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_topic_distribution(topics_paths: List[Path]) -> Dict[str, int]:
    """Document counts per topic_label across one or more *.topics.jsonl
    files (Track F1 output)."""
    counts: Counter = Counter()
    for path in topics_paths:
        if not path.exists():
            continue
        for doc in _read_jsonl(path):
            label = doc.get("topic_label")
            if label:
                counts[label] += 1
    return dict(counts.most_common(20))


def load_kg_graph_summary(path: Path = DEFAULT_KG_GRAPH_PATH) -> Optional[Dict[str, Any]]:
    """Node/edge counts + top-degree entities from the saved KG GraphML —
    a lightweight summary for the dashboard overview, not the full graph
    (the KG Explorer panel loads the graph itself separately, on demand)."""
    if not path.exists():
        return None
    from src.knowlegde_graph.graph_store import load_graph

    graph = load_graph(path)
    degrees = dict(graph.degree())
    top_by_degree = sorted(degrees.items(), key=lambda kv: -kv[1])[:10]
    return {
        "num_nodes": graph.number_of_nodes(),
        "num_edges": graph.number_of_edges(),
        "top_entities_by_degree": [
            {"canonical_name": graph.nodes[eid].get("canonical_name"), "degree": deg}
            for eid, deg in top_by_degree
            if deg > 0
        ],
    }
