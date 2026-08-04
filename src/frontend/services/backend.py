"""Backend service adapter — wires the frontend to the real pipeline.

Every function returns exactly what its matching components/views/*.py
consumer expects (see mock/demo_data.py for the schema each one was
originally designed against), but the values come from real pipeline
output: the corpus JSONL files, the saved KG GraphML, the Track F report
JSON files, the topic-model output, and — for Ask — the live RAG pipeline.

Two real gaps, handled honestly rather than papered over with fabricated
numbers:
  - Topic Map's 2D scatter needs per-document coordinates, which nothing
    upstream persisted before this integration; scripts/run_topic_model.py
    now writes topic_x/topic_y alongside topic_id/topic_label (see
    src/nlp/topic_model.py's compute_2d_projection/aggregate_document_coords).
  - Bias Meter's dimension list is data-driven from whatever ContentCategory
    values actually appear in reports/bias_measurement.json (up to 17
    possible categories), not a fixed hardcoded 4/5 — "uncategorized" is
    excluded from the percentage math since it's a coverage gap, not a
    signal, and a source with thin classification coverage shows fewer/
    smaller real dimensions rather than invented ones.

Every get_*_data() is Streamlit-cached: app.py reruns top to bottom on
every widget interaction, so without caching each tab's real data (file
reads, GraphML parsing, decade extraction over full document text) would
reload on every click anywhere in the app, not just in that tab.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import streamlit as st

from src.frontend.data_loaders import (
    DEFAULT_BIAS_REPORT_PATH,
    DEFAULT_KG_GRAPH_PATH,
    DEFAULT_TEMPORAL_REPORT_PATH,
    load_corpus_overview,
    load_json_report,
    load_topic_distribution,
)

CACHE_TTL_SECONDS = 30  # short enough to notice a freshly re-run pipeline script

SOURCE_NAME_AR = {
    "wikipedia-ar": "ويكيبيديا العربية",
    "wikipedia-en": "ويكيبيديا الإنجليزية",
    "wafa-news": "وكالة وفا",
    "gdelt": "GDELT",
    "semantic-scholar": "Semantic Scholar",
}

LANGUAGE_NAME_AR = {
    "ar-MSA": "العربية",
    "ar": "العربية",
    "en": "الإنجليزية",
    "unknown": "غير محدد",
}

ENTITY_TYPE_AR = {
    "PERSON": "أشخاص",
    "LOCATION": "أماكن",
    "ORGANIZATION": "منظمات",
    "HERITAGE_FOOD": "تراث غذائي",
    "HERITAGE_CRAFT": "حرف تراثية",
    "HERITAGE_PLACE": "أماكن تراثية",
    "HERITAGE_CLOTHING": "أزياء تراثية",
    "HERITAGE_PLANT": "نباتات تراثية",
    "HERITAGE_HERITAGE_PRACTICE": "ممارسات تراثية",
    "MISC": "أخرى",
}

# Mirrors src/ingestion/schemas.py::ContentCategory
CATEGORY_AR = {
    "conflict": "الصراع",
    "culture": "الثقافة",
    "history": "التاريخ",
    "arts_literature": "الأدب والفنون",
    "politics": "السياسة",
    "daily_life": "الحياة اليومية",
    "food_cuisine": "المأكولات",
    "religion": "الدين",
    "architecture": "العمارة",
    "education": "التعليم",
    "economy": "الاقتصاد",
    "music": "الموسيقى",
    "folklore": "الفولكلور",
    "heritage": "التراث",
    "biography": "السِيَر",
    "geography": "الجغرافيا",
    "uncategorized": "غير مصنف",
}
_CATEGORY_STACK_ORDER = [c for c in CATEGORY_AR if c != "uncategorized"]

TOPIC_COLOR_SEQ = ["#6B8E23", "#009736", "#C5A55A", "#CE1126", "#1a1a1a", "#556B2F", "#8B8580"]
OUTLIER_COLOR = "#D4CFC8"

MAX_GAUGE_DIMENSIONS = 6


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _mtime(path: Path) -> float:
    return path.stat().st_mtime if path.exists() else 0.0


@st.cache_resource(show_spinner=False)
def _load_graph_cached(path_str: str, mtime: float):
    from src.knowlegde_graph.graph_store import load_graph

    return load_graph(Path(path_str))


def _get_graph():
    """The KG graph, reloaded only when kg_graph.graphml's mtime changes —
    so a background re-run of scripts/build_kg_graph.py is picked up
    without restarting the dashboard."""
    if not DEFAULT_KG_GRAPH_PATH.exists():
        return None
    return _load_graph_cached(str(DEFAULT_KG_GRAPH_PATH), _mtime(DEFAULT_KG_GRAPH_PATH))


# ============================================================
# OVERVIEW
# ============================================================

def _kg_entity_type_counts() -> List[Dict[str, Any]]:
    graph = _get_graph()
    if graph is None:
        return []
    counts = Counter(data.get("type", "MISC") for _, data in graph.nodes(data=True))
    return [
        {"type": t, "label": ENTITY_TYPE_AR.get(t, t), "count": c}
        for t, c in counts.most_common(8)
    ]


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_overview_data() -> Dict[str, Any]:
    overview = load_corpus_overview()
    graph = _get_graph()
    kg_entities = graph.number_of_nodes() if graph is not None else 0
    kg_relations = graph.number_of_edges() if graph is not None else 0

    documents_by_source = [
        {"source": SOURCE_NAME_AR.get(sid, sid), "count": count}
        for sid, count in overview["by_source"].items()
    ]
    documents_by_language = [
        {"language": LANGUAGE_NAME_AR.get(lang, lang), "count": count}
        for lang, count in overview["by_language"].items()
    ]

    topics_files = list(Path("data/processed").glob("*.topics.jsonl"))
    distribution = load_topic_distribution(topics_files)
    topic_distribution = [{"topic": label, "count": count} for label, count in distribution.items()][:7]

    kg_entity_types = _kg_entity_type_counts()

    return {
        "kpis": {
            "total_documents": overview["total_documents"],
            "sources": len(overview["by_source"]),
            "kg_entities": kg_entities,
            "kg_relations": kg_relations,
        },
        "documents_by_source": documents_by_source,
        "documents_by_language": documents_by_language,
        "topic_distribution": topic_distribution,
        "kg_entity_types": kg_entity_types,
    }


# ============================================================
# TOPIC MAP
# ============================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_topic_data() -> Dict[str, Any]:
    from src.nlp.temporal_analysis import document_decade_from_text

    topics_files = sorted(Path("data/processed").glob("*.topics.jsonl"))
    docs_raw = []
    for path in topics_files:
        docs_raw.extend(_read_jsonl(path))

    # Only documents that were actually topic-modeled *and* have the 2D
    # projection (scripts/run_topic_model.py must have been re-run since
    # topic_x/topic_y was added).
    docs_raw = [d for d in docs_raw if d.get("topic_id") is not None and "topic_x" in d]

    if not docs_raw:
        return {
            "topics": [],
            "documents": [],
            "topic_details": {},
            "filters": {"sources": ["الكل"], "languages": ["الكل"], "decades": ["الكل"]},
        }

    topic_ids = sorted({d["topic_id"] for d in docs_raw if d["topic_id"] != -1})
    topic_color = {tid: TOPIC_COLOR_SEQ[i % len(TOPIC_COLOR_SEQ)] for i, tid in enumerate(topic_ids)}
    topic_color[-1] = OUTLIER_COLOR

    topic_label_by_id: Dict[int, str] = {}
    for d in docs_raw:
        topic_label_by_id.setdefault(d["topic_id"], d.get("topic_label") or f"topic_{d['topic_id']}")

    topics = [
        {"id": tid, "name": topic_label_by_id.get(tid, f"topic_{tid}"), "color": topic_color.get(tid, "#8B8580")}
        for tid in topic_ids
    ]

    documents = []
    docs_by_topic: Dict[int, list] = {}
    for d in docs_raw:
        tid = d["topic_id"]
        decade = d.get("decade")
        if decade is None:
            decade = document_decade_from_text(d.get("text", "") or "")
        decade_str = str(decade) if decade is not None else "غير معروف"
        doc = {
            "x": d["topic_x"],
            "y": d["topic_y"],
            "topic_id": tid,
            "topic_name": topic_label_by_id.get(tid, f"topic_{tid}"),
            "color": topic_color.get(tid, "#8B8580"),
            "title": d.get("title") or (d.get("doc_id") or "")[:16],
            "language": LANGUAGE_NAME_AR.get(d.get("language"), d.get("language") or "غير محدد"),
            "source": SOURCE_NAME_AR.get(d.get("source_id"), d.get("source_id") or "غير معروف"),
            "decade": decade_str,
        }
        documents.append(doc)
        docs_by_topic.setdefault(tid, []).append(d)

    topic_details = {}
    for tid, docs in docs_by_topic.items():
        if tid == -1:
            continue
        label = topic_label_by_id.get(tid, f"topic_{tid}")
        top_words = [w.strip() for w in label.split("/") if w.strip()]
        example_docs = [d.get("title") for d in docs[:3] if d.get("title")]
        topic_details[tid] = {
            "name": label,
            "doc_count": len(docs),
            "top_words": top_words,
            "example_docs": example_docs,
        }

    sources_present = sorted({doc["source"] for doc in documents})
    languages_present = sorted({doc["language"] for doc in documents})
    decades_present = sorted(
        {doc["decade"] for doc in documents if doc["decade"] != "غير معروف"}, key=int
    )

    return {
        "topics": topics,
        "documents": documents,
        "topic_details": topic_details,
        "filters": {
            "sources": ["الكل"] + sources_present,
            "languages": ["الكل"] + languages_present,
            "decades": ["الكل"] + decades_present,
        },
    }


# ============================================================
# TIMELINE
# ============================================================

# General historical context, not derived from the corpus/pipeline —
# editorial framing for the era card, same role as an axis caption.
_ERA_INFO = {
    "1800": "فترة الحكم العثماني المبكر - وثائق محدودة عن الحياة الثقافية في فلسطين",
    "1900": "نهاية الحكم العثماني وبدء الانتداب البريطاني - تنامي الوعي الوطني الفلسطيني",
    "1940": "فترة النكبة والتأسيس - تحول جذري في البنية الاجتماعية والثقافية الفلسطينية",
    "1960": "ما بعد النكسة - صعود حركات المقاومة وتطور الأدب الفلسطيني",
    "1980": "الانتفاضة الأولى - تنامي الهوية الثقافية والفنية كأداة مقاومة",
    "2000": "الانتفاضة الثانية والعصر الرقمي - توسع الإنتاج الثقافي والأدبي الفلسطيني",
}


@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_timeline_data() -> Dict[str, Any]:
    report = load_json_report(DEFAULT_TEMPORAL_REPORT_PATH)
    if not report:
        return {"docs_by_decade": [], "available_terms": [], "term_frequencies": {}, "era_info": {}}

    decades = report["decade_distribution"]
    docs_by_decade = [
        {"decade": str(d), "count": c} for d, c in sorted(decades.items(), key=lambda kv: int(kv[0]))
    ]

    term_freq_raw = report["term_frequency_per_1000_words_by_decade"]
    term_frequencies = {
        term: [
            {"year": int(decade), "freq": freq}
            for decade, freq in sorted(series.items(), key=lambda kv: int(kv[0]))
        ]
        for term, series in term_freq_raw.items()
    }

    return {
        "docs_by_decade": docs_by_decade,
        "available_terms": list(term_freq_raw.keys()),
        "term_frequencies": term_frequencies,
        "era_info": _ERA_INFO,
    }


# ============================================================
# BIAS METER
# ============================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_bias_data() -> Dict[str, Any]:
    report = load_json_report(DEFAULT_BIAS_REPORT_PATH)
    if not report:
        return {
            "dimension_scores": [],
            "source_comparison": [],
            "stacked_categories": [],
            "weat_score": 0.0,
            "weat_description": "لا توجد بيانات قياس تحيز بعد — شغّل scripts/run_bias_measurement.py.",
            "filters": {"sources": ["الكل"], "languages": ["الكل"], "decades": ["الكل"]},
        }

    dist_by_source = report.get("category_distribution_by_source", {})
    source_ar = {sid: SOURCE_NAME_AR.get(sid, sid) for sid in dist_by_source}

    all_categories = set()
    for dist in dist_by_source.values():
        all_categories.update(dist.keys())
    all_categories.discard("uncategorized")
    ordered = [c for c in _CATEGORY_STACK_ORDER if c in all_categories]
    ordered += sorted(all_categories - set(ordered))

    def _known_total(dist: Dict[str, float]) -> float:
        return sum(v for k, v in dist.items() if k != "uncategorized") or 1.0

    dimension_scores = []
    for cat in ordered:
        per_source_pct = {
            source_ar.get(sid, sid): round(dist.get(cat, 0.0) / _known_total(dist) * 100, 1)
            for sid, dist in dist_by_source.items()
        }
        avg_value = round(sum(per_source_pct.values()) / len(per_source_pct), 1) if per_source_pct else 0.0
        label = CATEGORY_AR.get(cat, cat)
        dimension_scores.append({
            "dimension": label,
            "dimension_en": cat,
            "value": avg_value,
            "sources": per_source_pct,
            "description": f'حصة فئة "{label}" من الوثائق المصنَّفة لكل مصدر (بعد استبعاد غير المصنف).',
        })
    dimension_scores.sort(key=lambda d: -d["value"])
    dimension_scores = dimension_scores[:MAX_GAUGE_DIMENSIONS]

    stacked_categories = [(cat, CATEGORY_AR.get(cat, cat)) for cat in ordered]
    source_comparison = []
    for sid, dist in dist_by_source.items():
        row: Dict[str, Any] = {"source": source_ar.get(sid, sid)}
        known_total = _known_total(dist)
        for cat in ordered:
            row[cat] = round(dist.get(cat, 0.0) / known_total * 100, 1)
        source_comparison.append(row)

    weat = report.get("weat", {})

    return {
        "dimension_scores": dimension_scores,
        "source_comparison": source_comparison,
        "stacked_categories": stacked_categories,
        "weat_score": weat.get("effect_size", 0.0),
        "weat_description": weat.get("interpretation", ""),
        "filters": {
            "sources": ["الكل"] + list(source_ar.values()),
            "languages": ["الكل", "العربية", "الإنجليزية"],
            "decades": ["الكل"],
        },
    }


# ============================================================
# KNOWLEDGE GRAPH EXPLORER
# ============================================================

@st.cache_data(ttl=CACHE_TTL_SECONDS, show_spinner=False)
def get_kg_data() -> Dict[str, Any]:
    graph = _get_graph()
    if graph is None:
        return {"stats": {"total_entities": 0, "total_relations": 0, "linked_to_wikidata": 0}, "sample_entities": []}

    linked = sum(1 for _, data in graph.nodes(data=True) if data.get("wikidata_qid"))
    degrees = dict(graph.degree())
    top_names = [
        graph.nodes[n].get("canonical_name")
        for n, _ in sorted(degrees.items(), key=lambda kv: -kv[1])[:15]
        if graph.nodes[n].get("canonical_name")
    ]

    return {
        "stats": {
            "total_entities": graph.number_of_nodes(),
            "total_relations": graph.number_of_edges(),
            "linked_to_wikidata": linked,
        },
        "sample_entities": top_names,
    }


def search_knowledge_graph(query: str) -> Dict[str, Any]:
    from src.knowlegde_graph.graph_store import find_entities_by_name

    graph = _get_graph()
    empty = {"center": {"name": query, "type": "ENTITY", "wikidata": None}, "neighbors": []}
    if graph is None:
        return empty

    matches = find_entities_by_name(graph, query)
    if not matches:
        return empty

    entity_id = matches[0]
    node = graph.nodes[entity_id]
    center = {
        "name": node.get("canonical_name", query),
        "type": node.get("type", "MISC"),
        "wikidata": node.get("wikidata_qid") or None,
    }

    neighbors = []
    for _, target, data in graph.out_edges(entity_id, data=True):
        neighbors.append({
            "name": graph.nodes[target].get("canonical_name", target),
            "type": graph.nodes[target].get("type", "MISC"),
            "relation": data.get("predicate", ""),
            "direction": "out",
        })
    for source, _, data in graph.in_edges(entity_id, data=True):
        neighbors.append({
            "name": graph.nodes[source].get("canonical_name", source),
            "type": graph.nodes[source].get("type", "MISC"),
            "relation": data.get("predicate", ""),
            "direction": "in",
        })

    return {"center": center, "neighbors": neighbors}


# ============================================================
# ASK (live RAG — not cached, every question hits the pipeline fresh)
# ============================================================

def ask_question(question: str) -> Dict[str, Any]:
    try:
        from src.rag.config import RagConfig
        from src.rag.db import get_connection
        from src.rag.embedder import Embedder
        from src.rag.generator import OllamaGenerator
        from src.rag.pipeline import RAGPipeline
        from src.rag.retriever import Retriever

        config = RagConfig.load()
        embedder = Embedder(config.embedding)
        conn = get_connection()
        try:
            pipeline = RAGPipeline(Retriever(conn, embedder, config), OllamaGenerator(config.generation), config)
            answer = pipeline.ask(question)
        finally:
            conn.close()

        citations = [
            {
                "id": c.index,
                "title": c.title or "بدون عنوان",
                "source": c.title or c.doc_id[:16],
                "url": c.source_url,
            }
            for c in answer.citations
        ]
        return {
            "answer": answer.text,
            "citations": citations,
            "metadata": {"sources_used": len(citations), "chunks_retrieved": len(citations)},
        }
    except Exception as exc:  # noqa: BLE001 - show a friendly Arabic error, don't crash the dashboard
        return {
            "answer": f"تعذّر الإجابة على هذا السؤال الآن (تأكد من تشغيل Postgres وOllama): {exc}",
            "citations": [],
            "metadata": {},
        }
