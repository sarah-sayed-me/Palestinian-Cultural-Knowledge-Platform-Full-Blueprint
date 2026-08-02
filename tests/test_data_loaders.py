import json

from src.frontend.data_loaders import (
    load_corpus_overview,
    load_json_report,
    load_kg_graph_summary,
    load_topic_distribution,
)
from src.knowlegde_graph.graph_store import build_graph, save_graph
from src.knowlegde_graph.schemas import KGEntity, KGRelation, make_entity_id, make_relation_id


def _write_jsonl(path, docs):
    with path.open("w", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(json.dumps(doc, ensure_ascii=False) + "\n")


def test_load_corpus_overview_counts_by_source_and_language(tmp_path):
    path = tmp_path / "docs.jsonl"
    _write_jsonl(
        path,
        [
            {"source_id": "wafa-news", "language": "ar-MSA"},
            {"source_id": "wafa-news", "language": "ar-MSA"},
            {"source_id": "gdelt", "language": "en"},
        ],
    )

    overview = load_corpus_overview([path])

    assert overview["total_documents"] == 3
    assert overview["by_source"] == {"wafa-news": 2, "gdelt": 1}
    assert overview["by_language"] == {"ar-MSA": 2, "en": 1}


def test_load_corpus_overview_skips_missing_files(tmp_path):
    overview = load_corpus_overview([tmp_path / "does_not_exist.jsonl"])

    assert overview["total_documents"] == 0


def test_load_json_report_returns_none_when_missing(tmp_path):
    assert load_json_report(tmp_path / "missing.json") is None


def test_load_json_report_reads_existing_file(tmp_path):
    path = tmp_path / "report.json"
    path.write_text(json.dumps({"key": "value"}), encoding="utf-8")

    assert load_json_report(path) == {"key": "value"}


def test_load_topic_distribution_counts_labels_across_files(tmp_path):
    path1 = tmp_path / "a.topics.jsonl"
    path2 = tmp_path / "b.topics.jsonl"
    _write_jsonl(path1, [{"topic_label": "فلسطين / تراث"}, {"topic_label": "فلسطين / تراث"}])
    _write_jsonl(path2, [{"topic_label": "غزة / احتلال"}])

    distribution = load_topic_distribution([path1, path2])

    assert distribution == {"فلسطين / تراث": 2, "غزة / احتلال": 1}


def test_load_topic_distribution_skips_docs_without_a_label(tmp_path):
    path = tmp_path / "a.topics.jsonl"
    _write_jsonl(path, [{"doc_id": "d1"}])

    assert load_topic_distribution([path]) == {}


def test_load_kg_graph_summary_returns_none_when_missing(tmp_path):
    assert load_kg_graph_summary(tmp_path / "missing.graphml") is None


def test_load_kg_graph_summary_reports_nodes_edges_and_top_degree(tmp_path):
    jaffa = KGEntity(
        entity_id=make_entity_id("يافا", "LOCATION"), canonical_name="يافا", type="LOCATION", mention_count=5
    )
    sea = KGEntity(
        entity_id=make_entity_id("البحر", "LOCATION"), canonical_name="البحر", type="LOCATION", mention_count=2
    )
    relation = KGRelation(
        relation_id=make_relation_id(jaffa.entity_id, "located_near", sea.entity_id, "doc-1"),
        subject_entity_id=jaffa.entity_id,
        predicate="located_near",
        object_entity_id=sea.entity_id,
        confidence=0.9,
        source_doc_id="doc-1",
    )
    graph = build_graph([jaffa, sea], [relation])
    path = tmp_path / "graph.graphml"
    save_graph(graph, path)

    summary = load_kg_graph_summary(path)

    assert summary["num_nodes"] == 2
    assert summary["num_edges"] == 1
    assert summary["top_entities_by_degree"][0]["canonical_name"] in {"يافا", "البحر"}
