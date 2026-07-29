from src.knowlegde_graph.graph_store import (
    build_graph,
    find_entities_by_name,
    load_graph,
    neighbors_of,
    save_graph,
)
from src.knowlegde_graph.schemas import KGEntity, KGRelation, make_entity_id, make_relation_id


def _entities():
    jaffa = KGEntity(
        entity_id=make_entity_id("يافا", "LOCATION"),
        canonical_name="يافا",
        type="LOCATION",
        wikidata_qid="Q1234",
        mention_count=5,
        source_doc_ids=["doc-1"],
    )
    sea = KGEntity(
        entity_id=make_entity_id("البحر المتوسط", "LOCATION"),
        canonical_name="البحر المتوسط",
        type="LOCATION",
        mention_count=2,
        source_doc_ids=["doc-1"],
    )
    return jaffa, sea


def _relation(jaffa, sea):
    return KGRelation(
        relation_id=make_relation_id(jaffa.entity_id, "located_in", sea.entity_id, "doc-1"),
        subject_entity_id=jaffa.entity_id,
        predicate="located_in",
        object_entity_id=sea.entity_id,
        confidence=0.9,
        source_doc_id="doc-1",
        evidence_sentence="تقع يافا على ساحل البحر المتوسط.",
    )


def test_build_graph_adds_nodes_and_edges():
    jaffa, sea = _entities()
    relation = _relation(jaffa, sea)

    graph = build_graph([jaffa, sea], [relation])

    assert graph.number_of_nodes() == 2
    assert graph.number_of_edges() == 1
    assert graph.nodes[jaffa.entity_id]["canonical_name"] == "يافا"
    assert graph.nodes[jaffa.entity_id]["wikidata_qid"] == "Q1234"
    edge_data = graph.get_edge_data(jaffa.entity_id, sea.entity_id, key=relation.relation_id)
    assert edge_data["predicate"] == "located_in"


def test_build_graph_skips_relations_referencing_unknown_entities():
    jaffa, sea = _entities()
    bad_relation = KGRelation(
        relation_id="r1",
        subject_entity_id=jaffa.entity_id,
        predicate="located_in",
        object_entity_id="unknown-entity",
        confidence=0.9,
        source_doc_id="doc-1",
    )

    graph = build_graph([jaffa, sea], [bad_relation])

    assert graph.number_of_edges() == 0
    assert graph.graph["skipped_relations"] == 1


def test_save_and_load_graph_roundtrip(tmp_path):
    jaffa, sea = _entities()
    relation = _relation(jaffa, sea)
    graph = build_graph([jaffa, sea], [relation])
    path = tmp_path / "graph.graphml"

    save_graph(graph, path)
    loaded = load_graph(path)

    assert loaded.number_of_nodes() == 2
    assert loaded.number_of_edges() == 1
    assert loaded.nodes[jaffa.entity_id]["canonical_name"] == "يافا"


def test_find_entities_by_name_substring_match():
    jaffa, sea = _entities()
    graph = build_graph([jaffa, sea], [])

    matches = find_entities_by_name(graph, "يافا")

    assert matches == [jaffa.entity_id]


def test_neighbors_of_returns_outgoing_relations_with_target_metadata():
    jaffa, sea = _entities()
    relation = _relation(jaffa, sea)
    graph = build_graph([jaffa, sea], [relation])

    neighbors = neighbors_of(graph, jaffa.entity_id)

    assert len(neighbors) == 1
    assert neighbors[0]["predicate"] == "located_in"
    assert neighbors[0]["target_canonical_name"] == "البحر المتوسط"


def test_neighbors_of_unknown_entity_returns_empty_list():
    jaffa, sea = _entities()
    graph = build_graph([jaffa, sea], [])

    assert neighbors_of(graph, "does-not-exist") == []
