from src.knowlegde_graph.schemas import KGEntity, KGRelation, make_entity_id, make_relation_id


def test_make_entity_id_stable_and_type_sensitive():
    a = make_entity_id("كوفية", "HERITAGE_CLOTHING")
    b = make_entity_id("كوفية", "HERITAGE_CLOTHING")
    c = make_entity_id("كوفية", "PERSON")
    assert a == b
    assert a != c


def test_kg_entity_and_relation_construct():
    entity = KGEntity(
        entity_id=make_entity_id("يافا", "LOCATION"),
        canonical_name="يافا",
        type="LOCATION",
        mention_count=3,
        source_doc_ids=["doc-1", "doc-2"],
    )
    relation = KGRelation(
        relation_id=make_relation_id(entity.entity_id, "located_in", "entity-2", "doc-1"),
        subject_entity_id=entity.entity_id,
        predicate="located_in",
        object_entity_id="entity-2",
        confidence=0.8,
        source_doc_id="doc-1",
    )
    assert relation.subject_entity_id == entity.entity_id
