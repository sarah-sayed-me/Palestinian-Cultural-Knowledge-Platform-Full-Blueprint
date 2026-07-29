from src.knowlegde_graph.canonicalize import canonicalize_entities
from src.knowlegde_graph.schemas import make_entity_id


def _doc(doc_id, entities):
    return {"doc_id": doc_id, "entities": entities}


def test_merges_same_entity_across_documents():
    docs = [
        _doc(
            "doc-1",
            [{"text": "يافا", "normalized": "يافا", "type": "LOCATION", "mention_count": 2, "confidence": 0.75}],
        ),
        _doc(
            "doc-2",
            [{"text": "يافا", "normalized": "يافا", "type": "LOCATION", "mention_count": 1, "confidence": 0.75}],
        ),
    ]

    entities = canonicalize_entities(docs)

    assert len(entities) == 1
    entity = entities[0]
    assert entity.canonical_name == "يافا"
    assert entity.type == "LOCATION"
    assert entity.mention_count == 3
    assert entity.source_doc_ids == ["doc-1", "doc-2"]
    assert entity.entity_id == make_entity_id("يافا", "LOCATION")


def test_keeps_distinct_types_of_the_same_text_separate():
    docs = [
        _doc(
            "doc-1",
            [
                {"text": "غزة", "normalized": "غزة", "type": "LOCATION", "mention_count": 1, "confidence": 0.75},
                {"text": "غزة", "normalized": "غزة", "type": "ORGANIZATION", "mention_count": 1, "confidence": 0.75},
            ],
        )
    ]

    entities = canonicalize_entities(docs)

    assert len(entities) == 2
    types = {e.type for e in entities}
    assert types == {"LOCATION", "ORGANIZATION"}


def test_prefers_heritage_dictionary_canonical_form():
    docs = [
        _doc(
            "doc-1",
            [
                {
                    "text": "بالكوفية",
                    "normalized": "كوفية",
                    "type": "HERITAGE_CLOTHING",
                    "canonical": "كوفية",
                    "mention_count": 1,
                    "confidence": 1.0,
                }
            ],
        )
    ]

    entities = canonicalize_entities(docs)

    assert entities[0].canonical_name == "كوفية"


def test_a_document_appearing_twice_for_the_same_entity_is_only_listed_once_in_source_doc_ids():
    docs = [
        _doc(
            "doc-1",
            [
                {"text": "حيفا", "normalized": "حيفا", "type": "LOCATION", "mention_count": 1, "confidence": 0.75},
                {"text": "حيفا", "normalized": "حيفا", "type": "LOCATION", "mention_count": 1, "confidence": 0.75},
            ],
        )
    ]

    entities = canonicalize_entities(docs)

    assert entities[0].source_doc_ids == ["doc-1"]
    assert entities[0].mention_count == 2


def test_entities_missing_normalized_or_type_are_skipped():
    docs = [_doc("doc-1", [{"text": "x", "mention_count": 1}])]

    entities = canonicalize_entities(docs)

    assert entities == []


def test_sorted_by_mention_count_descending():
    docs = [
        _doc(
            "doc-1",
            [
                {"text": "أ", "normalized": "أ", "type": "MISC", "mention_count": 1, "confidence": 0.75},
                {"text": "ب", "normalized": "ب", "type": "MISC", "mention_count": 5, "confidence": 0.75},
            ],
        )
    ]

    entities = canonicalize_entities(docs)

    assert [e.canonical_name for e in entities] == ["ب", "أ"]
