import json

from src.knowlegde_graph.relations import (
    OllamaRelationExtractor,
    candidate_pairs,
    parse_llm_response,
)
from src.ingestion.entity_extractor import split_sentences
from src.knowlegde_graph.schemas import make_entity_id


def _entity(text, normalized, entity_type, sentence_index, start_char):
    return {
        "text": text,
        "normalized": normalized,
        "type": entity_type,
        "mention_count": 1,
        "confidence": 0.75,
        "positions": [{"start_char": start_char, "end_char": start_char + len(text), "sentence_index": sentence_index}],
    }


def _doc(doc_id, text, entities):
    return {"doc_id": doc_id, "text": text, "entities": entities}


def test_parse_llm_response_extracts_predicate_and_confidence():
    raw = '{"predicate": "located_in", "confidence": 0.9}'

    result = parse_llm_response(raw)

    assert result == ("located_in", 0.9)


def test_parse_llm_response_tolerates_surrounding_text_and_markdown_fence():
    raw = 'Sure, here is the answer:\n```json\n{"predicate": "part_of", "confidence": 0.8}\n```'

    result = parse_llm_response(raw)

    assert result == ("part_of", 0.8)


def test_parse_llm_response_returns_none_for_unparseable_text():
    assert parse_llm_response("I cannot determine a relation.") is None


def test_parse_llm_response_returns_none_when_predicate_missing():
    assert parse_llm_response('{"confidence": 0.9}') is None


def test_candidate_pairs_yields_entities_co_occurring_in_the_same_sentence():
    text = "زارت وفود دولية مدينة القدس. تقع يافا على ساحل البحر المتوسط."
    doc = _doc(
        "doc-1",
        text,
        [
            _entity("القدس", "القدس", "LOCATION", sentence_index=0, start_char=text.index("القدس")),
            _entity("يافا", "يافا", "LOCATION", sentence_index=1, start_char=text.index("يافا")),
            _entity(
                "البحر المتوسط",
                "البحر المتوسط",
                "LOCATION",
                sentence_index=1,
                start_char=text.index("البحر المتوسط"),
            ),
        ],
    )
    sentences = split_sentences(text)

    pairs = list(candidate_pairs(doc, sentences))

    # القدس is alone in sentence 0 -> no pair; يافا + البحر المتوسط co-occur in sentence 1 -> one pair
    assert len(pairs) == 1
    sentence, subject, obj = pairs[0]
    assert subject.canonical_name == "يافا"
    assert obj.canonical_name == "البحر المتوسط"
    assert sentence.index == 1


def test_candidate_pairs_caps_entities_per_sentence():
    text = "أ ب ج د هـ في جملة واحدة."
    entities = [
        _entity(letter, letter, "MISC", sentence_index=0, start_char=text.index(letter))
        for letter in ["أ", "ب", "ج", "د", "هـ"]
    ]
    doc = _doc("doc-1", text, entities)
    sentences = split_sentences(text)

    pairs = list(candidate_pairs(doc, sentences))

    # 5 distinct entities capped to 4 -> C(4,2) = 6 pairs, not C(5,2) = 10
    assert len(pairs) == 6


class _FakeChatResponse:
    def __init__(self, content):
        class _Msg:
            pass

        self.message = _Msg()
        self.message.content = content


def test_extract_relation_builds_kg_relation_from_llm_response(monkeypatch):
    extractor = OllamaRelationExtractor.__new__(OllamaRelationExtractor)
    from src.rag.config import GenerationConfig

    extractor.config = GenerationConfig()
    monkeypatch.setattr(
        extractor, "_call_llm", lambda *a, **k: '{"predicate": "located_in", "confidence": 0.85}'
    )

    text = "تقع يافا على ساحل البحر المتوسط."
    sentence = split_sentences(text)[0]
    from src.knowlegde_graph.relations import EntityMention

    subject = EntityMention(
        entity_id=make_entity_id("يافا", "LOCATION"),
        canonical_name="يافا",
        type="LOCATION",
        surface="يافا",
        start_char=4,
    )
    obj = EntityMention(
        entity_id=make_entity_id("البحر المتوسط", "LOCATION"),
        canonical_name="البحر المتوسط",
        type="LOCATION",
        surface="البحر المتوسط",
        start_char=18,
    )

    relation = extractor.extract_relation("doc-1", sentence, subject, obj)

    assert relation is not None
    assert relation.predicate == "located_in"
    assert relation.confidence == 0.85
    assert relation.subject_entity_id == subject.entity_id
    assert relation.object_entity_id == obj.entity_id
    assert relation.source_doc_id == "doc-1"
    assert relation.evidence_sentence == text


def test_extract_relation_returns_none_for_null_predicate(monkeypatch):
    extractor = OllamaRelationExtractor.__new__(OllamaRelationExtractor)
    from src.rag.config import GenerationConfig

    extractor.config = GenerationConfig()
    monkeypatch.setattr(extractor, "_call_llm", lambda *a, **k: '{"predicate": null, "confidence": 0.0}')

    text = "بعض الجملة هنا."
    sentence = split_sentences(text)[0]
    from src.knowlegde_graph.relations import EntityMention

    subject = EntityMention(entity_id="e1", canonical_name="أ", type="MISC", surface="أ", start_char=0)
    obj = EntityMention(entity_id="e2", canonical_name="ب", type="MISC", surface="ب", start_char=1)

    assert extractor.extract_relation("doc-1", sentence, subject, obj) is None


def test_extract_relation_returns_none_below_min_confidence(monkeypatch):
    extractor = OllamaRelationExtractor.__new__(OllamaRelationExtractor)
    from src.rag.config import GenerationConfig

    extractor.config = GenerationConfig()
    monkeypatch.setattr(
        extractor, "_call_llm", lambda *a, **k: '{"predicate": "related_to", "confidence": 0.1}'
    )

    text = "بعض الجملة هنا."
    sentence = split_sentences(text)[0]
    from src.knowlegde_graph.relations import EntityMention

    subject = EntityMention(entity_id="e1", canonical_name="أ", type="MISC", surface="أ", start_char=0)
    obj = EntityMention(entity_id="e2", canonical_name="ب", type="MISC", surface="ب", start_char=1)

    assert extractor.extract_relation("doc-1", sentence, subject, obj) is None


def test_extract_document_caps_llm_calls_via_max_pairs(monkeypatch):
    extractor = OllamaRelationExtractor.__new__(OllamaRelationExtractor)
    from src.rag.config import GenerationConfig

    extractor.config = GenerationConfig()
    call_count = {"n": 0}

    def fake_call(*a, **k):
        call_count["n"] += 1
        return '{"predicate": "related_to", "confidence": 0.9}'

    monkeypatch.setattr(extractor, "_call_llm", fake_call)

    text = "أ ب ج د هـ في جملة واحدة."
    entities = [
        _entity(letter, letter, "MISC", sentence_index=0, start_char=text.index(letter))
        for letter in ["أ", "ب", "ج", "د", "هـ"]
    ]
    doc = _doc("doc-1", text, entities)

    relations = extractor.extract_document(doc, max_pairs=2)

    assert call_count["n"] == 2
    assert len(relations) == 2
