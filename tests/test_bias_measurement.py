import numpy as np

from src.nlp.bias_measurement import (
    OllamaFramingProbe,
    category_distribution_by_source,
    parse_framing_response,
    total_variation_distance,
    weat_effect_size,
)
from src.rag.config import GenerationConfig


def test_category_distribution_by_source_normalizes_to_fractions():
    docs = [
        {"source_id": "wafa", "category": "conflict"},
        {"source_id": "wafa", "category": "conflict"},
        {"source_id": "wafa", "category": "culture"},
        {"source_id": "wikipedia-ar", "category": "culture"},
        {"source_id": "wikipedia-ar", "category": "culture"},
    ]

    distributions = category_distribution_by_source(docs)

    assert distributions["wafa"]["conflict"] == 0.6667
    assert distributions["wafa"]["culture"] == 0.3333
    assert distributions["wikipedia-ar"]["culture"] == 1.0


def test_category_distribution_skips_docs_missing_category_or_source():
    docs = [{"source_id": "wafa"}, {"category": "conflict"}, {"source_id": "wafa", "category": "conflict"}]

    distributions = category_distribution_by_source(docs)

    assert distributions == {"wafa": {"conflict": 1.0}}


def test_total_variation_distance_identical_distributions_is_zero():
    dist = {"conflict": 0.5, "culture": 0.5}

    assert total_variation_distance(dist, dist) == 0.0


def test_total_variation_distance_disjoint_distributions_is_one():
    a = {"conflict": 1.0}
    b = {"culture": 1.0}

    assert total_variation_distance(a, b) == 1.0


def test_total_variation_distance_handles_categories_missing_from_one_side():
    a = {"conflict": 0.5, "culture": 0.5}
    b = {"conflict": 0.5, "heritage": 0.5}

    # overlap on conflict (0.5=0.5, contributes 0), culture and heritage each
    # differ by 0.5 -> (0.5+0.5)/2 = 0.5
    assert total_variation_distance(a, b) == 0.5


def test_weat_effect_size_positive_when_target_x_associates_more_with_attribute_a():
    # target_x is identical to attribute_a direction, target_y to attribute_b
    target_x = [np.array([1.0, 0.0])]
    target_y = [np.array([0.0, 1.0])]
    attr_a = [np.array([1.0, 0.0])]
    attr_b = [np.array([0.0, 1.0])]

    effect_size, associations = weat_effect_size(target_x, target_y, attr_a, attr_b)

    assert effect_size > 0
    assert associations["target_x_associations"][0] > associations["target_y_associations"][0]


def test_weat_effect_size_zero_when_no_variance_in_associations():
    # every target word is equidistant from both attribute sets -> std is 0
    target_x = [np.array([1.0, 1.0])]
    target_y = [np.array([1.0, 1.0])]
    attr_a = [np.array([1.0, 0.0])]
    attr_b = [np.array([0.0, 1.0])]

    effect_size, _ = weat_effect_size(target_x, target_y, attr_a, attr_b)

    assert effect_size == 0.0


def test_parse_framing_response_extracts_valid_framing():
    assert parse_framing_response('{"framing": "conflict", "confidence": 0.8}') == ("conflict", 0.8)


def test_parse_framing_response_accepts_mixed():
    assert parse_framing_response('{"framing": "mixed", "confidence": 0.5}') == ("mixed", 0.5)


def test_parse_framing_response_rejects_invalid_framing_value():
    assert parse_framing_response('{"framing": "neutral", "confidence": 0.5}') is None


def test_parse_framing_response_returns_none_for_garbage():
    assert parse_framing_response("not json at all") is None


def test_ollama_framing_probe_probe_delegates_to_call_llm_and_parses(monkeypatch):
    probe = OllamaFramingProbe.__new__(OllamaFramingProbe)
    probe.config = GenerationConfig()
    monkeypatch.setattr(probe, "_call_llm", lambda *a, **k: '{"framing": "non_conflict", "confidence": 0.7}')

    assert probe.probe("نص عن التراث") == ("non_conflict", 0.7)
