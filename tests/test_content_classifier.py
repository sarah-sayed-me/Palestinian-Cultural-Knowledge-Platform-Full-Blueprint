from src.nlp.content_classifier import OllamaContentClassifier, parse_classification_response
from src.rag.config import GenerationConfig


def test_parse_classification_response_extracts_category_and_confidence():
    raw = '{"category": "heritage", "confidence": 0.85}'

    result = parse_classification_response(raw)

    assert result == ("heritage", 0.85)


def test_parse_classification_response_tolerates_surrounding_text():
    raw = 'Here is my answer:\n```json\n{"category": "food_cuisine", "confidence": 0.9}\n```'

    result = parse_classification_response(raw)

    assert result == ("food_cuisine", 0.9)


def test_parse_classification_response_rejects_unknown_category():
    raw = '{"category": "not_a_real_category", "confidence": 0.9}'

    assert parse_classification_response(raw) is None


def test_parse_classification_response_returns_none_for_unparseable_text():
    assert parse_classification_response("I cannot classify this.") is None


def test_parse_classification_response_accepts_uncategorized():
    raw = '{"category": "uncategorized", "confidence": 0.2}'

    assert parse_classification_response(raw) == ("uncategorized", 0.2)


def test_classify_uses_call_llm_and_parses_result(monkeypatch):
    classifier = OllamaContentClassifier.__new__(OllamaContentClassifier)
    classifier.config = GenerationConfig()
    monkeypatch.setattr(
        classifier, "_call_llm", lambda *a, **k: '{"category": "conflict", "confidence": 0.75}'
    )

    result = classifier.classify("عنوان", "نص عن نزاع مسلح")

    assert result == ("conflict", 0.75)


def test_classify_returns_none_when_llm_response_is_unusable(monkeypatch):
    classifier = OllamaContentClassifier.__new__(OllamaContentClassifier)
    classifier.config = GenerationConfig()
    monkeypatch.setattr(classifier, "_call_llm", lambda *a, **k: "")

    assert classifier.classify("عنوان", "نص") is None
