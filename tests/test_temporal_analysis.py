from src.nlp.temporal_analysis import (
    bucket_documents_by_decade,
    decade_summary,
    document_decade_from_text,
    extract_year_mentions,
    term_frequency_by_decade,
)


def test_extract_year_mentions_finds_plausible_years():
    counts = extract_year_mentions("وقعت النكبة عام 1948 وتلاها احتلال عام 1967 ثم اتفاقية 1993.")

    assert counts[1948] == 1
    assert counts[1967] == 1
    assert counts[1993] == 1


def test_extract_year_mentions_ignores_implausible_numbers():
    counts = extract_year_mentions("بلغ عدد السكان 4500 نسمة، والمساحة 12000 دونم.")

    assert counts == {}


def test_document_decade_from_text_uses_most_frequent_year():
    text = "1948 1948 1948 1967"

    assert document_decade_from_text(text) == 1940


def test_document_decade_from_text_returns_none_when_no_year_found():
    assert document_decade_from_text("لا يوجد سنة هنا على الإطلاق") is None


def test_bucket_documents_prefers_metadata_decade_over_content():
    docs = [{"doc_id": "d1", "decade": 1990, "text": "نص يذكر عام 1948 كثيرا"}]

    buckets = bucket_documents_by_decade(docs)

    assert buckets[1990][0]["doc_id"] == "d1"
    assert 1940 not in buckets


def test_bucket_documents_falls_back_to_content_when_decade_missing():
    docs = [{"doc_id": "d1", "decade": None, "text": "وقعت الحرب عام 1967 وعام 1967 مرة أخرى"}]

    buckets = bucket_documents_by_decade(docs)

    assert buckets[1960][0]["doc_id"] == "d1"


def test_bucket_documents_skips_docs_with_no_temporal_signal_at_all():
    docs = [{"doc_id": "d1", "decade": None, "text": "لا يوجد أي سنة هنا"}]

    buckets = bucket_documents_by_decade(docs)

    assert buckets == {}


def test_decade_summary_counts_per_bucket():
    buckets = {1940: [{"doc_id": "a"}, {"doc_id": "b"}], 1960: [{"doc_id": "c"}]}

    assert decade_summary(buckets) == {1940: 2, 1960: 1}


def test_term_frequency_by_decade_computes_per_1000_words():
    buckets = {
        1940: [{"text": "فلسطين فلسطين أرض", "word_count": 3}],
        1960: [{"text": "أرض أرض أرض أرض", "word_count": 4}],
    }

    freq = term_frequency_by_decade(buckets, ["فلسطين"])

    assert freq["فلسطين"][1940] == round(2 / 3 * 1000, 4)
    assert freq["فلسطين"][1960] == 0.0
