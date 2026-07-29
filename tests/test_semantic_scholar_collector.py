from src.ingestion.collectors.semantic_scholar_collector import SemanticScholarCollector
from src.ingestion.schemas import LicenseStatus, SourceType

ABSTRACT = (
    "This paper examines Palestinian cultural heritage preservation through digital "
    "archives, exploring how oral history projects and community-led documentation "
    "efforts contribute to safeguarding intangible cultural practices for future "
    "generations across the diaspora and within historic Palestine."
)


def _collector(pages, **kwargs):
    collector = SemanticScholarCollector(
        source_config={"queries": ["Palestinian culture"], "max_retries": 1, **kwargs},
        credibility_map={"semanticscholar.org": {"tier": "tier_1", "score": 0.91}},
        max_docs=kwargs.get("max_docs", 10),
        request_delay=0,
    )
    call_count = {"n": 0}

    def fake_fetch_page(query, offset, limit):
        idx = call_count["n"]
        call_count["n"] += 1
        return pages[idx] if idx < len(pages) else {"data": []}

    collector._fetch_page = fake_fetch_page
    return collector


def test_collects_document_from_paper_with_abstract():
    pages = [
        {
            "data": [
                {
                    "paperId": "p1",
                    "title": "Digital Archives and Palestinian Heritage",
                    "abstract": ABSTRACT,
                    "url": "https://www.semanticscholar.org/paper/p1",
                    "year": 2021,
                    "authors": [{"name": "A. Researcher"}],
                    "venue": "Journal of Digital Humanities",
                    "externalIds": {"DOI": "10.1234/example"},
                }
            ]
        }
    ]
    collector = _collector(pages)

    docs = list(collector.collect())

    assert len(docs) == 1
    doc = docs[0]
    assert doc.title == "Digital Archives and Palestinian Heritage"
    assert doc.source_id == "semantic-scholar"
    assert doc.source_type == SourceType.ACADEMIC_PAPER.value
    assert doc.license_status == LicenseStatus.CLEAR.value
    assert doc.seed_category == "Palestinian culture"
    assert doc.source_name == "Journal of Digital Humanities"
    assert doc.tags == ["A. Researcher"]
    assert doc.date_published.year == 2021
    assert doc.text_raw == ABSTRACT


def test_skips_papers_without_an_abstract():
    pages = [{"data": [{"paperId": "p1", "title": "No Abstract Here", "abstract": None}]}]
    collector = _collector(pages)

    docs = list(collector.collect())

    assert docs == []


def test_deduplicates_papers_seen_across_pages_or_queries():
    pages = [
        {"data": [{"paperId": "p1", "title": "T", "abstract": ABSTRACT}]},
        {"data": [{"paperId": "p1", "title": "T", "abstract": ABSTRACT}]},  # same paper again
    ]
    collector = _collector(pages)
    collector.queries = ["query one", "query two"]

    docs = list(collector.collect())

    assert len(docs) == 1


def test_stops_at_max_docs():
    pages = [
        {
            "data": [
                {"paperId": f"p{i}", "title": f"T{i}", "abstract": ABSTRACT}
                for i in range(5)
            ]
        }
    ]
    collector = _collector(pages, max_docs=2)
    collector.max_docs = 2

    docs = list(collector.collect())

    assert len(docs) == 2


FULL_TEXT = "This is the full body of the paper. " * 30  # well over MIN_PDF_CHARS


def test_resolve_text_uses_abstract_when_no_oa_pdf():
    collector = SemanticScholarCollector(source_config={"queries": []}, max_docs=1, request_delay=0)

    text, is_full_text = collector._resolve_text({"abstract": ABSTRACT})

    assert text == ABSTRACT
    assert is_full_text is False


def test_resolve_text_uses_pdf_when_fetch_succeeds(monkeypatch):
    collector = SemanticScholarCollector(source_config={"queries": []}, max_docs=1, request_delay=0)
    monkeypatch.setattr(collector, "_fetch_pdf_text", lambda url: FULL_TEXT)

    text, is_full_text = collector._resolve_text(
        {"abstract": ABSTRACT, "openAccessPdf": {"url": "https://example.com/paper.pdf"}}
    )

    assert text == FULL_TEXT
    assert is_full_text is True


def test_resolve_text_falls_back_to_abstract_when_pdf_extraction_too_short(monkeypatch):
    collector = SemanticScholarCollector(
        source_config={"queries": [], "max_retries": 1}, max_docs=1, request_delay=0
    )
    monkeypatch.setattr(collector, "_fetch_pdf_text", lambda url: "too short")

    text, is_full_text = collector._resolve_text(
        {"abstract": ABSTRACT, "openAccessPdf": {"url": "https://example.com/paper.pdf"}}
    )

    assert text == ABSTRACT
    assert is_full_text is False


def test_resolve_text_falls_back_to_abstract_when_pdf_fetch_fails(monkeypatch):
    collector = SemanticScholarCollector(
        source_config={"queries": [], "max_retries": 1}, max_docs=1, request_delay=0
    )

    def raise_error(url):
        raise RuntimeError("network error")

    monkeypatch.setattr(collector, "_fetch_pdf_text", raise_error)
    monkeypatch.setattr(collector, "_sleep", lambda *a, **k: None)

    text, is_full_text = collector._resolve_text(
        {"abstract": ABSTRACT, "openAccessPdf": {"url": "https://example.com/paper.pdf"}}
    )

    assert text == ABSTRACT
    assert is_full_text is False


def test_fetch_pdf_text_rejects_non_pdf_content_type(monkeypatch):
    collector = SemanticScholarCollector(source_config={"queries": []}, max_docs=1, request_delay=0)

    class FakeResponse:
        headers = {"content-type": "text/html"}
        content = b"<html>not a pdf</html>"

        def raise_for_status(self):
            pass

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    try:
        collector._fetch_pdf_text("https://example.com/not-a-paper")
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_paper_to_document_collects_full_text_and_keeps_abstract_as_summary(monkeypatch):
    collector = SemanticScholarCollector(source_config={"queries": ["q"]}, max_docs=1, request_delay=0)
    monkeypatch.setattr(collector, "_fetch_pdf_text", lambda url: FULL_TEXT)

    doc = collector._paper_to_document(
        {
            "paperId": "p1",
            "title": "Full Text Paper",
            "abstract": ABSTRACT,
            "url": "https://www.semanticscholar.org/paper/p1",
            "openAccessPdf": {"url": "https://example.com/paper.pdf"},
        },
        "q",
    )

    assert doc is not None
    assert doc.word_count > len(ABSTRACT.split())  # full text is longer than the abstract alone
    assert doc.license_status == "clear"
    assert doc.summary is not None and "cultural heritage" in doc.summary


def test_search_returns_nothing_when_all_retries_fail(monkeypatch):
    collector = SemanticScholarCollector(
        source_config={"queries": ["x"], "max_retries": 1},
        max_docs=5,
        request_delay=0,
    )
    monkeypatch.setattr(collector, "_fetch_page", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(collector, "_sleep", lambda *a, **k: None)

    docs = list(collector.collect())

    assert docs == []
