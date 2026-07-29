from src.ingestion.collectors.gdelt_collector import GdeltCollector

ARTICLE_HTML = """
<html><head><title>Fallback Title</title></head><body>
<p>This short bit is nav junk.</p>
<p>This is a real paragraph of article content that is long enough to survive the minimum paragraph length filter used by the extraction heuristic.</p>
<p>This is a second real paragraph, also long enough, continuing the story with more substantive detail about the events described in the piece.</p>
</body></html>
"""


def _collector(**kwargs):
    return GdeltCollector(
        source_config={"queries": ["sourcecountry:PS"], "max_retries": 1, **kwargs},
        credibility_map={},
        max_docs=kwargs.get("max_docs", 10),
        request_delay=0,
    )


def test_robots_allow_true_when_no_robots_txt_found(monkeypatch):
    collector = _collector()

    class FakeResponse:
        status_code = 404
        text = ""

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    assert collector._robots_allow("https://example.com/article-1") is True


def test_robots_allow_respects_explicit_disallow(monkeypatch):
    collector = _collector()

    class FakeResponse:
        status_code = 200
        text = "User-agent: *\nDisallow: /"

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    assert collector._robots_allow("https://example.com/article-1") is False


def test_robots_allow_respects_specific_path_disallow(monkeypatch):
    collector = _collector()

    class FakeResponse:
        status_code = 200
        text = "User-agent: *\nDisallow: /private/\nAllow: /"

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    assert collector._robots_allow("https://example.com/news/article-1") is True
    assert collector._robots_allow("https://example.com/private/secret") is False


def test_robots_allow_defaults_true_on_fetch_failure(monkeypatch):
    collector = _collector()

    def raise_error(*a, **k):
        raise httpx_error

    import httpx as httpx_module

    httpx_error = httpx_module.ConnectError("boom")
    monkeypatch.setattr(collector._client, "get", raise_error)

    assert collector._robots_allow("https://example.com/article-1") is True


def test_robots_cache_only_fetches_once_per_domain(monkeypatch):
    collector = _collector()
    call_count = {"n": 0}

    class FakeResponse:
        status_code = 200
        text = "User-agent: *\nAllow: /"

    def counting_get(url, **kwargs):
        call_count["n"] += 1
        return FakeResponse()

    monkeypatch.setattr(collector._client, "get", counting_get)

    collector._robots_allow("https://example.com/a")
    collector._robots_allow("https://example.com/b")

    assert call_count["n"] == 1


def test_paragraph_extraction_filters_short_junk_paragraphs(monkeypatch):
    collector = _collector()
    monkeypatch.setattr(collector, "_robots_allow", lambda url: True)

    class FakeResponse:
        text = ARTICLE_HTML

        def raise_for_status(self):
            pass

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    doc = collector._fetch_document({"url": "https://example.com/a", "title": "Real Title"})

    assert doc is not None
    assert "nav junk" not in doc.text
    assert "real paragraph" in doc.text or "ral paragraph" in doc.text  # normalize_arabic doesn't touch ASCII
    assert doc.title == "Real Title"


def test_parse_seendate_valid_and_invalid():
    parsed = GdeltCollector._parse_seendate("20260720T120000Z")
    assert parsed.year == 2026 and parsed.month == 7 and parsed.day == 20
    assert GdeltCollector._parse_seendate(None) is None
    assert GdeltCollector._parse_seendate("not-a-date") is None


def test_map_language_prefers_gdelt_field_over_detection():
    assert GdeltCollector._map_language("Arabic", "some text") == "ar-MSA"
    assert GdeltCollector._map_language("English", "some text") == "en"


def test_collect_skips_urls_disallowed_by_robots(monkeypatch):
    collector = _collector(max_docs=5)
    collector.max_docs = 5
    monkeypatch.setattr(
        collector,
        "_search",
        lambda query: [
            {"url": "https://blocked.example.com/a", "title": "Blocked"},
            {"url": "https://allowed.example.com/a", "title": "Allowed"},
        ],
    )
    monkeypatch.setattr(
        collector, "_robots_allow", lambda url: "allowed" in url
    )

    class FakeResponse:
        text = ARTICLE_HTML

        def raise_for_status(self):
            pass

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    docs = list(collector.collect())

    assert len(docs) == 1
    assert docs[0].title == "Allowed"
