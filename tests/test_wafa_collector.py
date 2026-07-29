from datetime import date

from src.ingestion.collectors.wafa_collector import WafaCollector

SAMPLE_SITEMAP_XML = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">
  <url><loc>https://www.wafa.ps/news/2026/7/20/article-one-150647</loc></url>
  <url><loc>https://www.wafa.ps/news/2026/7/20/article-two-150648</loc></url>
  <url><loc>https://www.wafa.ps/other/not-an-article</loc></url>
</urlset>"""

SAMPLE_ARTICLE_HTML = """
<html><head>
<meta property="og:title" content="حالة الطقس: توالي ارتفاع درجات الحرارة" />
</head><body>
<div class="col-lg-8 col-12 mb-50">
الرئيسية محلية تاريخ النشر: 20/07/2026 07:50 ص حالة الطقس: توالي ارتفاع درجات الحرارة
رام الله 20-7-2026 وفا_ توقعت دائرة الأرصاد الجوية أن يكون الجو حارا اليوم في المناطق الجبلية
وشديد الحرارة في بقية المناطق مع بقاء درجات الحرارة أعلى من معدلها السنوي بشكل ملحوظ في جميع المحافظات.
</div>
</body></html>
"""


def _collector(**kwargs):
    return WafaCollector(
        source_config={"max_retries": 1, **kwargs},
        credibility_map={},
        max_docs=kwargs.get("max_docs", 10),
        request_delay=0,
    )


def test_extract_date_parses_pm_correctly():
    block = "تاريخ النشر: 20/07/2026 07:50 م"
    parsed = WafaCollector._extract_date(block)
    assert parsed.hour == 19
    assert parsed.year == 2026 and parsed.month == 7 and parsed.day == 20


def test_extract_date_parses_am_correctly():
    block = "تاريخ النشر: 20/07/2026 07:50 ص"
    parsed = WafaCollector._extract_date(block)
    assert parsed.hour == 7


def test_extract_date_returns_none_when_absent():
    assert WafaCollector._extract_date("no date here") is None


def test_extract_category_from_real_block_text_pattern():
    # "الرئيسية <category> تاريخ النشر: ..." — verified against real WAFA pages.
    assert WafaCollector._extract_category("الرئيسية محلية تاريخ النشر: 20/07/2026") == "محلية"
    assert WafaCollector._extract_category("الرئيسية رياضة تاريخ النشر: 20/07/2026") == "رياضة"


def test_extract_category_returns_none_when_pattern_absent():
    assert WafaCollector._extract_category("no matching pattern here") is None


def test_fetch_daily_sitemap_filters_to_news_urls(monkeypatch):
    collector = _collector()

    class FakeResponse:
        content = SAMPLE_SITEMAP_XML

        def raise_for_status(self):
            pass

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    urls = collector._fetch_daily_sitemap(date(2026, 7, 20))

    assert urls == [
        "https://www.wafa.ps/news/2026/7/20/article-one-150647",
        "https://www.wafa.ps/news/2026/7/20/article-two-150648",
    ]


def test_fetch_document_extracts_title_body_and_date(monkeypatch):
    collector = _collector()

    class FakeResponse:
        text = SAMPLE_ARTICLE_HTML

        def raise_for_status(self):
            pass

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    doc = collector._fetch_document("https://www.wafa.ps/news/2026/7/20/article-one-150647")

    assert doc is not None
    assert doc.title == "حالة الطقس: توالي ارتفاع درجات الحرارة"
    assert "توقعت دائرة الارصاد الجوية" in doc.text  # normalize_arabic converts أ->ا (only "الأرصاد" here); teh-marbuta is untouched
    assert doc.title not in doc.text  # body starts after the title, not repeating it
    assert doc.source_id == "wafa-news"
    assert doc.license_status == "needs_review"
    assert doc.date_published is not None
    assert doc.date_published.hour == 7
    assert doc.seed_category == "محلية"
    assert doc.tags == ["محلية"]


def test_collect_stops_at_max_docs(monkeypatch):
    collector = _collector(max_docs=1)
    collector.max_docs = 1
    monkeypatch.setattr(
        collector,
        "_iter_article_urls",
        lambda: iter(["https://www.wafa.ps/news/a", "https://www.wafa.ps/news/b"]),
    )

    class FakeResponse:
        text = SAMPLE_ARTICLE_HTML

        def raise_for_status(self):
            pass

    monkeypatch.setattr(collector._client, "get", lambda *a, **k: FakeResponse())

    docs = list(collector.collect())

    assert len(docs) == 1
