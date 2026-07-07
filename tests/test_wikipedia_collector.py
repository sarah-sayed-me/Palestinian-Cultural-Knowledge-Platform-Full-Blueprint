from src.ingestion.collectors.wikipedia_collector import WikipediaCollector


class FakeWiki:
    def __init__(self, pages):
        self.pages = pages

    def page(self, title):
        return self.pages[title]


class FakePage:
    def __init__(
        self,
        title,
        text="",
        *,
        ns=0,
        members=None,
        categories=None,
        pageid=1,
    ):
        self.title = title
        self.text = text
        self.ns = ns
        self.categorymembers = members or {}
        self.categories = categories or {}
        self.pageid = pageid
        self.revision_id = 10
        self.fullurl = f"https://ar.wikipedia.org/wiki/{title}"

    def exists(self):
        return True


def test_wikipedia_collector_yields_document_from_seed_category(monkeypatch, tmp_path):
    article = FakePage(
        "فلسطين",
        text="فلسطين ثقافة تاريخ تراث مدينة قرية موسيقى أدب فن مطبخ " * 25,
        categories={"تصنيف:فلسطين": object()},
    )
    category = FakePage("تصنيف:فلسطين", ns=14, members={"فلسطين": article})

    def fake_build_client(self):
        return FakeWiki({"تصنيف:فلسطين": category})

    monkeypatch.setattr(WikipediaCollector, "_build_wiki_client", fake_build_client)
    collector = WikipediaCollector(
        source_config={
            "seed_categories": {"ar": ["فلسطين"]},
            "languages": [
                {
                    "code": "ar",
                    "name": "Arabic Wikipedia",
                    "source_id": "wikipedia-ar",
                    "credibility_tier": "tier_1",
                }
            ],
            "category_depth": 0,
            "max_articles_per_category": 10,
            "max_retries": 1,
        },
        credibility_map={"ar.wikipedia.org": {"tier": "tier_1", "score": 0.9}},
        output_dir=str(tmp_path),
        max_docs=1,
        request_delay=0,
    )

    docs = list(collector.collect())

    assert len(docs) == 1
    assert docs[0].title == "فلسطين"
    assert docs[0].source_id == "wikipedia-ar"
    assert "فلسطين" in docs[0].wikipedia_categories
