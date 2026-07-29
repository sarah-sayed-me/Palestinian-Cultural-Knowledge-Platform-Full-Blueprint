from src.ingestion.collectors.wikipedia_collector import WikipediaCollector, is_off_target_category


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
    assert docs[0].license_status == "clear"


# Real drift observed in wikipedia_ar_stats.json's category_distribution before
# this filter existed — these exact category names leaked into the corpus.
def test_is_off_target_category_catches_real_observed_drift():
    assert is_off_target_category("تصنيف:مطبخ أردني")
    assert is_off_target_category("تصنيف:مطبخ لبناني")
    assert is_off_target_category("تصنيف:مطبخ سوري")
    assert is_off_target_category("تصنيف:مطبخ إسرائيلي")
    assert is_off_target_category("تصنيف:مطبخ عراقي")
    assert is_off_target_category("تصنيف:مواقع أثرية في إسرائيل")


def test_is_off_target_category_does_not_exclude_palestinian_categories():
    assert not is_off_target_category("تصنيف:مطبخ فلسطيني")
    assert not is_off_target_category("تصنيف:عمارة فلسطينية")
    assert not is_off_target_category("تصنيف:النكبة")
    assert not is_off_target_category("تصنيف:دبكة")
    assert not is_off_target_category("تصنيف:قرى فلسطين")


def test_is_off_target_category_leaves_ambiguous_regional_categories_alone():
    # Broader regional categories (not a specific other country) are a judgment
    # call beyond what the observed drift evidence supports — left un-excluded.
    assert not is_off_target_category("تصنيف:مطبخ عربي")
    assert not is_off_target_category("تصنيف:مطبخ الشرق الأوسط")
    assert not is_off_target_category("تصنيف:عمارة إسلامية")


def test_walk_category_skips_off_target_subcategories(monkeypatch, tmp_path):
    jordanian_article = FakePage("منسف", text="طبق أردني" * 30)
    jordanian_cuisine = FakePage(
        "تصنيف:مطبخ أردني", ns=14, members={"منسف": jordanian_article}
    )
    palestinian_article = FakePage(
        "مسخن", text="فلسطين ثقافة تاريخ تراث مدينة قرية موسيقى أدب فن مطبخ " * 25,
        categories={"تصنيف:مطبخ فلسطيني": object()},
    )
    root = FakePage(
        "تصنيف:مطبخ فلسطيني",
        ns=14,
        members={"مسخن": palestinian_article, "تصنيف:مطبخ أردني": jordanian_cuisine},
    )

    def fake_build_client(self):
        return FakeWiki({"تصنيف:مطبخ فلسطيني": root})

    monkeypatch.setattr(WikipediaCollector, "_build_wiki_client", fake_build_client)
    collector = WikipediaCollector(
        source_config={
            "seed_categories": {"ar": ["مطبخ فلسطيني"]},
            "languages": [
                {"code": "ar", "name": "Arabic Wikipedia", "source_id": "wikipedia-ar", "credibility_tier": "tier_1"}
            ],
            "category_depth": 2,
            "max_articles_per_category": 10,
            "max_retries": 1,
        },
        credibility_map={"ar.wikipedia.org": {"tier": "tier_1", "score": 0.9}},
        output_dir=str(tmp_path),
        max_docs=10,
        request_delay=0,
    )

    docs = list(collector.collect())

    titles = {d.title for d in docs}
    assert "مسخن" in titles
    assert "منسف" not in titles
