from src.ingestion.entity_extractor import (
    EntityExtractor,
    HeritageEntry,
    HeritageMatcher,
    split_sentences,
    tokenize_with_offsets,
)


def _matcher() -> HeritageMatcher:
    return HeritageMatcher(
        [
            HeritageEntry(canonical="صابون نابلسي", category="craft"),
            HeritageEntry(canonical="ماء ورد", category="plant"),
            HeritageEntry(canonical="دبس تمر", category="food"),
            HeritageEntry(canonical="بيت لحم", category="place"),
            HeritageEntry(canonical="كوفية", category="clothing"),
        ]
    )


def test_split_sentences_tracks_offsets():
    text = "هذه جملة أولى. وهذه جملة ثانية؟"
    sentences = split_sentences(text)

    assert len(sentences) == 2
    for sentence in sentences:
        assert text[sentence.start : sentence.end] == sentence.text


def test_heritage_matcher_handles_definite_article_on_both_words():
    # "الصابون النابلسي" — article attaches to both words (adjective phrase)
    text = "اشترى الصابون النابلسي من نابلس."
    sentence = split_sentences(text)[0]
    tokens = tokenize_with_offsets(sentence)

    mentions = list(_matcher().find_mentions(tokens, sentence.index))

    assert any(m.canonical == "صابون نابلسي" for m in mentions)
    hit = next(m for m in mentions if m.canonical == "صابون نابلسي")
    assert text[hit.start_char : hit.end_char] == "الصابون النابلسي"


def test_heritage_matcher_handles_idafa_construct_article_on_last_word():
    # "ماء الورد" — construct phrase, article attaches to the second word only
    text = "استخدمت ماء الورد في الطبخ."
    sentence = split_sentences(text)[0]
    tokens = tokenize_with_offsets(sentence)

    mentions = list(_matcher().find_mentions(tokens, sentence.index))

    assert any(m.canonical == "ماء ورد" for m in mentions)


def test_heritage_matcher_matches_exact_proper_noun_place():
    text = "ولد في بيت لحم عام ١٩٩٠."
    sentence = split_sentences(text)[0]
    tokens = tokenize_with_offsets(sentence)

    mentions = list(_matcher().find_mentions(tokens, sentence.index))

    assert any(m.canonical == "بيت لحم" and m.entity_type == "HERITAGE_PLACE" for m in mentions)


def test_extractor_aggregates_repeated_mentions_without_camel():
    text = "الكوفية رمز فلسطيني. يرتدي الفلسطينيون الكوفية في المناسبات."
    extractor = EntityExtractor(heritage_matcher=_matcher(), use_camel=False, camel_backend=None)

    entities = extractor.extract(text)
    kufiya = next(e for e in entities if e["canonical"] == "كوفية")

    assert kufiya["mention_count"] == 2
    assert len(kufiya["positions"]) == 2
    assert kufiya["confidence"] == 1.0
    assert kufiya["source"] == "heritage_dictionary"


def test_extract_many_batches_across_documents_without_camel():
    extractor = EntityExtractor(heritage_matcher=_matcher(), use_camel=False, camel_backend=None)
    texts = ["ولد في بيت لحم.", "اشترت ماء الورد ودبس التمر."]

    results = extractor.extract_many(texts)

    assert len(results) == 2
    assert any(e["canonical"] == "بيت لحم" for e in results[0])
    canonicals = {e["canonical"] for e in results[1]}
    assert "ماء ورد" in canonicals
    assert "دبس تمر" in canonicals