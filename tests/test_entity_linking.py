from src.knowlegde_graph.entity_linking import WikidataAliasLinker
from src.knowlegde_graph.schemas import KGEntity, make_entity_id

ALIAS_ENTRIES = [
    {"qid": "Q1234", "label": "يافا", "aliases": ["Jaffa", "يافا"], "instance_of": ["Q515"]},
    {"qid": "Q5678", "label": "الكوفية", "aliases": ["Keffiyeh", "كوفية"], "instance_of": []},
    {
        "qid": "Q9012",
        "label": "المتحف الفلسطيني للتراث والفنون الشعبية",
        "aliases": [],
        "instance_of": ["Q33506"],
    },
]


def _entity(name, entity_type="LOCATION"):
    return KGEntity(
        entity_id=make_entity_id(name, entity_type),
        canonical_name=name,
        type=entity_type,
        mention_count=1,
        source_doc_ids=["doc-1"],
    )


def test_exact_match_links_to_correct_qid():
    linker = WikidataAliasLinker(ALIAS_ENTRIES)

    result = linker.link("يافا")

    assert result.qid == "Q1234"
    assert result.method == "exact"
    assert result.score == 1.0


def test_exact_match_on_an_alias_not_just_the_primary_label():
    linker = WikidataAliasLinker(ALIAS_ENTRIES)

    result = linker.link("كوفية")

    assert result.qid == "Q5678"
    assert result.method == "exact"


def test_fuzzy_match_catches_close_but_not_identical_spelling():
    linker = WikidataAliasLinker(ALIAS_ENTRIES)

    # one dropped character on a long name — close enough to clear the
    # fuzzy threshold even though it isn't an exact string match
    result = linker.link("المتحف الفلسطيني للتراث والفنون الشعبي")

    assert result.qid == "Q9012"
    assert result.method == "fuzzy"
    assert result.score >= 0.90


def test_unrelated_name_does_not_link():
    linker = WikidataAliasLinker(ALIAS_ENTRIES)

    result = linker.link("مصطلح غير موجود إطلاقا")

    assert result.qid is None
    assert result.method == "none"


def test_excludes_disambiguation_pages_and_falls_through_to_a_real_entity():
    entries = [
        # A disambiguation page happens to share the exact label of a real
        # place — this is precisely the real bug found on this corpus's
        # top entities (see module docstring): without exclusion, this
        # would win the "first exact match" slot and shadow the real item.
        {"qid": "Q_DISAMBIG", "label": "غزة", "aliases": [], "instance_of": ["Q4167410"]},
        {"qid": "Q_REAL_PLACE", "label": "غزة", "aliases": [], "instance_of": ["Q515"]},
    ]
    linker = WikidataAliasLinker(entries)

    result = linker.link("غزة")

    assert result.qid == "Q_REAL_PLACE"


def test_excludes_films_and_newspapers_sharing_a_place_label():
    entries = [
        {"qid": "Q_FILM", "label": "حيفا", "aliases": [], "instance_of": ["Q11424"]},
        {"qid": "Q_NEWSPAPER", "label": "فلسطين", "aliases": [], "instance_of": ["Q11032"]},
    ]
    linker = WikidataAliasLinker(entries)

    assert linker.link("حيفا").qid is None
    assert linker.link("فلسطين").qid is None


def test_link_entities_returns_entity_and_result_pairs():
    linker = WikidataAliasLinker(ALIAS_ENTRIES)
    entities = [_entity("يافا"), _entity("مصطلح غير موجود إطلاقا", entity_type="MISC")]

    pairs = linker.link_entities(entities)

    assert len(pairs) == 2
    linked_entity, result = pairs[0]
    assert linked_entity.wikidata_qid == "Q1234"
    assert result.method == "exact"
    unmatched_entity, unmatched_result = pairs[1]
    assert unmatched_entity.wikidata_qid is None
    assert unmatched_result.method == "none"
    # original entity objects are untouched (model_copy, not mutate)
    assert entities[0].wikidata_qid is None
