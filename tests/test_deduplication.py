from src.ingestion.deduplication import DuplicationIndex


def test_duplication_index_detects_duplicate_text():
    index = DuplicationIndex(threshold=0.8, num_perm=64)
    text = "فلسطين ثقافة وتاريخ وتراث" * 20

    first = index.check_and_register("doc-1", text)
    second = index.check_and_register("doc-2", text)

    assert not first.is_duplicate
    assert second.is_duplicate
    assert second.canonical_id == "doc-1"


def test_duplication_index_accepts_distinct_text():
    index = DuplicationIndex(threshold=0.8, num_perm=64)

    first = index.check_and_register("doc-1", "فلسطين ثقافة وتاريخ" * 20)
    second = index.check_and_register("doc-2", "المطبخ الفلسطيني والتطريز" * 20)

    assert not first.is_duplicate
    assert not second.is_duplicate


def test_duplication_index_same_doc_id_with_dissimilar_text_is_a_duplicate_not_a_crash():
    # Regression test: a page re-fetched live (e.g. GDELT) can produce the
    # same doc_id (hashed from url + first 200 chars) while the rest of the
    # text drifts enough (ads, related-articles widgets) that MinHash/LSH
    # similarity falls below threshold. Re-registering the same doc_id must
    # be treated as a duplicate, not raise from datasketch's LSH.insert().
    index = DuplicationIndex(threshold=0.8, num_perm=64)

    first = index.check_and_register("doc-1", "فلسطين ثقافة وتاريخ وتراث" * 20)
    second = index.check_and_register("doc-1", "نص مختلف تماما لا علاقة له بالنص الأول" * 20)

    assert not first.is_duplicate
    assert second.is_duplicate
    assert second.canonical_id == "doc-1"
