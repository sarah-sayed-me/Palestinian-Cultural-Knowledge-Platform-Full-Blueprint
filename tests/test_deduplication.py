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
