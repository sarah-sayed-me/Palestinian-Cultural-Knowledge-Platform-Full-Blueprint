from src.preprocessing.arabic_normalizer import count_arabic_ratio, full_clean, normalize_arabic


def test_normalize_arabic_removes_diacritics_and_normalizes_alef():
    assert normalize_arabic("أَرض إِلى") == "ارض الي"


def test_full_clean_removes_simple_wiki_markup():
    cleaned = full_clean("[[فلسطين|فلسطين التاريخية]] '''نص''' [1]", is_wikipedia=True)

    assert "فلسطين التاريخية" in cleaned
    assert "'''" not in cleaned
    assert "[1]" not in cleaned


def test_count_arabic_ratio():
    assert count_arabic_ratio("فلسطين abc") > 0.5
