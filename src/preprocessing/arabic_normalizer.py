"""
Arabic text normalisation utilities.

These functions are intentionally lightweight so collectors can clean text at
collection time. Deeper linguistic processing belongs to later NLP phases.
"""

from __future__ import annotations

import re
import unicodedata

try:
    import ftfy

    _HAS_FTFY = True
except ImportError:
    _HAS_FTFY = False

_DIACRITICS_RE = re.compile(r"[\u0617-\u061A\u064B-\u065F]")
_TATWEEL_RE = re.compile(r"\u0640+")
_ALEF_RE = re.compile(r"[\u0623\u0625\u0622\u0671]")
_ARABIC_INDIC = str.maketrans("٠١٢٣٤٥٦٧٨٩", "0123456789")
_SPACE_RE = re.compile(r"[ \t]+")
_NEWLINE_RE = re.compile(r"\n{3,}")
_HTML_TAG_RE = re.compile(r"<[^>]+>")

_BOILERPLATE_PATTERNS = [
    re.compile(r"اشترك في النشرة الإخبارية.*", re.DOTALL),
    re.compile(r"Subscribe to.*newsletter.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"©\s*\d{4}.*", re.IGNORECASE),
    re.compile(r"All rights reserved.*", re.IGNORECASE),
    re.compile(r"جميع الحقوق محفوظة.*"),
    re.compile(r"Cookie\s+Policy.*", re.IGNORECASE),
    re.compile(r"Privacy\s+Policy.*", re.IGNORECASE),
    re.compile(r"تابعونا على.*"),
    re.compile(r"Follow us on.*", re.IGNORECASE),
    re.compile(r"شارك هذا الخبر.*"),
    re.compile(r"Share this article.*", re.IGNORECASE),
    re.compile(r"اقرأ أيضا.*", re.IGNORECASE),
    re.compile(r"Read more.*", re.IGNORECASE),
    re.compile(r"تم النشر في.*"),
    re.compile(r"Published on.*", re.IGNORECASE)
]

_WIKI_SECTION_RE = re.compile(
    r"==\s*(انظر أيضًا|مراجع|وصلات خارجية|المصادر|روابط خارجية|See also|References|External links)\s*==.*",
    re.DOTALL | re.IGNORECASE,
)
_WIKI_FILE_RE = re.compile(r"\[\[(?:ملف|File|Image):[^\]]+\]\]", re.IGNORECASE)
_WIKI_TEMPLATE_RE = re.compile(r"\{\{[^}]+\}\}")
_WIKI_LINK_RE = re.compile(r"\[\[(?:[^\|\]]+\|)?([^\]]+)\]\]")
_WIKI_STYLE_RE = re.compile(r"'{2,3}")
_WIKI_LIST_RE = re.compile(r"^[#\*:;]+", re.MULTILINE)
_WIKI_CITATION_RE = re.compile(r"\[\d+\]")


def fix_encoding(text: str) -> str:
    """Fix broken Unicode with ftfy when available, then normalise to NFC."""
    if _HAS_FTFY:
        text = ftfy.fix_text(text)
    return unicodedata.normalize("NFC", text)


def remove_html(text: str) -> str:
    """Strip HTML/XML tags."""
    return _HTML_TAG_RE.sub(" ", text)


def remove_boilerplate(text: str) -> str:
    """Remove common web boilerplate fragments."""
    for pattern in _BOILERPLATE_PATTERNS:
        text = pattern.sub("", text)
    return text


def clean_wikipedia_markup(text: str) -> str:
    """Remove simple MediaWiki markup artifacts while preserving link labels."""
    text = _WIKI_SECTION_RE.sub(" ", text)
    text = _WIKI_FILE_RE.sub(" ", text)
    text = _WIKI_TEMPLATE_RE.sub(" ", text)
    text = _WIKI_LINK_RE.sub(r"\1", text)
    text = _WIKI_STYLE_RE.sub("", text)
    text = _WIKI_LIST_RE.sub("", text)
    text = _WIKI_CITATION_RE.sub("", text)
    return text


def normalize_arabic(text: str, *, remove_diacritics: bool = True) -> str:
    """Apply standard Arabic orthographic normalisation."""
    text = _ALEF_RE.sub("ا", text)
    text = text.replace("ى", "ي")
    if remove_diacritics:
        text = _DIACRITICS_RE.sub("", text)
    text = _TATWEEL_RE.sub("", text)
    return text.translate(_ARABIC_INDIC)


def collapse_whitespace(text: str) -> str:
    """Normalise horizontal whitespace and reduce excessive blank lines."""
    text = _SPACE_RE.sub(" ", text)
    text = _NEWLINE_RE.sub("\n\n", text)
    return text.strip()


def full_clean(
    text: str,
    *,
    is_wikipedia: bool = False,
    remove_diacritics: bool = True,
) -> str:
    """Run the complete lightweight cleaning pipeline."""
    text = fix_encoding(text)
    text = remove_html(text)
    text = remove_boilerplate(text)
    if is_wikipedia:
        text = clean_wikipedia_markup(text)
    text = normalize_arabic(text, remove_diacritics=remove_diacritics)
    return collapse_whitespace(text)


def count_arabic_ratio(text: str) -> float:
    """Return the fraction of characters that are Arabic script."""
    if not text:
        return 0.0
    arabic_chars = sum(
        1 for char in text
        if "\u0600" <= char <= "\u06FF" or "\u0750" <= char <= "\u077F"
    )
    return arabic_chars / len(text)
