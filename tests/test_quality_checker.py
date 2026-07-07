from src.ingestion.quality_checker import check_document
from src.ingestion.schemas import (
    CredibilityTier,
    DocumentMetadata,
    Language,
    QualityDecision,
    SourceType,
    make_doc_id,
)


def _doc(text: str, language=Language.ARABIC_MSA):
    return DocumentMetadata(
        doc_id=make_doc_id("https://ar.wikipedia.org/wiki/Test", text),
        source_id="wikipedia-ar",
        title="اختبار",
        text=text,
        word_count=len(text.split()),
        char_count=len(text),
        language=language,
        source_name="Arabic Wikipedia",
        source_type=SourceType.ENCYCLOPEDIA,
        source_url="https://ar.wikipedia.org/wiki/Test",
        source_domain="ar.wikipedia.org",
        credibility=CredibilityTier.TIER_1,
    )


def test_quality_accepts_valid_arabic_document():
    text = "فلسطين ثقافة تاريخ تراث مدينة قرية موسيقى أدب فن مطبخ " * 25

    report = check_document(_doc(text))

    assert report.is_valid
    assert report.decision == QualityDecision.ACCEPT


def test_quality_rejects_short_document():
    report = check_document(_doc("قصير جدا"))

    assert not report.is_valid
    assert report.decision in {QualityDecision.REJECT, QualityDecision.HARD_REJECT}


def test_quality_rejects_low_arabic_ratio_for_arabic_doc():
    text = "english text only " * 60

    report = check_document(_doc(text))

    assert not report.is_valid
    assert "Arabic script ratio" in (report.rejection_reason or "")
