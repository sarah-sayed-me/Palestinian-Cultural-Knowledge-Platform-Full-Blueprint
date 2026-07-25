from src.ingestion.quality_checker import QualityConfig, _richness_score, check_document
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


def test_richness_score_is_monotonically_non_decreasing_across_the_full_range():
    config = QualityConfig.default()
    scores = [_richness_score(wc, config) for wc in range(0, 2001)]

    for previous, current in zip(scores, scores[1:]):
        assert current >= previous


def test_richness_score_at_the_full_breakpoint_does_not_dip_below_the_previous_band():
    # Regression test: a 200-word document used to score 0.20 (word_count / 1000)
    # while a 199-word document scored 0.70 — a 200-word doc must never score
    # lower than a 199-word doc.
    config = QualityConfig.default()

    assert _richness_score(199, config) == 0.70
    assert _richness_score(200, config) == 0.70
    assert _richness_score(200, config) >= _richness_score(199, config)


def test_richness_score_ramps_to_one_at_saturation():
    config = QualityConfig.default()

    assert _richness_score(config.richness_saturation, config) == 1.0
    assert _richness_score(config.richness_saturation * 2, config) == 1.0


def test_quality_config_load_reads_configs_quality_thresholds_yaml():
    config = QualityConfig.load()

    # These values are asserted against configs/quality_thresholds.yaml directly,
    # so this test fails if the YAML and the loader ever drift apart again.
    assert config.min_word_count == 50
    assert config.richness_full == 200
    assert config.richness_saturation == 1000
    assert config.weight_richness == 0.40
