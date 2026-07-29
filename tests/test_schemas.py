from src.ingestion.schemas import (
    CredibilityTier,
    DocumentMetadata,
    Language,
    LicenseStatus,
    SourceType,
    make_doc_id,
)


def _doc(**overrides) -> DocumentMetadata:
    text = overrides.pop("text", "فلسطين ثقافة وتاريخ " * 20)
    defaults = dict(
        doc_id=make_doc_id("https://ar.wikipedia.org/wiki/فلسطين", text),
        source_id="wikipedia-ar",
        title="فلسطين",
        text=text,
        word_count=len(text.split()),
        char_count=len(text),
        language=Language.ARABIC_MSA,
        source_name="Arabic Wikipedia",
        source_type=SourceType.ENCYCLOPEDIA,
        source_url="https://ar.wikipedia.org/wiki/فلسطين",
        source_domain="ar.wikipedia.org",
        credibility=CredibilityTier.TIER_1,
    )
    defaults.update(overrides)
    return DocumentMetadata(**defaults)


def test_document_metadata_to_hf_dict_serializes_datetimes():
    doc = _doc()

    exported = doc.to_hf_dict()

    assert exported["doc_id"] == doc.doc_id
    assert exported["source_id"] == "wikipedia-ar"
    assert isinstance(exported["date_collected"], str)
    assert "text_raw" not in exported
    assert "embedding_id" not in exported


def test_license_status_defaults_to_needs_review_not_clear():
    # A collector that forgets to set license_status must fail safe (excluded
    # from publish), not silently default to clear. See LicenseStatus's docstring.
    doc = _doc()

    assert doc.license_status == LicenseStatus.NEEDS_REVIEW.value


def test_license_status_can_be_set_explicitly():
    doc = _doc(license_status=LicenseStatus.CLEAR)

    assert doc.license_status == LicenseStatus.CLEAR.value
