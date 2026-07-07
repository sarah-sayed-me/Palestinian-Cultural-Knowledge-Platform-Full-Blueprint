from src.ingestion.schemas import (
    CredibilityTier,
    DocumentMetadata,
    Language,
    SourceType,
    make_doc_id,
)


def test_document_metadata_to_hf_dict_serializes_datetimes():
    text = "فلسطين ثقافة وتاريخ " * 20
    doc = DocumentMetadata(
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

    exported = doc.to_hf_dict()

    assert exported["doc_id"] == doc.doc_id
    assert exported["source_id"] == "wikipedia-ar"
    assert isinstance(exported["date_collected"], str)
    assert "text_raw" not in exported
    assert "embedding_id" not in exported
