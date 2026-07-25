from src.ingestion.schemas import CredibilityTier
from src.rag.schemas import Chunk, make_chunk_id


def test_make_chunk_id_is_stable_for_same_inputs():
    a = make_chunk_id("doc-1", 0, "recursive-500-v1")
    b = make_chunk_id("doc-1", 0, "recursive-500-v1")
    assert a == b


def test_make_chunk_id_changes_with_chunking_version():
    a = make_chunk_id("doc-1", 0, "recursive-500-v1")
    b = make_chunk_id("doc-1", 0, "recursive-500-v2")
    assert a != b


def test_chunk_round_trips_denormalized_metadata():
    chunk = Chunk(
        chunk_id=make_chunk_id("doc-1", 0, "recursive-500-v1"),
        doc_id="doc-1",
        chunk_index=0,
        text="نص تجريبي",
        token_count=2,
        start_char=0,
        end_char=9,
        chunking_version="recursive-500-v1",
        title="عنوان",
        credibility=CredibilityTier.TIER_1,
        quality_score=0.9,
    )
    assert chunk.credibility == CredibilityTier.TIER_1.value
    assert chunk.doc_id == "doc-1"
