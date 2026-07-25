from src.rag.generator import _format_context, _looks_arabic, insufficient_context_message
from src.rag.schemas import Chunk, RetrievedChunk


def _retrieved(text: str, title: str = "T", url: str = "https://x") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1",
        doc_id="d1",
        chunk_index=0,
        text=text,
        token_count=1,
        start_char=0,
        end_char=len(text),
        chunking_version="v1",
        title=title,
        source_url=url,
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


def test_looks_arabic_detects_arabic_script():
    assert _looks_arabic("ما هي الكنافة؟")
    assert not _looks_arabic("What is knafeh?")


def test_format_context_numbers_sources_with_title_and_url():
    block = _format_context([_retrieved("some text", title="My Title", url="https://example.com")])

    assert "[1] Title: My Title" in block
    assert "Source: https://example.com" in block
    assert "some text" in block


def test_format_context_numbers_multiple_sources_sequentially():
    block = _format_context([_retrieved("first"), _retrieved("second")])

    assert "[1] Title:" in block
    assert "[2] Title:" in block


def test_insufficient_context_message_matches_question_language():
    assert "لا تتوفر" in insufficient_context_message("ما هي الكنافة؟")
    assert "don't have enough information" in insufficient_context_message("What is knafeh?")
