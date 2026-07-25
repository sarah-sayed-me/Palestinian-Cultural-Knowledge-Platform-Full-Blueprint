from src.rag.answer import assemble_answer
from src.rag.schemas import Chunk, RetrievedChunk


def _retrieved(doc_id: str, chunk_index: int, title: str, url: str, score: float) -> RetrievedChunk:
    chunk = Chunk(
        chunk_id=f"{doc_id}-{chunk_index}",
        doc_id=doc_id,
        chunk_index=chunk_index,
        text="نص",
        token_count=1,
        start_char=0,
        end_char=2,
        chunking_version="v1",
        title=title,
        source_url=url,
    )
    return RetrievedChunk(chunk=chunk, score=score)


def test_assemble_answer_numbers_citations_in_retrieval_order():
    chunks = [
        _retrieved("doc-1", 0, "Title A", "https://a", 0.9),
        _retrieved("doc-2", 0, "Title B", "https://b", 0.8),
    ]

    answer = assemble_answer("some grounded answer [1][2]", chunks)

    assert answer.text == "some grounded answer [1][2]"
    assert [c.index for c in answer.citations] == [1, 2]
    assert answer.citations[0].doc_id == "doc-1"
    assert answer.citations[1].doc_id == "doc-2"


def test_assemble_answer_dedupes_citations_from_the_same_document():
    chunks = [
        _retrieved("doc-1", 0, "Title A", "https://a", 0.9),
        _retrieved("doc-1", 1, "Title A", "https://a", 0.85),
        _retrieved("doc-2", 0, "Title B", "https://b", 0.8),
    ]

    answer = assemble_answer("answer", chunks)

    assert [c.doc_id for c in answer.citations] == ["doc-1", "doc-2"]


def test_assemble_answer_with_no_context_has_no_citations():
    answer = assemble_answer("no sources", [])

    assert answer.citations == []
