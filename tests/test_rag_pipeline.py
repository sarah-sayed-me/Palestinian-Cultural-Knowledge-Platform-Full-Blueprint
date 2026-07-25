from dataclasses import replace

from src.rag.config import RagConfig
from src.rag.pipeline import RAGPipeline
from src.rag.schemas import Chunk, RetrievedChunk


class _StubRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def retrieve(self, query, *, top_k=None, min_credibility_tier=None):
        self.calls.append((query, top_k, min_credibility_tier))
        return self.results


class _StubGenerator:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def generate(self, question, context_chunks):
        self.calls.append((question, context_chunks))
        return self.text


def _retrieved(doc_id: str = "d1") -> RetrievedChunk:
    chunk = Chunk(
        chunk_id="c1",
        doc_id=doc_id,
        chunk_index=0,
        text="نص",
        token_count=1,
        start_char=0,
        end_char=2,
        chunking_version="v1",
        title="T",
        source_url="https://x",
    )
    return RetrievedChunk(chunk=chunk, score=0.9)


def test_ask_returns_insufficient_context_message_when_nothing_retrieved():
    retriever = _StubRetriever([])
    generator = _StubGenerator("should not be called")
    pipeline = RAGPipeline(retriever, generator, RagConfig.default())

    answer = pipeline.ask("ما هي الكنافة؟")

    assert answer.citations == []
    assert "لا تتوفر" in answer.text
    assert generator.calls == []  # no wasted generator call on empty context


def test_ask_generates_and_assembles_citations_when_context_found():
    retriever = _StubRetriever([_retrieved("d1")])
    generator = _StubGenerator("answer text [1]")
    pipeline = RAGPipeline(retriever, generator, RagConfig.default())

    answer = pipeline.ask("What is knafeh?")

    assert answer.text == "answer text [1]"
    assert len(answer.citations) == 1
    assert answer.citations[0].doc_id == "d1"
    assert generator.calls[0][0] == "What is knafeh?"


def test_ask_passes_configured_top_k_and_min_credibility_tier_to_retriever():
    retriever = _StubRetriever([_retrieved()])
    base = RagConfig.default()
    config = replace(base, retrieval=replace(base.retrieval, top_k=3, min_credibility_tier="tier_2"))
    pipeline = RAGPipeline(retriever, _StubGenerator("x"), config)

    pipeline.ask("q")

    assert retriever.calls[0] == ("q", 3, "tier_2")
