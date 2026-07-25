import json
from dataclasses import replace

from src.rag.chunker import chunk_corpus, chunk_document
from src.rag.config import ChunkingConfig, RagConfig


def _sentence(i: int) -> str:
    return f"sentence{i} has exactly five words."


def _config(target_tokens: int, overlap_tokens: int) -> ChunkingConfig:
    return ChunkingConfig(target_tokens=target_tokens, overlap_tokens=overlap_tokens, chunking_version="test-v1")


def test_chunk_document_returns_empty_list_for_empty_text():
    doc = {"doc_id": "d1", "text": ""}

    assert chunk_document(doc, _config(500, 75)) == []


def test_chunk_document_single_chunk_when_under_target():
    text = " ".join(_sentence(i) for i in range(3))  # 15 words
    doc = {"doc_id": "d1", "text": text}

    chunks = chunk_document(doc, _config(500, 75))

    assert len(chunks) == 1
    assert chunks[0].chunk_index == 0
    assert chunks[0].doc_id == "d1"
    assert chunks[0].token_count == 15
    assert chunks[0].chunking_version == "test-v1"


def test_chunk_document_never_splits_a_single_oversized_sentence():
    # One sentence alone (10 words) exceeds target_tokens=3 — must still be kept whole.
    doc = {"doc_id": "d1", "text": _sentence(0)}

    chunks = chunk_document(doc, _config(3, 1))

    assert len(chunks) == 1
    assert chunks[0].text == _sentence(0)


def test_chunk_document_splits_into_multiple_chunks_and_overlaps():
    # 40 sentences of 5 words each = 200 words; target_tokens=50 forces multiple chunks.
    text = " ".join(_sentence(i) for i in range(40))
    doc = {"doc_id": "d1", "text": text}

    chunks = chunk_document(doc, _config(50, 10))

    assert len(chunks) > 1
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))
    for c in chunks:
        assert c.token_count <= 50

    # Consecutive chunks overlap: the next chunk starts at or before the
    # previous chunk's end (whole trailing sentences carried forward).
    for prev, nxt in zip(chunks, chunks[1:]):
        assert nxt.start_char <= prev.end_char

    # Every chunk id is unique and deterministic for (doc_id, index, version).
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_chunk_document_carries_denormalized_document_metadata():
    doc = {
        "doc_id": "d1",
        "text": _sentence(0),
        "title": "عنوان",
        "source_url": "https://ar.wikipedia.org/wiki/Test",
        "credibility": "tier_1",
        "quality_score": 0.9,
        "seed_category": "فلسطين",
    }

    chunks = chunk_document(doc, _config(500, 75))

    assert chunks[0].title == "عنوان"
    assert chunks[0].credibility == "tier_1"
    assert chunks[0].quality_score == 0.9
    assert chunks[0].seed_category == "فلسطين"


def test_chunk_corpus_writes_jsonl_and_returns_summary(tmp_path):
    input_path = tmp_path / "docs.jsonl"
    doc = {
        "doc_id": "d1",
        "text": " ".join(_sentence(i) for i in range(3)),
        "title": "t",
        "credibility": "tier_1",
    }
    input_path.write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")
    output_path = tmp_path / "chunks.jsonl"
    config = replace(RagConfig.default(), chunking=_config(500, 75))

    summary = chunk_corpus(input_path=input_path, output_path=output_path, config=config)

    assert summary["documents_processed"] == 1
    assert summary["chunks_written"] == 1
    assert output_path.exists()

    lines = output_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["doc_id"] == "d1"
    assert record["credibility"] == "tier_1"
