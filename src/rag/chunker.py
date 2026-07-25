"""
Recursive, sentence-aware chunker for the RAG pipeline (ROADMAP.md Section 3.3).

Splits a document into overlapping passages without ever cutting a sentence in
half. Reuses split_sentences() from src/ingestion/entity_extractor.py — the same
segmentation the NER pipeline already uses — so chunk boundaries and entity-
mention boundaries agree, which matters later when Track E links an entity
mention back to the chunk it came from.

"Tokens" here means whitespace-delimited words (the same word_count convention
used throughout src/ingestion), not model subword tokens — good enough for
chunk-size control and avoids loading a second tokenizer at chunk time. If a
real subword-token budget is ever needed, that is a chunking_version bump, not
a schema change (see Chunk.chunking_version).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Iterator, List

from src.ingestion.entity_extractor import Sentence, split_sentences
from src.rag.config import ChunkingConfig, RagConfig
from src.rag.schemas import Chunk, make_chunk_id

DEFAULT_INPUT = Path("data/processed/wikipedia_ar_documents.jsonl")
DEFAULT_OUTPUT = Path("data/processed/chunks.jsonl")


def _word_count(text: str) -> int:
    return len(text.split())


def chunk_document(doc: dict[str, Any], config: ChunkingConfig) -> List[Chunk]:
    """Chunk one document's text into overlapping, sentence-aligned Chunks."""
    sentences = split_sentences(doc.get("text", "") or "")
    if not sentences:
        return []

    doc_id = doc["doc_id"]
    chunks: List[Chunk] = []
    chunk_index = 0
    i = 0
    n = len(sentences)

    while i < n:
        window, word_count, j = _fill_window(sentences, i, config.target_tokens)
        chunks.append(_build_chunk(doc, doc_id, chunk_index, window, word_count, config))
        chunk_index += 1

        if j >= n:
            break
        i = _next_start(sentences, i, j, config.overlap_tokens)

    return chunks


def _fill_window(
    sentences: List[Sentence], start: int, target_tokens: int
) -> tuple[List[Sentence], int, int]:
    """Greedily accumulate sentences from `start` until target_tokens would be
    exceeded. Always includes at least one sentence, even if it alone exceeds
    target_tokens (a chunk never splits a sentence in half)."""
    window: List[Sentence] = []
    word_count = 0
    j = start
    n = len(sentences)
    while j < n:
        sentence_words = _word_count(sentences[j].text)
        if window and word_count + sentence_words > target_tokens:
            break
        window.append(sentences[j])
        word_count += sentence_words
        j += 1
    return window, word_count, j


def _next_start(sentences: List[Sentence], prev_start: int, window_end: int, overlap_tokens: int) -> int:
    """Back up from window_end by whole trailing sentences worth ~overlap_tokens,
    so the next chunk restarts with overlapping context. Guarantees forward
    progress (next start is always > prev_start) to avoid an infinite loop."""
    overlap_words = 0
    k = window_end - 1
    while k > prev_start and overlap_words < overlap_tokens:
        overlap_words += _word_count(sentences[k].text)
        k -= 1
    return max(k + 1, prev_start + 1)


def _build_chunk(
    doc: dict[str, Any],
    doc_id: str,
    chunk_index: int,
    window: List[Sentence],
    word_count: int,
    config: ChunkingConfig,
) -> Chunk:
    text = " ".join(s.text for s in window)
    return Chunk(
        chunk_id=make_chunk_id(doc_id, chunk_index, config.chunking_version),
        doc_id=doc_id,
        chunk_index=chunk_index,
        text=text,
        token_count=word_count,
        start_char=window[0].start,
        end_char=window[-1].end,
        chunking_version=config.chunking_version,
        title=doc.get("title"),
        source_url=doc.get("source_url"),
        credibility=doc.get("credibility"),
        quality_score=doc.get("quality_score"),
        seed_category=doc.get("seed_category"),
    )


# ---------------------------------------------------------------------------
# Corpus-level driver
# ---------------------------------------------------------------------------


def _read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def chunk_corpus(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_path: Path = DEFAULT_OUTPUT,
    config: RagConfig | None = None,
) -> dict[str, Any]:
    """Chunk every document in input_path (accepted-corpus JSONL) and write all
    resulting Chunks to output_path as JSONL."""
    if not input_path.exists():
        raise FileNotFoundError(f"Input corpus not found: {input_path}")

    config = config or RagConfig.load()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    total_docs = 0
    total_chunks = 0
    total_words = 0

    with output_path.open("w", encoding="utf-8") as out_handle:
        for doc in _read_jsonl(input_path):
            total_docs += 1
            for chunk in chunk_document(doc, config.chunking):
                total_chunks += 1
                total_words += chunk.token_count
                out_handle.write(chunk.model_dump_json() + "\n")

    return {
        "input": str(input_path),
        "output": str(output_path),
        "documents_processed": total_docs,
        "chunks_written": total_chunks,
        "average_chunk_words": round(total_words / total_chunks, 2) if total_chunks else 0.0,
        "chunking_version": config.chunking.chunking_version,
    }


def iter_chunks(path: Path = DEFAULT_OUTPUT) -> Iterable[Chunk]:
    """Read a chunks.jsonl file back into Chunk objects."""
    for record in _read_jsonl(path):
        yield Chunk.model_validate(record)
