"""
Answer + citation assembly (ROADMAP.md Track B, C2).

Turns a generator's raw text plus the RetrievedChunks it was grounded in into a
structured Answer: numbered citations matching the [1], [2] markers the
generator's prompt asks it to use, deduplicated by source document.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel

from src.rag.schemas import RetrievedChunk


class Citation(BaseModel):
    index: int  # matches the [N] marker the generator was instructed to use
    doc_id: str
    title: Optional[str] = None
    source_url: Optional[str] = None


class Answer(BaseModel):
    text: str
    citations: List[Citation]


def assemble_answer(generated_text: str, context_chunks: List[RetrievedChunk]) -> Answer:
    """Build an Answer from generated text and the chunks it was grounded in.

    Citations are numbered in retrieval order (matching the numbering the
    generator's prompt was given, see generator.py::_format_context) and
    deduplicated by doc_id — several chunks from the same article collapse to
    one citation.
    """
    citations: List[Citation] = []
    seen_doc_ids: set[str] = set()
    for i, retrieved in enumerate(context_chunks, start=1):
        chunk = retrieved.chunk
        if chunk.doc_id in seen_doc_ids:
            continue
        seen_doc_ids.add(chunk.doc_id)
        citations.append(
            Citation(index=i, doc_id=chunk.doc_id, title=chunk.title, source_url=chunk.source_url)
        )
    return Answer(text=generated_text, citations=citations)
