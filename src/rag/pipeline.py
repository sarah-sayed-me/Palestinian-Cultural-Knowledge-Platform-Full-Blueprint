"""
RAGPipeline — the B5 integration point.

Wires a Retriever and a Generator behind one ask() call. This is the seam the
whole roadmap was designed around: everything upstream (chunking, embedding,
indexing) and downstream (generation, citation assembly) only has to agree on
the Chunk/RetrievedChunk/Answer contracts, not on each other's internals.
"""

from __future__ import annotations

from src.rag.answer import Answer, assemble_answer
from src.rag.config import RagConfig
from src.rag.generator import Generator, insufficient_context_message
from src.rag.retriever import Retriever


class RAGPipeline:
    def __init__(self, retriever: Retriever, generator: Generator, config: RagConfig):
        self.retriever = retriever
        self.generator = generator
        self.config = config

    def ask(self, question: str) -> Answer:
        retrieved = self.retriever.retrieve(
            question,
            top_k=self.config.retrieval.top_k,
            min_credibility_tier=self.config.retrieval.min_credibility_tier,
        )
        if not retrieved:
            return Answer(text=insufficient_context_message(question), citations=[])

        generated_text = self.generator.generate(question, retrieved)
        return assemble_answer(generated_text, retrieved)
