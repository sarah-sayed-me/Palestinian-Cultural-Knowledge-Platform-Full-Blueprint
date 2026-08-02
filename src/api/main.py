"""
RAG API endpoint (Track G1) — wraps the existing RAGPipeline (src/rag/pipeline.py)
behind HTTP, the same pipeline scripts/ask.py drives from the CLI. No new RAG
logic here; this is purely a transport layer.

The DB connection, embedder, and generator are built ONCE at process startup
(FastAPI lifespan) and reused across requests — building an Embedder per
request would reload the sentence-transformers model every call, and a fresh
psycopg2 connection per request doesn't pool. Known, accepted limitation for
this first pass: one shared psycopg2 connection is not safe under concurrent
requests (psycopg2 connections aren't thread-safe for simultaneous queries).
Fine for a research/demo API serving one user at a time; a connection pool
(e.g. psycopg2.pool.ThreadedConnectionPool) is the natural upgrade if/when
this needs real concurrent traffic — not built ahead of that need, per this
project's own "validate cheap first" convention.

Run:
    docker compose up -d
    uv run uvicorn src.api.main:app --reload --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.rag.config import RagConfig
from src.rag.db import get_connection
from src.rag.embedder import Embedder
from src.rag.generator import OllamaGenerator
from src.rag.pipeline import RAGPipeline
from src.rag.retriever import Retriever

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    config = RagConfig.load()
    embedder = Embedder(config.embedding)
    conn = get_connection()
    retriever = Retriever(conn, embedder, config)
    generator = OllamaGenerator(config.generation)
    _state["config"] = config
    _state["conn"] = conn
    _state["pipeline"] = RAGPipeline(retriever, generator, config)
    try:
        yield
    finally:
        conn.close()
        _state.clear()


app = FastAPI(
    title="Palestinian Cultural Knowledge Platform — RAG API",
    description="Ask a question, get a grounded, cited answer over the ingested corpus.",
    version="0.1.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str
    top_k: Optional[int] = None
    min_credibility_tier: Optional[str] = None


class CitationResponse(BaseModel):
    index: int
    doc_id: str
    title: Optional[str] = None
    source_url: Optional[str] = None


class AskResponse(BaseModel):
    text: str
    citations: List[CitationResponse]


@app.get("/health")
def health() -> dict:
    """Reports whether the pipeline is wired up — does not itself call Ollama
    (a slow, model-dependent round trip) on every health check."""
    ready = "pipeline" in _state
    return {"status": "ok" if ready else "starting", "pipeline_ready": ready}


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest) -> AskResponse:
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    pipeline: RAGPipeline = _state["pipeline"]
    config: RagConfig = _state["config"]

    # top_k / min_credibility_tier overrides apply only to retrieval, so we
    # temporarily override the config the retriever reads rather than
    # threading extra parameters through RAGPipeline.ask() itself.
    original_retrieval = pipeline.config.retrieval
    try:
        if request.top_k is not None or request.min_credibility_tier is not None:
            from dataclasses import replace

            pipeline.config = replace(
                config,
                retrieval=replace(
                    original_retrieval,
                    top_k=request.top_k if request.top_k is not None else original_retrieval.top_k,
                    min_credibility_tier=(
                        request.min_credibility_tier
                        if request.min_credibility_tier is not None
                        else original_retrieval.min_credibility_tier
                    ),
                ),
            )
        try:
            answer = pipeline.ask(request.question)
        except RuntimeError as exc:
            # RAGPipeline surfaces a clear RuntimeError (e.g. Ollama unreachable
            # or the model isn't pulled) — pass that message through as a 503
            # rather than a raw 500 traceback.
            raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        pipeline.config = config

    return AskResponse(
        text=answer.text,
        citations=[
            CitationResponse(index=c.index, doc_id=c.doc_id, title=c.title, source_url=c.source_url)
            for c in answer.citations
        ],
    )
