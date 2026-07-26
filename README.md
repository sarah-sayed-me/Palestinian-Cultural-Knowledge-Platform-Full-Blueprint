# Palestinian Cultural Knowledge Platform

## Overview

The Palestinian Cultural Knowledge Platform builds a reproducible, quality-scored
corpus of Palestinian cultural text and turns it into structured, queryable knowledge.
The pipeline collects documents from credible sources, cleans and scores them against
an explicit quality contract, removes near-duplicates, and publishes a versioned
dataset. Downstream stages enrich the corpus with named entities, a knowledge graph,
and retrieval-augmented question answering.

The current implementation focuses on **Arabic Wikipedia** as a clean, licensable first
source. Ingestion, quality, deduplication, named-entity-recognition, and a working
retrieval-augmented QA (RAG) system are all functional end to end over the real corpus. The
knowledge graph and the analysis phases (topic modeling, cultural classification, bias
measurement, temporal analysis) are fully designed — see `ROADMAP.md` — and next in the
build order, which reached RAG first without narrowing the platform's scope.

## Goals

- Assemble a high-quality, well-documented Palestinian cultural corpus with transparent
  provenance and quality scoring for every document.
- Prioritize **culture and heritage** — food, dress, crafts, folklore, place, practice —
  alongside history, rather than framing the record solely through conflict.
- Enforce a single data contract so that any future source (news, archives, papers, oral
  testimony) flows through the same quality and deduplication guarantees.
- Extract entities and relations to construct a knowledge graph, and expose the corpus
  through retrieval-augmented QA.
- Keep the corpus reproducible and publishable (Hugging Face Datasets, semantic versions).

## Architecture

Config-driven collectors emit a unified `DocumentMetadata` record. Everything downstream —
RAG, the knowledge graph, and every analysis phase — reads from that same record and its
sibling `Chunk` / `KGEntity` / `KGRelation` records rather than re-deriving its own shape;
see `ROADMAP.md` §2 for the full data-contract design.

```
                 configs/sources.yaml          configs/quality_thresholds.yaml
                        │                                │
   ┌────────────────────▼────────────────────┐          │
   │   Collectors (BaseCollector interface)   │          │
   │   Wikipedia AR  ✓   GDELT · WAFA · archives · papers  ✗
   └────────────────────┬────────────────────┘          │
                        ▼                                │
        Normalization  ──►  Quality scoring  ──►  Deduplication (MinHash / LSH)
        (Arabic clean)      (composite 0–1)  ◄─────────────┘
                        │
                        ▼
             DocumentMetadata (unified Pydantic schema — stable, extended via sibling models only)
                        │
        ┌───────────────┼────────────────────────────┬─────────────────────┐
        ▼               ▼                            ▼                     ▼
  NER: CAMeL +   Chunk + embed (Qwen3-Embedding)  Topic modeling /   Temporal analysis
  heritage dict ✓  ──► pgvector index ✓         content classification  (decade buckets) ✗
        │               │                            (BERTopic/AraBERT) ✗
        ▼               ▼
  Entity linking    Retriever ──► Generator (Ollama·Qwen3)
  + relations ✗       ──► grounded, cited RAG answers ✓
        │
        ▼
  Knowledge graph (Neo4j) ✗ ──► Bias measurement (WEAT / framing / LLM probe) ✗
        │
        ▼
  Dashboard (Streamlit → HF Spaces): Bias Meter · Topic Map · Timeline · KG Explorer ✗

  ✓ implemented   ✗ planned — see ROADMAP.md for build order and status
```

## Development Philosophy

This project is built **iteratively, not big-bang**. Rather than collecting tens of
thousands of documents up front, we run the loop:

> **small corpus → full pipeline → validate against real output → expand → repeat.**

The whole pipeline is exercised against a small, real corpus first, so schema decisions,
quality thresholds, deduplication behavior, and source-traversal bias are validated on
actual data while they are cheap to change. This has already surfaced concrete issues at
small scale — dead/renamed seed categories and topical drift into neighboring-country
content — that would have been expensive to discover after a large collection run.

The trade-off is explicit: **the first source (Wikipedia) is deliberately the easy case** —
clean, structured, and licensable. Harder inputs (web scraping, OCR, dialectal Arabic,
oral testimony, and non-redistributable licensing) are introduced incrementally so each
can stress the shared pipeline before we commit to scale.

This same philosophy shapes the build order, not just the corpus: RAG was sequenced before
the knowledge graph and the analysis phases because it was the fastest way to validate the
core data contracts (chunking, embeddings, retrieval) against real output — not because the
KG, topic modeling, cultural classification, bias measurement, or temporal analysis matter
less. Every one of those phases was fully designed up front precisely so that reaching RAG
first never forced a redesign later, and none did. See `ROADMAP.md` for the full reasoning.

## Milestones

Full priorities, dependencies, technical-decision rationale, and a two-developer execution
plan live in **`ROADMAP.md`**. Summary:

| # | Phase | Status |
|---|-------|--------|
| 1 | Ingestion — Arabic Wikipedia collection with provenance | **Done (iterating)** |
| 2 | Text normalization, quality scoring, deduplication | **Done** |
| 3 | Corpus packaging & publishing (Parquet / Hugging Face) | **Done** |
| 4 | Named Entity Recognition (CAMeL + heritage dictionary) | **Done — F1 0.47 (see ROADMAP.md)** |
| 5 | Retrieval-augmented QA (chunk → embed → pgvector → retrieve → generate) | **Done — verified end to end** |
| 6 | Evaluation layer (NER, embeddings, retrieval, RAG) | **Done — Recall@5 0.93, see ROADMAP.md** |
| 7 | Multi-source expansion (EN Wikipedia, news, archives, papers) + licensing gate | Planned |
| 8 | Entity linking, relation extraction & knowledge graph (Neo4j) | Planned |
| 9 | Topic modeling, cultural classification, bias measurement, temporal analysis | Planned |
| 10 | Dashboard (Streamlit → Hugging Face Spaces) | Planned |

## Repository Structure

```
configs/            Source, quality, and heritage-entity configuration (YAML)
src/
  ingestion/        Collectors, pipeline orchestration, schema, quality, dedup, NER
    collectors/     Per-source collectors (Wikipedia; base class for the rest)
  preprocessing/    Arabic text normalization
  utils/            Collection logging
  rag/              Chunking, embedding, pgvector index, retriever, generator (done)
  knowlegde_graph/  Entity canonicalization, linking, relations, KG store (planned — empty)
  nlp/              Topic modeling, cultural classification, bias, temporal analysis (planned — empty)
  api/              RAG API endpoint (planned — empty)
  frontend/         Dashboard (planned — empty)
eval/               Evaluation harness (done for NER/retrieval/RAG — see eval/gold/, eval_reports/, ROADMAP.md); KG eval planned
scripts/            Runnable entrypoints (NER, HF export/publish, seed audit)
tests/              Unit tests for schema, quality, dedup, normalizer, NER, export
docs/               Publishing guide and supporting documentation
reports/            Generated audit reports (seed categories)
data/               Local corpus artifacts (git-ignored)
main.py             Pipeline entrypoint
ROADMAP.md          Full milestone breakdown, technical decisions, execution plan
```

## Current Status

The Arabic Wikipedia path is functional end to end, **including a working RAG system**. A
representative run collected ~500 articles, accepted 484 after quality and duplicate
filtering, and NER has been run across the accepted corpus. The unified schema, quality/dedup
logic, seed-category audit tooling, and Hugging Face export are all working and covered by
tests. RAG (chunking, embedding with Qwen3-Embedding, a pgvector index, retriever, and an
Ollama-based local-first generator behind a stable interface) has been built and verified
against the real corpus — 484 documents chunked into 1282 passages, fully embedded and
indexed, with `scripts/ask.py` returning grounded, cited answers. See `ROADMAP.md` for the
technical-decision reasoning and what's next.

Typical run:

```powershell
python main.py --max-docs 100        # collect → clean → score → dedup → JSONL + stats
python scripts/export_to_hf.py       # package accepted docs as Parquet
python scripts/run_ner.py            # enrich the corpus with entities

docker compose up -d                 # start the pgvector store
python scripts/chunk_corpus.py       # split accepted docs into retrieval-sized chunks
python scripts/build_index.py        # embed chunks and load them into pgvector
python scripts/ask.py "ما هي الكنافة النابلسية؟"   # ask a question, get a grounded, cited answer
```

Generated datasets are intentionally git-ignored; see `docs/huggingface_publishing_guide.md`.
The RAG generator requires [Ollama](https://ollama.com) running locally with the model named
in `configs/rag.yaml`'s `generation.model` pulled (`ollama pull <model>`).

## Future Work

See `ROADMAP.md` for the complete, prioritized breakdown (Tracks A–H: evaluation,
multi-source expansion, knowledge graph, topic modeling, cultural classification, bias
measurement, temporal analysis, dashboard, and infrastructure hardening) with priorities,
dependencies, and effort estimates for each item.

## Contributing

1. Work against a small corpus first; validate pipeline changes on real output before scaling.
2. Keep `DocumentMetadata` the single source of truth — new sources conform to the schema.
3. Put tunable rules in `configs/`, not in code, and keep tests green (`uv run pytest`).
4. Preserve provenance: every document must carry its source and (where applicable) seed category.
5. Respect source licensing; do not add non-redistributable content to published datasets.
