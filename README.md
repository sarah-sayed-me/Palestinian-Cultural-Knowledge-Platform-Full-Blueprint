# Palestinian Cultural Knowledge Platform

## Overview

The Palestinian Cultural Knowledge Platform builds a reproducible, quality-scored
corpus of Palestinian cultural text and turns it into structured, queryable knowledge.
The pipeline collects documents from credible sources, cleans and scores them against
an explicit quality contract, removes near-duplicates, and publishes a versioned
dataset. Downstream stages enrich the corpus with named entities, a knowledge graph,
and retrieval-augmented question answering.

The current implementation focuses on **Arabic Wikipedia** as a clean, licensable first
source, with the ingestion, quality, deduplication, and named-entity-recognition stages
working end to end. Everything beyond NER is scaffolded but not yet implemented.

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

Config-driven collectors emit a unified `DocumentMetadata` record that every downstream
stage consumes.

```
                 configs/sources.yaml          configs/quality_thresholds.yaml
                        │                                │
   ┌────────────────────▼────────────────────┐          │
   │   Collectors (BaseCollector interface)   │          │
   │   Wikipedia AR  ✓                         │          │
   │   GDELT · WAFA · archives · papers  ✗     │          │
   └────────────────────┬────────────────────┘          │
                        ▼                                │
        Normalization  ──►  Quality scoring  ──►  Deduplication (MinHash / LSH)
        (Arabic clean)      (composite 0–1)  ◄─────────────┘
                        │
                        ▼
             DocumentMetadata (unified Pydantic schema)
                        │
             ┌──────────┴──────────────┐
             ▼                         ▼
     JSONL corpus              HF export (Parquet)  ──►  Hugging Face dataset
             │
             ▼
     NER: CAMeL + heritage dictionary  ✓
             │
             ▼
     Entity linking + relation extraction   ✗
             │
             ▼
     Knowledge graph   ✗   ──►   Topic modeling / classification   ✗
             │
             ▼
     Embeddings + RAG (retrieval-augmented QA)   ✗

     ✓ implemented   ✗ planned
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

## Milestones

| # | Phase | Status |
|---|-------|--------|
| 1 | Ingestion — Arabic Wikipedia collection with provenance | **Done (iterating)** |
| 2 | Text normalization, quality scoring, deduplication | **Done** |
| 3 | Corpus packaging & publishing (Parquet / Hugging Face) | **Done** |
| 4 | Multi-source expansion (EN Wikipedia, news, archives, papers) | Planned |
| 5 | Named Entity Recognition (CAMeL + heritage dictionary) | **Implemented — needs evaluation** |
| 6 | Entity linking, canonicalization & relation extraction | Planned |
| 7 | Knowledge graph construction | Planned |
| 8 | Topic modeling & content classification | Planned |
| 9 | Embeddings + retrieval-augmented QA (RAG) | Planned |

## Repository Structure

```
configs/            Source, quality, and heritage-entity configuration (YAML)
src/
  ingestion/        Collectors, pipeline orchestration, schema, quality, dedup, NER
    collectors/     Per-source collectors (Wikipedia; base class for the rest)
  preprocessing/    Arabic text normalization
  utils/            Collection logging
scripts/            Runnable entrypoints (NER, HF export/publish, seed audit)
tests/              Unit tests for schema, quality, dedup, normalizer, NER, export
docs/               Publishing guide and supporting documentation
reports/            Generated audit reports (seed categories)
data/               Local corpus artifacts (git-ignored)
main.py             Pipeline entrypoint
```

## Current Status

The Arabic Wikipedia path is functional end to end. A representative run collected ~500
articles, accepted 484 after quality and duplicate filtering, and NER has been run across
the accepted corpus. The unified schema, quality/dedup logic, seed-category audit tooling,
and Hugging Face export are all working and covered by tests.

Typical run:

```powershell
python main.py --max-docs 100        # collect → clean → score → dedup → JSONL + stats
python scripts/export_to_hf.py       # package accepted docs as Parquet
python scripts/run_ner.py            # enrich the corpus with entities
```

Generated datasets are intentionally git-ignored; see `docs/huggingface_publishing_guide.md`.
All stages after NER (entity linking, knowledge graph, topic modeling, RAG) are defined in
the schema and roadmap but not yet implemented.

## Future Work

- **Incremental collection**: persist the deduplication index and dedup against the existing
  corpus so the corpus can grow across runs instead of being overwritten.
- **Multi-source ingestion**: implement collectors for the sources currently defined-but-disabled
  in `configs/sources.yaml`, including a per-source licensing/rights check before publishing.
- **NER evaluation**: build a small gold-annotated set and report precision/recall before
  downstream stages depend on entity output.
- **Entity linking + relation extraction** to bridge NER into a knowledge graph.
- **Knowledge graph, topic modeling, and RAG** per phases 7–9.
- **Collection-time relevance gating** to reduce topical drift into neighboring-country content.

## Contributing

1. Work against a small corpus first; validate pipeline changes on real output before scaling.
2. Keep `DocumentMetadata` the single source of truth — new sources conform to the schema.
3. Put tunable rules in `configs/`, not in code, and keep tests green (`uv run pytest`).
4. Preserve provenance: every document must carry its source and (where applicable) seed category.
5. Respect source licensing; do not add non-redistributable content to published datasets.
