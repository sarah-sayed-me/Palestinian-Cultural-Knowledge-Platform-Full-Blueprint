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
   │   Wikipedia AR/EN · Semantic Scholar · GDELT · WAFA  ✓
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
| 7 | Multi-source expansion (EN Wikipedia, Semantic Scholar w/ full-text OA, GDELT, WAFA) + persistent dedup | **Done — see ROADMAP.md; Nakba Archive/Palestine Remembered held on ethical/access grounds** |
| 8 | Entity linking, relation extraction & knowledge graph (Neo4j) | Planned |
| 9 | Topic modeling, cultural classification, bias measurement, temporal analysis | Planned |
| 10 | Dashboard (Streamlit → Hugging Face Spaces) | Planned |

## Repository Structure

```
configs/            Source, quality, and heritage-entity configuration (YAML)
src/
  ingestion/        Collectors, pipeline orchestration, schema, quality, dedup, NER
    collectors/     Per-source collectors (Wikipedia AR/EN, Semantic Scholar, GDELT, WAFA; base class for the rest)
  preprocessing/    Arabic text normalization
  utils/            Collection logging
  rag/              Chunking, embedding, pgvector index, retriever, generator (done)
  knowlegde_graph/  Entity canonicalization, linking, relations, KG store (planned — empty)
  nlp/              Topic modeling, cultural classification, bias, temporal analysis (planned — empty)
  api/              RAG API endpoint (planned — empty)
  frontend/         Dashboard (planned — empty)
eval/               Evaluation harness (done for NER/retrieval/RAG — see eval/gold/, eval_reports/, ROADMAP.md); KG eval planned
scripts/            Runnable entrypoints (collection, NER, HF export/publish, seed audit)
tests/              Unit tests for schema, quality, dedup, normalizer, NER, export, collectors
docs/               Publishing guide, licensing checklist, supporting documentation
reports/            Generated audit reports (seed categories)
data/               Local corpus artifacts (git-ignored)
main.py             Pipeline entrypoint
ROADMAP.md          Full milestone breakdown, technical decisions, execution plan
```

## Current Status

The corpus is now multi-source, and the Arabic Wikipedia path is functional end to end,
**including a working RAG system**. Arabic Wikipedia contributes 484 quality/duplicate-filtered
documents; English Wikipedia, Semantic Scholar (now with full open-access paper text where
available, not just abstracts), GDELT (international news coverage of Palestinian
culture/heritage), and WAFA (Palestinian news agency) are all collecting for real. This corpus
is currently **private research use** (not published/redistributed), so
`scripts/export_to_hf.py` exports everything by default rather than gating on
`license_status` — that field is still tracked accurately per document (see
`docs/licensing_checklist.md`) so a rights-cleared `--clear-only` public subset stays possible
later without re-collecting anything. Deduplication is persistent across runs and sources
(Postgres-backed), not just within a single run. NER has been run across the accepted corpus.
The unified schema, quality/dedup logic, seed-category audit tooling (including a live
off-target-category filter, not just post-hoc auditing), and Hugging Face export are all
working and covered by tests. RAG (chunking, embedding with Qwen3-Embedding, a pgvector index,
retriever, and an Ollama-based local-first generator behind a stable interface) has been built
and verified against the real corpus — 484 documents chunked into 1282 passages, fully embedded
and indexed, with `scripts/ask.py` returning grounded, cited answers. See `ROADMAP.md` for the
technical-decision reasoning and what's next.

Generated datasets are intentionally git-ignored; see `docs/huggingface_publishing_guide.md`.
The RAG generator requires [Ollama](https://ollama.com) running locally with the model named
in `configs/rag.yaml`'s `generation.model` pulled (`ollama pull <model>`).

## Data Collection

All five collectors below implement the same `BaseCollector` interface and go through the
same shared pipeline (`src/ingestion/pipeline.py`): collect → Arabic/English normalization →
quality scoring → deduplication → append to JSONL. Every command is safe to re-run — outputs
are append-mode, so re-running (or adding a new source) grows the corpus instead of
overwriting it.

### Prerequisites

- `uv sync` (installs everything, including collector-specific deps like `pdfplumber` for
  Semantic Scholar PDF extraction).
- **PostgreSQL is optional but recommended**: `docker compose up -d` starts a pgvector-enabled
  Postgres. Collection works without it — quality scoring, normalization, and dedup all still
  run — but deduplication falls back to **in-memory-only for that single run** (a warning is
  logged) instead of checking against everything collected previously. Run `docker compose up
  -d` once before your first real collection session if you plan to run collectors more than
  once or across sources.
- No API keys are required for any of the four external sources — all are public,
  unauthenticated APIs/pages.

### Commands, outputs, and requirements

| Source | Command | Output JSONL | Docker/Postgres |
|---|---|---|---|
| Wikipedia (Arabic) | `uv run python main.py --max-docs 100` | `data/processed/wikipedia_ar_documents.jsonl` | Optional (recommended) |
| Wikipedia (English) | `uv run python main.py --language en --max-docs 100` | `data/processed/wikipedia_en_documents.jsonl` | Optional (recommended) |
| Semantic Scholar (papers, full OA text where available) | `uv run python scripts/collect_semantic_scholar.py --max-docs 30` | `data/processed/semantic_scholar_documents.jsonl` | Optional (recommended) |
| WAFA (Palestinian news agency) | `uv run python scripts/collect_wafa.py --max-docs 30` | `data/processed/wafa_documents.jsonl` | Optional (recommended) |
| GDELT (international coverage of Palestinian culture/heritage) | `uv run python scripts/collect_gdelt.py --max-docs 30` | `data/processed/gdelt_documents.jsonl` | Optional (recommended) |

Each command also writes a rejected-documents JSONL (`data/metadata/<source>_rejected.jsonl`)
and a per-run stats JSON (`data/metadata/<source>_stats.json`) — see "Checking results" below.
All commands print their stats JSON to the terminal when they finish, so you don't need to
open the files just to see whether a run worked.

### Recommended corpus sizes for a first pass

For a first corpus, prioritize **source diversity over per-source volume** — a smaller,
varied corpus is more useful than one source maxed out. A reasonable starting split:

| Source | Suggested `--max-docs` | Why |
|---|---|---|
| Wikipedia AR | 200–300 (already have 484 from earlier runs) | Largest, cleanest, already the backbone of the corpus |
| Wikipedia EN | 50–100 | Useful for English-language coverage, but a secondary source |
| Semantic Scholar | 30–50 | Slow (shared rate-limited API) and low yield-per-query; don't over-request |
| WAFA | 30–50 | High per-document quality but only one outlet's editorial voice — more volume doesn't add diversity |
| GDELT | 30–50 | Valuable specifically *because* it's diverse (many outlets/countries), not because of volume |

Run each collector once at these sizes, check the results (below), then decide whether a
specific source is worth a second, larger run — rather than guessing a large number up front.

### How persistent deduplication works

`PersistentDuplicationIndex` (`src/ingestion/deduplication.py`) uses MinHash/LSH near-duplicate
detection, same as before, but now backed by a Postgres table
(`ingestion_dedup_index`) instead of an in-memory set that resets every run:

- On startup, it loads every previously-seen document's MinHash signature from Postgres into
  the LSH index — so a document collected by an *earlier* run, or by a *different source*,
  is caught as a duplicate even though this run's collector never saw it before.
- Every newly-accepted, non-duplicate document's signature is written back to Postgres
  immediately, so the next run (any source) sees it too.
- This is source-agnostic: if WAFA and GDELT both surface the same underlying article (e.g.
  a WAFA piece a wire service republished), the second one collected is correctly flagged as
  a duplicate of the first, regardless of run order.
- Without Postgres reachable, this degrades to a fresh in-memory index per run (a logged
  warning, not a crash) — duplicates within *that run* are still caught, but nothing persists
  across runs.

### Checking collection results and quality

Each run prints (and saves to `data/metadata/<source>_stats.json`) a summary with
`attempted_documents`, `accepted_documents`, `rejected_documents`, `duplicate_documents`,
`total_words`, `average_document_length`, `quality_decision_distribution` (accept /
accept_with_warning / reject / hard_reject), `rejection_reason_distribution`, and
`seed_category_distribution`. That's usually enough to sanity-check one run. For a broader
look across the whole corpus — sources, languages, lengths, quality — inspect the JSONL
directly, e.g.:

```powershell
uv run python -c "
import json, glob
from collections import Counter
docs = [json.loads(l) for f in glob.glob('data/processed/*_documents.jsonl') for l in open(f, encoding='utf-8')]
print('total docs:', len(docs))
print('by source_id:', Counter(d['source_id'] for d in docs))
print('by language:', Counter(d.get('language') for d in docs))
print('by license_status:', Counter(d.get('license_status') for d in docs))
lengths = [d['word_count'] for d in docs]
print('avg words:', sum(lengths) / len(lengths), 'min:', min(lengths), 'max:', max(lengths))
"
```

Also worth spot-checking a few actual rejected documents (`data/metadata/<source>_rejected.jsonl`)
if `rejected_documents` looks high for a source — the `rejection_reason` field on each record
says exactly why (usually `too_short` or `duplicate`).

### Exporting

```powershell
uv run python scripts/export_to_hf.py                # everything, by default (private research corpus)
uv run python scripts/export_to_hf.py --clear-only    # restrict to license_status == "clear", for a public-release subset
```

`export_to_hf.py` reads one JSONL file at a time (`--input`, defaults to the Wikipedia AR
file) — to export the combined multi-source corpus, first concatenate the JSONL files you
want into one file, then point `--input` at it.

### Known limitations / rate limits

- **Semantic Scholar** is a shared, unauthenticated public API and 429s (rate-limited)
  regularly in practice, even with exponential backoff retries built in — expect some queries
  to return fewer documents than requested, especially on repeated runs in a short window.
- **GDELT** enforces a hard ≥5s delay between requests (respected via `rate_limit_delay` in
  `configs/sources.yaml`) and only fetches the underlying article when that domain's
  `robots.txt` allows it — some GDELT-indexed articles will be skipped for that reason, by
  design, not as a bug.
- **WAFA** discovery walks backward through daily sitemaps (`max_days_back`, default 60 days
  in `configs/sources.yaml`) — very old articles outside that window won't be found.
- **Semantic Scholar full-text** only succeeds when the paper has an open-access PDF *and*
  it extracts cleanly — most papers (roughly 3 in 4, in verified testing) fall back to
  abstract-only, which is expected, not a failure.
- **Palestine Remembered and Nakba Archive are intentionally not implemented** — the former
  is behind active bot-detection this project won't bypass; the latter raises a
  survivor-testimony consent question distinct from licensing. See `ROADMAP.md` Track D.

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
5. Track `license_status` accurately on every document even though the private-use corpus
   doesn't gate exports on it today — respect it (via `--clear-only`) before any public
   release, and do not bypass active bot-detection to reach a source, regardless of use.
