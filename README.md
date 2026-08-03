# Palestinian Cultural Knowledge Platform

## Overview

The Palestinian Cultural Knowledge Platform builds a reproducible, quality-scored
corpus of Palestinian cultural text and turns it into structured, queryable knowledge.
The pipeline collects documents from credible sources, cleans and scores them against
an explicit quality contract, removes near-duplicates, and publishes a versioned
dataset. Downstream stages enrich the corpus with named entities, a knowledge graph,
and retrieval-augmented question answering.

The current implementation focuses on **Arabic Wikipedia** as a clean, licensable first
source. Ingestion, quality, deduplication, named-entity-recognition, a working
retrieval-augmented QA (RAG) system, and a first real knowledge graph (entity
canonicalization, Wikidata linking, LLM-prompted relation extraction, NetworkX graph store)
are all functional end to end over the real corpus. The remaining analysis phases (topic
modeling, cultural classification, bias measurement, temporal analysis) are fully designed —
see `ROADMAP.md` — and next in the build order, which reached RAG and a first KG pass without
narrowing the platform's scope.

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
  (Wikidata) ✓         ──► grounded, cited RAG answers ✓
        │
        ▼
  Relation extraction (LLM) ✓ ──► Knowledge graph (NetworkX) ✓ ──► Bias measurement ✗
        │                                                          (WEAT / framing / LLM probe)
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
| 8 | Entity canonicalization, Wikidata linking, LLM relation extraction, NetworkX KG | **Done — see ROADMAP.md; Neo4j migration deferred until the graph scales past the NetworkX prototype** |
| 9 | Topic modeling, cultural classification, bias measurement, temporal analysis | **Done — all four have real runs (see ROADMAP.md Track F); F2 is a deliberate zero-shot-LLM deviation from the fine-tuned-AraBERT plan; F1's auto-generated topic labels skew toward years/numbers, a real c-TF-IDF characteristic on this corpus, not a bug — see note** |
| 10 | Dashboard (Streamlit → Hugging Face Spaces), RAG API (FastAPI) | **Done — verified in a real browser, see ROADMAP.md Track G** |

## Repository Structure

```
configs/            Source, quality, and heritage-entity configuration (YAML)
src/
  ingestion/        Collectors, pipeline orchestration, schema, quality, dedup, NER
    collectors/     Per-source collectors (Wikipedia AR/EN, Semantic Scholar, GDELT, WAFA; base class for the rest)
  preprocessing/    Arabic text normalization
  utils/            Collection logging
  rag/              Chunking, embedding, pgvector index, retriever, generator (done)
  knowlegde_graph/  Entity canonicalization, Wikidata linking, LLM relation extraction, NetworkX graph store (done)
  nlp/              Topic modeling, content classification, bias measurement, temporal analysis (done — see ROADMAP.md Track F)
  monitoring/       Pipeline run health / quality-decision drift reporting (done)
  api/              RAG API endpoint — FastAPI (done)
  frontend/         Dashboard — Streamlit, plus pure data-loader functions (done)
eval/               Evaluation harness (done for NER/retrieval/RAG/KG — see eval/gold/, eval_reports/, ROADMAP.md)
scripts/            Runnable entrypoints (collection, NER, KG, topic/bias/temporal analysis, API/dashboard launch, scheduling, health report)
tests/              Unit tests for schema, quality, dedup, normalizer, NER, export, collectors, KG, NLP, API, monitoring
docs/               Publishing guide, licensing checklist, supporting documentation
reports/            Generated analysis reports (seed categories, temporal analysis, bias measurement)
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
and indexed, with `scripts/ask.py` returning grounded, cited answers. A first real knowledge
graph has also been built end to end, spanning every Arabic-capable source (Wikipedia AR,
WAFA, GDELT, Semantic Scholar) — 13,063 canonicalized entities, 950 linked to Wikidata QIDs,
585 LLM-extracted relations from a scaled-up 88-document pass, stored as a NetworkX graph
(`data/graph/kg_graph.graphml`) — see the Knowledge Graph section below and `ROADMAP.md` Track
E for the real numbers, including several real bugs found and fixed by actually running it at
scale (a Wikidata homonym-collision problem, a qwen3 "thinking mode" issue, and NER silently
producing garbage on non-Arabic text). See `ROADMAP.md` for the technical-decision reasoning
and what's next.

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

## Knowledge Graph

Four stages, run in order, each a thin script over `src/knowlegde_graph/`. Every stage's
output is JSONL (or GraphML for the graph itself), so you can inspect results at any point
with a plain text editor or `jq`/Python, not just by re-running the next stage. `build_kg_entities.py`
and `extract_kg_relations.py` both accept **multiple `--input` files**, so the KG can span
every collected source, not just whichever one happened to get NER run on it first.

**NER is Arabic-only** — confirmed directly by running it on English text, which produced
nonsense ("Israel" tagged PERSON, section headers tagged as entities). `scripts/run_ner.py`
now skips non-Arabic documents per-document (not per-source, since GDELT and Semantic Scholar
are language-mixed), marking them `ner_skipped_reason: "non_arabic_language"` rather than
silently mis-tagging them. Run it once per source before the KG stages:

```powershell
uv run python scripts/run_ner.py                                    # refresh Arabic Wikipedia
uv run python scripts/run_ner.py --input data/processed/wafa_documents.jsonl --output data/processed/wafa_documents.ner.jsonl
uv run python scripts/run_ner.py --input data/processed/gdelt_documents.jsonl --output data/processed/gdelt_documents.ner.jsonl
uv run python scripts/run_ner.py --input data/processed/semantic_scholar_documents.jsonl --output data/processed/semantic_scholar_documents.ner.jsonl
# English Wikipedia isn't worth running — 0% of it is Arabic, so every document would be skipped.
```

Then the KG itself:

```powershell
uv run python scripts/build_kg_entities.py --input data/processed/wikipedia_ar_documents.ner.jsonl `
    data/processed/wafa_documents.ner.jsonl data/processed/gdelt_documents.ner.jsonl data/processed/semantic_scholar_documents.ner.jsonl   # E1
uv run python scripts/fetch_wikidata_aliases.py --limit 4000        # E2a: fetch Wikidata dump (occasional, not per-run)
uv run python scripts/link_kg_entities.py                          # E2b: link -> data/entities/kg_entities.linked.jsonl
uv run python scripts/extract_kg_relations.py `
    --input data/processed/wikipedia_ar_documents.ner.jsonl data/processed/wafa_documents.ner.jsonl `
            data/processed/gdelt_documents.ner.jsonl data/processed/semantic_scholar_documents.ner.jsonl `
    --max-docs 15 --max-pairs-per-doc 15                            # E3: needs Ollama running; --max-docs is PER FILE
uv run python scripts/build_kg_graph.py                             # E4: -> data/graph/kg_graph.graphml
uv run python -m eval.kg_eval                                       # E5: precision/accuracy against hand-checked gold sets
```

`--max-docs` on `extract_kg_relations.py` applies **per input file**, not to the combined
total — Arabic Wikipedia's file is far larger than the others, so a combined-total cap would
starve every other source of documents entirely.

**Entity linking (E2)** matches canonicalized entities against a Wikidata alias table scoped
to Palestine-related items (`wdt:P17`/`P27`/`P495` = Q219060), not general Wikidata — it will
correctly *not* link entities like Israel, Jordan, or Egypt (out of scope by design), and it
also can't currently resolve historically-Palestinian cities now administered by Israel (e.g.
Haifa, Jaffa), since Wikidata's country property reflects present-day sovereignty, not
historical identity — a known, named gap, not a silent miss. On the multi-source corpus this
links ~7% of canonicalized entities (950/13,063) — expected for an alias-table approach
against a broad, mostly-generic NER entity set (most PERSON mentions especially won't have a
Wikidata entry at all).

**Relation extraction (E3)** makes one LLM call per candidate entity pair (two entities
co-occurring in the same sentence), so runtime scales with `--max-docs` (per file) ×
`--max-pairs-per-doc` × number of input files. It requires [Ollama](https://ollama.com)
running with the model configured in `configs/rag.yaml`'s `generation.model` pulled. Start
small and scale up once you've spot-checked the output — `eval/gold/kg_relations_gold.json`
shows the actual failure patterns found on a real 40-relation sample (precision 0.60): mostly
backward subject/object relations and multi-way "between X and Y" facts that don't fit a
binary relation cleanly, both worth knowing about before trusting the graph at face value.
That gold sample was drawn from an earlier Arabic-Wikipedia-only run — it's a snapshot
characterizing extraction quality, not something re-derived from the live, now multi-source
`data/graph/kg_relations.jsonl`.

**Inspecting the graph:**

```python
from src.knowlegde_graph.graph_store import load_graph, find_entities_by_name, neighbors_of
graph = load_graph("data/graph/kg_graph.graphml")
matches = find_entities_by_name(graph, "القدس")   # substring match on canonical_name
neighbors_of(graph, matches[0])                    # outgoing relations + target entity info
```

See `ROADMAP.md` Track E for the full real-numbers writeup, including two real bugs found by
actually running this at scale (a Wikidata homonym-collision problem that mislinked this
corpus's top three entities, and a qwen3 "thinking mode" issue that silently returned empty
relation-extraction responses) and how each was fixed.

## Analysis, API, and Dashboard

Four independent analyses (Track F) plus a product surface (Track G) over the corpus and
knowledge graph. All are real, working code with real runs — see `ROADMAP.md` Track F/G for
full findings, including one deliberate deviation (F2) and one real bug found and fixed (F1).

```powershell
uv run python scripts/run_topic_model.py                    # F1 — needs pgvector chunks (build_index.py)
uv run python scripts/run_content_classification.py --max-docs 30   # F2 — needs Ollama
uv run python scripts/run_bias_measurement.py                # F3 — needs F2's output; --skip-framing-probe to avoid Ollama
uv run python scripts/run_temporal_analysis.py                # F4 — no DB/Ollama needed

uv run uvicorn src.api.main:app --reload --port 8000          # G1 — RAG API (POST /ask, GET /health)
uv run streamlit run src/frontend/dashboard.py                 # G2 — dashboard (Overview, Topic Map, Timeline, Bias Meter, KG Explorer, Ask)
```

**F2 (content classification)** uses zero-shot LLM prompting (`qwen3` via Ollama) instead of
the originally-planned fine-tuned AraBERT model — a deliberate, documented deviation (see
`src/nlp/content_classifier.py`'s module docstring), not a silent scope cut. Real run: 75
documents classified across Wikipedia AR/WAFA/GDELT, 0 unparseable, a genuinely varied
distribution (conflict, culture, heritage, arts_literature, history, ...). Fine-tuning needs a
real labeled training set this session didn't have time to build; this classifier's own
high-confidence outputs are a plausible way to bootstrap one later.

**F3 (bias measurement)** produced the most substantively interesting result in Track F, real
and not hypothetical: a WEAT embedding-association effect size of **-1.612** (conflict-coded
terms associate strongly with violence-connotation words, culture-coded terms don't — the
intuitive, non-degenerate direction), and an LLM framing probe showing **WAFA's own coverage
skewed heavily conflict-framed (7/8 sampled) while GDELT skewed mixed (6/8) and Wikipedia AR
leaned non-conflict** — a real, measured difference in how a Palestinian news wire frames its
own reporting versus an encyclopedia.

**F1 (topic modeling)** real run: 41 topics found across 488/581 Wikipedia AR documents with
indexed chunks. Hit a real bug on first attempt — psycopg2's registered pgvector type
deserializes the `embedding` column into a `Vector` object, not a plain list, so `list(r[3])`
failed; fixed to `r[3].to_list()`, with a regression test. Auto-generated topic *labels* skew
toward years/numbers rather than Arabic content words (e.g. `"1948 / 1947 / 1967 / 1949"`) —
not a bug: c-TF-IDF ranks words frequent-within-cluster-but-rare-elsewhere, and common Arabic
words appear in nearly every chunk so they can never win that ranking, while a concentrated
year genuinely does. The *clustering* itself is real and meaningful (that 1948/1947/1967/1949
topic really does group Nakba/1948-war content). LLM-generated labels would read better but
aren't built yet — see `ROADMAP.md` Track F for the reasoning.

**G2 (dashboard)** was verified running in a real browser: every panel except "Ask" reads
from files the other tracks already produce, so it works even while Postgres is down — which
it genuinely was for most of this session, a real test of that design choice, not a
hypothetical one.

## Operations

```powershell
uv run python scripts/run_scheduled_collection.py --max-docs 50     # one full collection cycle across every source
uv run python scripts/collection_health_report.py                   # current snapshot + drift + anomalies
```

`run_scheduled_collection.py` runs once and exits — recurrence is the OS scheduler's job
(Windows Task Scheduler `schtasks` command in the script's own docstring, or cron), not a
custom daemon. Each cycle appends to `data/metadata/scheduled_run_log.jsonl`, which
`collection_health_report.py` reads for accept-rate drift and anomaly detection (a source
failing outright, or its accept rate dropping ≥30 points vs. its own previous run) — useful
from day one via each source's existing `*_stats.json`, more useful once real run history
accumulates.

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
