# Roadmap — Palestinian Cultural Knowledge Platform

This is the project-management document for the full platform: ingestion → RAG →
knowledge graph → analysis (topic modeling, cultural classification, bias measurement,
temporal analysis) → dashboard. It supersedes the earlier RAG-only roadmap.

## What changed from the RAG-only plan

The previous version optimized *only* for time-to-first-RAG and pushed everything else
(KG, entity linking, topic modeling, bias measurement, temporal analysis, dashboard) into
an unscoped "post-MVP backlog." That was right about sequencing — RAG genuinely does not
depend on the KG — but wrong to leave the rest undesigned: this project's actual purpose
(recovering Palestinian cultural identity from text, per the HuggingFace publishing guide,
and the original "Digital Representation Bias" research framing this project grew from)
lives largely in the phases that plan treated as optional.

This version keeps the **same RAG-first critical path** — it is still the fastest way to a
working system and still the right place to start — but now:

1. Every phase from the original 9-phase vision is a fully specified milestone, not a bullet.
2. The **data contracts are designed once**, up front, to hold every phase's output — so
   expanding sources, adding the KG, or adding bias/topic/temporal analysis later never
   requires re-shaping what already exists (§2).
3. Every phase that produces a model or a score (NER, embeddings, retrieval, RAG, KG) gets
   an evaluation milestone, not just the ones that had one before.
4. The four technical decisions from the last roadmap (embedding model, vector store,
   chunking, LLM) are re-argued from scratch — see §3 — because "we already decided" is not
   a reason if a better long-term fit exists.

---

## 1. Definition of done, by horizon

**MVP (Track B, critical path) — ✅ ACHIEVED:** a working Arabic RAG system over the
*existing* corpus — ask a question, get a grounded, cited answer. Same DoD as the prior
roadmap, now verified against the real, fully-indexed 484-document / 1282-chunk corpus:

```powershell
uv run python scripts/ask.py "ما هي الكنافة النابلسية؟"
# → grounded Arabic answer + [1] Title — URL citations
```

**Full platform (everything else):** the corpus is multi-source and licensed for
publication; every document carries linked entities, a topic label, a content category, and
a decade bucket; the entities and their relations form a queryable knowledge graph; a
bias-measurement report quantifies how conflict-framing vs. culture-framing differs across
sources; a public dashboard exposes all of it. RAG, KG, and analysis all read from the same
underlying corpus and entity records — no phase re-derives what an earlier phase produced.

---

## 2. Core data contracts (designed once, extended never)

This is the piece that prevents future rewrites. Read this section before starting any
track below.

**`DocumentMetadata` (already in `src/ingestion/schemas.py`) does not change shape again.**
It already has forward-compatible hooks for everything downstream: `entities` (NER),
`topic_id`/`topic_label` (topic modeling), `category`/`category_confidence` (cultural
classification), `decade` (temporal analysis), `embedding_id` (RAG). This was correctly
designed the first time — the mistake would be bending it further. Every new capability
below gets its **own sibling model** that references `doc_id`, rather than growing new
fields onto the document record:

```python
# src/rag/schemas.py  — new, additive
class Chunk(BaseModel):
    chunk_id: str            # sha256(doc_id + chunk_index + chunking_version)
    doc_id: str               # FK -> DocumentMetadata.doc_id
    chunk_index: int
    text: str
    token_count: int
    start_char: int
    end_char: int
    chunking_version: str     # e.g. "recursive-512-v1" — bump on strategy change, don't mutate
    embedding_model: str | None
    embedding_version: str | None
    # denormalized for filter-without-join at retrieval time
    title: str | None
    source_url: str | None
    credibility: CredibilityTier
    quality_score: float | None
    seed_category: str | None

# src/knowlegde_graph/schemas.py  — new, additive (dir predates this plan; rename is optional cleanup, not required)
class KGEntity(BaseModel):
    entity_id: str             # canonical id: sha256(normalized_name + type) until linked
    canonical_name: str
    type: str                   # reuses entity_extractor.py's existing types (PERSON, HERITAGE_FOOD, ...)
    wikidata_qid: str | None    # filled by entity linking (Track E)
    mention_count: int
    source_doc_ids: list[str]

class KGRelation(BaseModel):
    relation_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    confidence: float
    source_doc_id: str
    evidence_sentence: str | None
```

Why this shape holds up across every later phase:

- **Chunking never blocks re-embedding, and re-embedding never blocks re-chunking.**
  `chunking_version` / `embedding_model` / `embedding_version` are columns, not table
  identity — switching the embedding model (§3) or the chunking strategy (§3) is a new row
  batch with a new version tag, not a schema migration.
- **`KGEntity` is a canonicalization of what NER already emits.** `entity_extractor.py`'s
  `_aggregate()` already produces `{text, normalized, type, mention_count, positions}` per
  document. `KGEntity` is the same shape lifted to corpus scope (dedup across documents,
  plus the Wikidata link). Track E is "aggregate what exists," not "invent a new pipeline."
- **Filter-then-retrieve is free.** Because `credibility`, `quality_score`, and
  `seed_category` are denormalized onto `Chunk`, retrieval can do
  `WHERE credibility = 'tier_1' AND quality_score > 0.7` in plain SQL instead of a second
  filtering layer — this only works because of the vector-store decision in §3.

**Shared evaluation contract** — every eval milestone below (NER, embeddings, retrieval,
RAG, KG) writes to the same shape so results are comparable and dashboards (Track G) can
render them uniformly, instead of five one-off report formats:

```python
# eval/schemas.py
class EvalReport(BaseModel):
    eval_name: str            # "ner_v1", "retrieval_v1", ...
    run_at: str
    dataset_size: int
    metrics: dict[str, float]  # {"precision": 0.81, "recall": 0.74, ...}
    notes: str | None
```

**Directory mapping** — the empty scaffolding already in the repo maps directly onto the
tracks below; no new top-level directories are needed:

| Existing stub | Track that fills it |
|---|---|
| `src/rag/` | Track B (MVP) |
| `src/knowlegde_graph/` | Track E |
| `src/nlp/` | Track F (topic modeling, classification, bias, temporal) |
| `src/api/` | Track G (serving layer) |
| `src/frontend/` | Track G (dashboard) |
| *(new)* `eval/` | Tracks C, plus an eval module per later phase |

---

## 3. Technical decisions, reconsidered

Each of the prior roadmap's four picks, re-argued against current alternatives rather than
assumed. Sources are inline; this section is the record of *why*, so it doesn't need
re-litigating later.

### 3.1 Embedding model

| Candidate | Verdict |
|---|---|
| **Qwen3-Embedding-0.6B — chosen default** | MTEB-leading among open models even at this size ("outperforms models twice its size on multilingual retrieval benchmarks"); the 8B variant tops the MTEB multilingual leaderboard outright, beating OpenAI's and Google's embedding APIs by 7–10 points, so the family has headroom to grow into. Strong Arabic performance via Qwen3's broader HELM Arabic leadership. Runs on 8GB VRAM or CPU; ~640MB at Q4. Native long context (up to 32K tokens) removes chunk-size ceiling surprises. Zero cost, fully local. |
| BGE-M3 | Also excellent and specifically validated on Arabic (see the dedicated ArabicMTEB/Swan benchmark). Its distinguishing feature is *built-in* hybrid retrieval — one model produces dense, sparse, and multi-vector (ColBERT-style) representations simultaneously, which would let hybrid search (§ Track C tuning) skip standing up BM25 separately. Kept as the fallback if retrieval eval (C2) shows dense-only recall is insufficient. |
| Qwen3-Embedding-4B / 8B | Same family as the default, no interface change to adopt — the upgrade path if C1 (embedding eval) shows the 0.6B model under-serving. |
| Voyage (`voyage-3` / multilingual) | Hosted, strong quality, but paid and non-local — kept only as a no-GPU-dev fallback, not a default, per the project's free/local-first direction. |
| E5-family | Competitive but not table-stakes ahead of Qwen3/BGE-M3 on the current MTEB multilingual results; no distinguishing reason to prefer it here. |

**Decision:** default to **Qwen3-Embedding-0.6B**, self-hosted. Escalate within the same
family (4B → 8B) if C1 shows a gap; switch to BGE-M3 only if C2 specifically shows the need
for built-in hybrid retrieval.

### 3.2 Vector store

| Candidate | Verdict |
|---|---|
| **pgvector on PostgreSQL — chosen default** | `psycopg2-binary` is *already* a project dependency — this introduces no new system. Production-grade at this project's realistic scale: current guidance is that pgvector comfortably serves RAG workloads under a few million vectors, which this corpus won't approach even fully expanded (50K docs × ~6 chunks ≈ 300K rows). Because vectors sit in the same database as `credibility`, `quality_score`, `seed_category`, `decade`, `category` (§2), retrieval filtering is plain SQL — no second filtering system, ever. This directly satisfies the "no future rewrite" requirement: there is no planned migration off it. |
| Chroma | Fastest zero-setup option — good for a same-day prototyping spike — but explicitly **not** the standing store. The exact anti-pattern this roadmap avoids is "prototype in Chroma, rewrite for production later" (recall the recommendation in general RAG guidance: "start with Chroma to ship fast, migrate to Qdrant when you hit scale"). Skipping straight to the store that doesn't need a later migration is cheaper for a two-person team than paying that migration twice. |
| Qdrant | The correct choice **if** the corpus reaches tens of millions of vectors or needs sub-10ms ANN at high QPS. Neither applies to this project's realistic horizon — noted as the explicit scale-up path, not built now. |
| Weaviate | Attractive for automatic embedding modules and native hybrid search, but introduces a second database system for no benefit pgvector doesn't already cover at this scale. |

**Decision:** **pgvector**, from day one. Add a `docker-compose.yml` Postgres service with
the `pgvector` extension — the only new setup cost, on the order of an hour.

### 3.3 Chunking strategy

Current 2026 guidance is consistent: **recursive/fixed-size token chunking is the strongest
simple default** (a February 2026 seven-strategy vendor benchmark ranked it first at the
512-token range), while **semantic chunking is the quality ceiling but costs ~14× more
compute** — worth paying for only when it measurably moves retrieval metrics.

**Decision:** default to **recursive, sentence-aware chunking at ~500 tokens with ~15%
(≈75-token) overlap**, reusing `split_sentences()` / `tokenize_with_offsets()` already in
`src/ingestion/entity_extractor.py:110` — using the *same* segmentation NER already uses
means chunk boundaries and entity-mention boundaries agree, which matters later when Track E
links an entity mention to the chunk it came from. Semantic chunking is explicitly gated
behind Track C's retrieval eval: adopt it only if recall is measurably short of target and
the `chunking_version` field (§2) makes that an additive re-index, not a rewrite.

### 3.4 LLM for generation — free/local first

Requirement: prioritize free or local over paid APIs, judged on quality, cost, Arabic
performance, and long-term maintainability.

| Candidate | Verdict |
|---|---|
| **Ollama + Qwen3 (local) — chosen default** | Zero per-token cost at any volume. Full data control — a real, non-hypothetical concern for a corpus of Palestinian cultural and historical material, not a hypothetical one (see Gemini caveat below). Qwen3 has the strongest Arabic support of the broadly-multilingual open families (leads HELM Arabic among 8B-class multilingual models) plus by far the most active Ollama/tooling ecosystem of any local candidate — the "long-term maintainability" tiebreaker. No roster or rate-limit churn: a locally-hosted open-weight model is a decision the team fully controls, unlike every hosted free tier below. |
| Gemini 2.5 Flash (free tier) | Real and generous — 1,500 requests/day, 1M TPM, no card required — a legitimate fallback for a dev without a capable GPU. **Caveat that must be a conscious choice, not a default:** free-tier prompts and responses may be used to improve Google's products. For a corpus centered on Palestinian cultural identity, that's worth the team explicitly deciding on, not defaulting into. Also unstable as a long-term dependency — Gemini pulled Pro from the free tier in April 2026 with no notice period visible from outside. |
| OpenRouter free models | Useful for the *initial dev spike* only (mock-retriever generator development, per the original MVP plan's Track B). Not recommended as a standing dependency: the free roster "shifts constantly" — trackers reported 18–29 free models in the same month (July 2026) — and free-tier limits are tight (20 req/min, 50–1000/day). Fine for prototyping, wrong for something the team depends on. |
| Grok API | No durable, guaranteed free tier as of current docs; the advertised $175/month credit requires enrolling in a data-sharing program — a stronger version of the Gemini caveat, for less certainty. Not recommended. |
| Claude API (paid, for reference) | Out of scope given the free/local-first requirement, but worth naming as the quality ceiling if the team ever needs a paid fallback: `claude-sonnet-5` or `claude-haiku-4-5` for cost-sensitive volume. Not part of any default path here. |

**Sizing Qwen3 to hardware** (pick one, same `Generator` interface either way):

| Dev hardware | Model |
|---|---|
| Modest laptop, no dedicated GPU | Qwen3-4B or -8B (quantized) |
| Single consumer GPU, 16–24GB VRAM — **recommended default** | Qwen3-14B |
| Strong workstation GPU, 24GB+ | Qwen3-32B |
| No GPU at all | Fall back to Gemini 2.5 Flash free tier (same interface, see caveat above) |

**Decision:** default to **Qwen3 via Ollama** (size per hardware table), behind the same
`Generator` protocol from the MVP plan so swapping to Gemini or OpenRouter is a config
change, never a pipeline change. Explicitly A/B this default in Track C's RAG eval against
**Jais-30B** (Core42/G42, Arabic-native, Apache 2.0) and/or **Falcon-H1-Arabic** (TII) —
both may produce more culturally fluent, idiomatic Arabic on culture-specific questions than
a general multilingual model, at the cost of a narrower ecosystem. Switch the default only
if that eval shows a real gap on Arabic-culture questions specifically — don't guess.

---

## 4. Tracks

Each track is independently schedulable once its dependencies clear. **A/B/C are the
critical path to MVP** (unchanged shape from the prior roadmap, now re-grounded in §2/§3).
Everything else is fully designed but sequenced after, per the reprioritization in the intro.

### Track A — Foundation (P0, gates B and every later track) — ✅ DONE

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. | Status |
|---|---|:--:|---|---|---|:--:|---|
| A1 | Core contracts: `Chunk`, `KGEntity`, `KGRelation`, `EvalReport` (§2) | P0 | — | Sequential — do first, together | `src/rag/schemas.py`, `src/knowlegde_graph/schemas.py`, `eval/schemas.py` | 0.5d | Done |
| A2 | Dependencies & environment: `pyproject.toml` additions (`ollama`/`sentence-transformers`/`pgvector`; `psycopg2` already present), `docker-compose.yml` for Postgres+pgvector, `configs/rag.yaml`. **Verify install resolves on Python 3.14 day one** (R1 below) | P0 | — | Parallel | Updated `pyproject.toml`, working `uv sync`, running local Postgres | 0.5d | Done — see note below |
| A3 | Fix the known quality-scoring bugs before more data flows through them: non-monotonic richness curve and dead `quality_thresholds.yaml` (config exists but isn't read) in `src/ingestion/quality_checker.py` | P1 | — | Parallel | Passing `tests/test_quality_checker.py` + a regression case for the 200-word boundary | 0.5d | Done |

**A2 follow-up (found during Track B, not originally scoped):** the base install (R1) resolved cleanly on Python 3.14, but the *CUDA* build of torch didn't come along for free. `camel-tools`/`sentence-transformers` pull torch in only as a transitive dependency, and `[tool.uv.sources]` overrides only bind to **direct** project dependencies — so torch silently kept resolving the CPU-only PyPI wheel even after adding a `[tool.uv.index]` pointing at PyTorch's CUDA index. Fix: add `torch` as an explicit direct dependency (pinned loosely, `>=2.12.0`) so the source override has something to attach to, then `uv lock --upgrade-package torch`. Also had to pick the right CUDA tag by checking which ones actually publish a `cp314`-`win_amd64` wheel (this project's Python), not just any CUDA tag — landed on `cu130`, matching the driver's reported CUDA 13.0 ceiling. See `pyproject.toml`'s `[tool.uv.sources]`/`[tool.uv.index]` block.

### Track B — Path to RAG MVP (critical path, unchanged shape) — ✅ DONE, verified end to end

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. | Status |
|---|---|:--:|---|---|---|:--:|---|
| B1 | Chunker — implements §3.3 | P0 | A1 | Sequential | `src/rag/chunker.py` + `data/processed/chunks.jsonl` | 1d | Done — 484 docs → 1282 chunks |
| B2 | Embedding pipeline — implements §3.1 default | P0 | A2, B1 | Sequential | `src/rag/embedder.py` | 1d | Done |
| B3 | Vector index — implements §3.2 default | P0 | B2 | Sequential | `src/rag/index.py`, persisted pgvector table | 1d | Done |
| B4 | Retriever (dense top-k; filter-then-retrieve per §2) | P0 | B3, A1 | Sequential | `src/rag/retriever.py` | 1d | Done |
| C1(gen) | Generator — implements §3.4 default, developed against a mock retriever | P0 | A1, A2 | Parallel to B1–B4 | `src/rag/generator.py` | 1.5d | Done |
| C2(gen) | Citation/answer assembly | P0 | A1 | Parallel to B1–B4 | `src/rag/answer.py` | 0.5d | Done |
| C3(gen) | CLI (`scripts/ask.py`) | P0 | C1(gen) | Sequential, Parallel track | `scripts/ask.py` | 1d | Done |
| B5 | **Integration = MVP** | P0 | B4, C1(gen), C2(gen), C3(gen) | Sequential — merge, both devs | Working `RAGPipeline.ask()` end to end | 1d | Done |

*(Labels reuse the prior roadmap's task IDs where the work is identical, so anyone
cross-referencing the earlier plan can still find it.)*

**Verified, not just built:** the full 484-document corpus was chunked (1282 chunks), embedded with Qwen3-Embedding-0.6B, and indexed into pgvector — on GPU (once the CUDA fix above landed), embedding+indexing all 1282 chunks took **3m11s**; the earlier CPU-only attempt hadn't finished 352 of them after 20+ minutes. `scripts/ask.py "ما هي الكنافة النابلسية؟"` against the full indexed corpus returned a grounded, correctly-cited Arabic answer (verified with the already-pulled `llama3.1:8b` as a stand-in for the configured `qwen3:14b`, which isn't downloaded yet).

**A real finding from that run, not a bug:** none of the 5 retrieved sources were a dedicated "Knafeh" article — checking directly, **no such article exists** in this 484-document corpus (`كنافة` appears only as a passing mention in 8 other documents). The retriever correctly surfaced the closest real matches instead of failing, and the generator stayed grounded in them rather than inventing a source. This is a genuine corpus-coverage gap, exactly the kind of thing Track C's retrieval/RAG evals (below) and Track D's multi-source expansion exist to catch and fix — not a flaw in the RAG mechanics themselves.

### Track C — Evaluation (P0/P1 — starts immediately after B5, before any expansion)

Nothing past this point should be built on an unmeasured foundation. This is where the "add
evaluation everywhere" requirement lives.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. |
|---|---|:--:|---|---|---|:--:|
| C1 | **Embedding eval** — is §3.1's default actually retrieving well on this corpus? Sample queries + known-relevant chunks, measure embedding-similarity quality directly | P0 | B2 | Parallel-after B2 | `eval/embedding_eval.py` → `EvalReport` | 1d |
| C2 | **Retrieval eval** — recall@k, MRR against a small hand-built relevance set; this is also the gate for the semantic-chunking decision in §3.3 | P0 | B4 | Sequential after C1 | `eval/retrieval_eval.py` → `EvalReport` | 1d |
| C3 | **RAG (end-to-end) eval** — faithfulness/groundedness (Claude-or-similar-as-judge + manual spot check), and the Qwen3-vs-Jais-30B-vs-Falcon-H1-Arabic A/B from §3.4 | P0 | B5 | Sequential after C2 | `eval/rag_eval.py` → `EvalReport` + model-choice writeup | 1.5d |
| C4 | **NER eval** — NER already shipped (`wikipedia_ar_documents.ner.jsonl` exists) with zero measurement: CAMeL confidence is a hardcoded placeholder, heritage matches are 1.0 by fiat. Build a small gold-annotated set, report precision/recall | P0 | — (independent of B/C1–3) | Parallel to all of Track C | `eval/ner_eval.py` → `EvalReport` + gold set | 1.5d |
| C5 | Tuning pass — chunk size/overlap, top-k, prompt — using C1–C3 results | P1 | C1, C2, C3 | Sequential | Tuned `configs/rag.yaml` + before/after numbers | 1d |

### Track D — Multi-source expansion (P1/P2)

`BaseCollector` (`src/ingestion/base_collector.py`) is already the right abstraction —
every source below is a new subclass, not a redesign.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. |
|---|---|:--:|---|---|---|:--:|
| D1 | Licensing/rights gate — a per-source checklist (redistribution rights, attribution requirements) that must pass before a collector's output reaches the published corpus. Wikipedia (CC-BY-SA) already clears it; news/archive sources generally do not by default | P1 | — | Parallel | `docs/licensing_checklist.md` + a `license_status` field check in the export step | 1d |
| D2 | GDELT collector (config already stubbed, disabled) | P2 | D1 | Parallel across sources | `src/ingestion/collectors/gdelt_collector.py` | 2d |
| D3 | WAFA / archive-source collectors (config already stubbed, disabled) | P2 | D1 | Parallel across sources | `src/ingestion/collectors/*_collector.py` | 2d each |
| D4 | Semantic Scholar collector (academic sources) | P2 | D1 | Parallel across sources | `src/ingestion/collectors/semantic_scholar_collector.py` | 2d |
| D5 | Collection-time relevance gating — reuse the existing seed-audit script's `RELEVANCE_KEYWORDS` as a live accept filter, not just a post-hoc audit (addresses the topical-drift finding: Jordanian/Lebanese/Syrian/Israeli cuisine leaking into seed-category traversal) | P1 | — | Parallel | Updated `wikipedia_collector.py` filter | 0.5d |
| D6 | Persistent / incremental deduplication — the LSH index is currently in-memory-per-run and outputs open in overwrite mode, so nothing dedups against the existing corpus. This becomes a real blocker once D2–D4 land, not before | P1 | D2 (or first new source) | Sequential, gates further expansion | Persisted `DuplicationIndex` (e.g. serialized MinHash bands in Postgres alongside `Chunk`) | 1.5d |
| D7 | English pipeline — `pipeline.py` currently hardcodes `ar`; `sources.yaml` already has an `en` config block ready | P2 | — | Parallel | Parameterized `pipeline.py --language en` | 1d |

### Track E — Knowledge layer (P1/P2)

Builds directly on the `KGEntity`/`KGRelation` shapes from §2, which are themselves an
aggregation of what `entity_extractor.py` already emits.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. |
|---|---|:--:|---|---|---|:--:|
| E1 | Entity canonicalization — aggregate the existing per-document `entities` output into corpus-scope `KGEntity` records (dedup by normalized name + type) | P1 | A1, C4 (trust NER first) | Sequential | `src/knowlegde_graph/canonicalize.py` | 1.5d |
| E2 | Entity linking to Wikidata (mGENRE, or a simpler alias-table approach first — see note) | P2 | E1 | Sequential | `wikidata_qid` populated on `KGEntity` | 2–3d |
| E3 | Relation extraction (LLM-prompted, using the Track B/C generator) | P2 | E1 | Parallel to E2 | `src/knowlegde_graph/relations.py` → `KGRelation` records | 2d |
| E4 | KG store — **NetworkX in-process for the first working prototype** (matches the project's own iterative-validation practice: prove the graph is useful before standing up infrastructure for it); **migrate to Neo4j Community** (free, self-hosted, mature Cypher for the multi-hop queries the dashboard's KG explorer will need) once E1–E3 are validated | P2 | E1, E3 | Sequential | Neo4j-backed KG, loader script | 2d |
| E5 | **KG eval** — precision of extracted relations against a hand-checked sample; entity-linking accuracy against a Wikidata gold sample | P1 | E2, E3 | Sequential after E2/E3 | `eval/kg_eval.py` → `EvalReport` | 1d |

*Note on E2: start with a cheap alias-table linker (exact/fuzzy match against a pre-pulled
Wikidata Palestine-entity SPARQL dump — the original blueprint already names
`query.wikidata.org` for this) before reaching for mGENRE's full model — consistent with
"validate cheap first."*

### Track F — Analysis & research phases (P2)

These are the phases that make this a *research* platform, not just a retrieval system —
carried over in full from the original 9-phase vision, not dropped.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. |
|---|---|:--:|---|---|---|:--:|
| F1 | Topic modeling (BERTopic over the embeddings from B2 — no new embedding step needed) | P2 | B2 | Parallel to Track E | `src/nlp/topic_model.py`, populates `topic_id`/`topic_label` on `DocumentMetadata` | 2d |
| F2 | Cultural/content classification — fine-tune AraBERT for the `ContentCategory` enum that already exists in `schemas.py` (conflict/culture/history/arts/…); annotate ~1,000–1,500 examples (LLM-assisted labeling, human-reviewed, per the original blueprint's own time-saving note) | P2 | — | Parallel to Track E/F1 | `src/nlp/classifier.py`, populates `category`/`category_confidence` | 3d |
| F3 | Bias measurement — topic-distribution comparison across sources, WEAT test on embeddings, an LLM probe (using the Track B/C generator) on culture-vs-conflict framing | P2 | F1, F2, D2–D4 (needs multiple sources to compare) | Sequential after F1/F2/D | `src/nlp/bias_measurement.py`, `eval/bias_eval.py` → `EvalReport` | 3d |
| F4 | Temporal analysis — the `decade` field already exists on `DocumentMetadata`; bucket embeddings by decade, measure semantic drift on key terms | P3 | B2 | Parallel to Track E/F1–F3 | `src/nlp/temporal_analysis.py` | 2d |

### Track G — Product surface (P2/P3)

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. |
|---|---|:--:|---|---|---|:--:|
| G1 | RAG API endpoint (FastAPI, wraps the Track B pipeline) | P1 | B5 | Parallel, can start right after MVP | `src/api/main.py` | 1d |
| G2 | Dashboard — Streamlit first (the original blueprint's own recommendation: "a working prototype in two hours," free HuggingFace Spaces hosting), Bias Meter / Topic Map / Timeline / KG Explorer panels as their data sources (Tracks E/F) land | P2 | G1, E4, F1, F3, F4 (per-panel, incremental) | Sequential per panel, otherwise parallel to E/F | `src/frontend/dashboard.py` | 2d + ~0.5d per panel |

### Track H — Infrastructure & hardening (P3 — deliberately last)

Everything here was explicitly deferred by the original MVP-first plan and stays deferred:
none of it blocks a working system, all of it matters once the system has real, sustained
usage rather than iterative research use.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. |
|---|---|:--:|---|---|---|:--:|
| H1 | Scheduler for recurring collection runs | P3 | D6 | Parallel | Cron/Task-scheduler config | 1d |
| H2 | Monitoring/observability (pipeline run health, quality-decision drift over time) | P3 | — | Parallel | Basic dashboards/alerts | 2d |
| H3 | Consolidate the two divergent NER code paths — `entity_extractor.py`/`run_ner.py` (the module actually used) vs. the older `Run_ner_on_corpus.py`/`heritage_ner_camel.py` prototype with a different heritage dictionary (`PAL_*` vs `HERITAGE_*` types) | P2 | — | Parallel, any time | Retire the prototype scripts | 0.5d |
| H4 | Scale-up migration off pgvector to Qdrant — **only if** corpus size or QPS ever actually approaches the thresholds in §3.2 | P3 | — | N/A until triggered | — | — |

---

## 5. Two-developer execution plan

**Phase 1 — MVP (critical path, ~1.5–2 weeks, unchanged from the prior plan):**

| | Dev A — Indexing/Retrieval | Dev B — Generation/Interface |
|---|---|---|
| Sync | A1 Contracts *(together)* | A1 Contracts *(together)* |
| Parallel | B1 → B2 → B3 → B4 | A2 → C1(gen) → C2(gen) → C3(gen) |
| Sync | B5 Integration = **MVP** *(together)* | B5 Integration = **MVP** *(together)* |

**Phase 2 — Evaluation (immediately after MVP, ~3–4 days, before any expansion starts):**

| | Dev A | Dev B |
|---|---|---|
| Parallel | C4 NER eval (independent — can even start during Phase 1 idle time) | C1 → C2 Embedding/Retrieval eval |
| Sequential | — | C3 RAG eval + model A/B |
| Sync | C5 Tuning *(together, brief)* | C5 Tuning *(together, brief)* |

**Phase 3 — Full vision (Tracks D–H, roughly 8–14 dev-weeks total across both developers,
depending on how many Track D sources are pursued — a rough range, not a commitment).** The
natural skill-aligned split, not a rigid mandate:

- **Dev A — ingestion & knowledge:** Track D (multi-source expansion, dedup, licensing) →
  Track E (entity linking, KG). Both are "more data flowing through the pipeline" work.
- **Dev B — analysis & product:** Track F (topic modeling, classification, bias, temporal)
  → Track G (API, dashboard). Both are "make sense of / expose the corpus" work.
- **Track H** is picked up opportunistically by whoever has slack — none of it gates
  anything else, which is exactly why it's P3.

Within Phase 3, F1 (topic modeling) and the start of E1 (entity canonicalization) can both
begin immediately after B2/C4 respectively — they don't need to wait for each other or for
D. F3 (bias measurement) is the one genuine cross-track dependency: it needs multiple
sources (D) to compare against, so schedule it after at least one non-Wikipedia source lands.

---

## 6. Risks

| ID | Risk | Impact | Mitigation |
|---|---|---|---|
| R1 | Python 3.14 may lack wheels for `pgvector`/`torch`/`sentence-transformers`/Ollama client libs | Blocks the whole stack | Verify in A2, day one; fall back to a pinned 3.11/3.12 venv for the RAG/ML stack specifically if needed |
| R2 | Local LLM (Qwen3) under-performs on Arabic-culture nuance vs. a native Arabic model | Weak answers despite "working" retrieval | C3's explicit Jais-30B/Falcon-H1-Arabic A/B — decide from data, not assumption |
| R3 | pgvector performance assumption (§3.2) is wrong at this project's actual eventual scale | Retrieval latency degrades | H4 is the pre-agreed escape hatch — the decision was made with its own exit condition, not open-endedly |
| R4 | Bias-measurement findings (Track F) are methodologically fragile (WEAT/framing classifiers are contestable) | Undermines the project's research credibility | Keep F3 conservative and documented — report methodology limitations alongside results, per the original blueprint's own emphasis on statistical rigor (p < 0.05 framing) |
| R5 | Multi-source expansion (Track D) hits licensing walls later than expected | Rework or a blocked publish | D1 runs *before* any collector in D2–D4, not after |
| R6 | Corpus too narrow to answer some questions even after expansion | "Insufficient context" responses | Acceptable outcome — the RAG prompt (Track B) already instructs the model to say so rather than invent |

---

## 7. The critical path, restated

**A1 → B1 → B2 → B3 → B4 → B5 = MVP.** Then **C1–C4 → C5** before anything expands. Past
that point, Tracks D through H are fully designed (§2's contracts already hold their output
shapes) but not sequence-critical — say no to reordering them ahead of evaluation, but the
order *among* D/E/F/G is a team-capacity decision, not an architectural one.
