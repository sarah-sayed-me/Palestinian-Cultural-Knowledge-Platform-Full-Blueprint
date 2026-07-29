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

### Track C — Evaluation (P0/P1 — starts immediately after B5, before any expansion) — ✅ DONE

Nothing past this point should be built on an unmeasured foundation. This is where the "add
evaluation everywhere" requirement lives.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. | Status |
|---|---|:--:|---|---|---|:--:|---|
| C1 | **Embedding eval** — is §3.1's default actually retrieving well on this corpus? Sample queries + known-relevant chunks, measure embedding-similarity quality directly | P0 | B2 | Parallel-after B2 | `eval/embedding_eval.py` → `EvalReport` | 1d | Done — folded into C2 (see note) |
| C2 | **Retrieval eval** — recall@k, MRR against a small hand-built relevance set; this is also the gate for the semantic-chunking decision in §3.3 | P0 | B4 | Sequential after C1 | `eval/retrieval_eval.py` → `EvalReport` | 1d | Done — Recall@5 0.93, Recall@10 0.96, MRR 0.89 |
| C3 | **RAG (end-to-end) eval** — faithfulness/groundedness (Claude-or-similar-as-judge + manual spot check), and the Qwen3-vs-Jais-30B-vs-Falcon-H1-Arabic A/B from §3.4 | P0 | B5 | Sequential after C2 | `eval/rag_eval.py` → `EvalReport` + model-choice writeup | 1.5d | Done — qwen3:14b vs llama3.1:8b compared, near-identical (see note); Jais/Falcon untested |
| C4 | **NER eval** — NER already shipped (`wikipedia_ar_documents.ner.jsonl` exists) with zero measurement: CAMeL confidence is a hardcoded placeholder, heritage matches are 1.0 by fiat. Build a small gold-annotated set, report precision/recall | P0 | — (independent of B/C1–3) | Parallel to all of Track C | `eval/ner_eval.py` → `EvalReport` + gold set | 1.5d | Done — F1 0.47 (exact match; likely undercounted, see note) |
| C5 | Tuning pass — chunk size/overlap, top-k, prompt — using C1–C3 results | P1 | C1, C2, C3 | Sequential | Tuned `configs/rag.yaml` + before/after numbers | 1d | Done — one evidence-driven attempt, real (negative) result, see note |

**Real results, not simulated — and where the honest edges are:**

- **Gold data source:** 108 NER-annotated paragraphs (`eval/gold/ner_gold.json`) and 200 retrieval
  queries with relevant-document judgments (`eval/gold/retrieval_queries.json`), built externally
  and integrated into this repo. Both scripts (`eval/ner_eval.py`, `eval/retrieval_eval.py`,
  `eval/rag_eval.py`) run against the real live corpus/index/models, not fixtures.
- **C1 folded into C2:** the gold set's `relevant_para_ids` use a `<doc_id>_p<N>` paragraph
  numbering that doesn't match this project's chunker and, on investigation, doesn't reliably
  reconstruct from the raw corpus for long/frequently-edited articles (cross-referencing against
  `ner_gold.json`'s actual paragraph text found overlap for only 2 of 328 references — not enough
  for a paragraph-level metric). `retrieval_eval.py` therefore scores at **document level**
  (did the retriever surface a chunk from the correct source document) and reports embedding-score
  signals (`avg_top1_score`, `avg_hit_score`) from the same run — covering both C1 and C2 honestly,
  rather than inventing a second script around data that can't support it.
- **Retrieval (C1/C2) is strong:** Recall@5 = 0.9347, Recall@10 = 0.9648, MRR = 0.8877 across 199
  evaluable queries (1 of 200 fully excluded — its only gold document no longer exists in the
  corpus; not counted as a miss). The 7 genuine misses cluster on only 4 distinct source
  documents (each missed 2–3 times), not random scatter — likely chronology/timeline-style
  articles where a single ~500-word chunk blends several distinct facts. Not enough evidence to
  justify a global chunking change (the §3.3 semantic-chunking gate stays closed), but worth
  revisiting if those documents keep failing after more sources land.
- **RAG (C3):** citation recall 0.90, groundedness 3.89/5 (LLM-judge, 79% rated 4-5) on the
  baseline run (30-query evenly-spaced sample, `llama3.1:8b` standing in for the not-yet-pulled
  `qwen3:14b`). Citation **precision** was low (0.27) — the generator cites most/all retrieved
  passages rather than only the ones a claim actually draws from.
- **The Qwen3-14b-vs-llama3.1:8b comparison from §3.4 is done** (`qwen3:14b` was pulled and
  the identical 30-query sample re-run against it — `eval_reports/rag_v1_qwen3-14b.json`).
  Result: **citation precision was 0.2689 on both models — identical to four decimal places.**
  That's strong evidence the low precision isn't a per-model quality gap at all, it's a
  structural property of asking a model to cite from 5 retrieved passages when most queries
  have exactly 1 gold-relevant document: recall (0.90 both models) implies the correct doc
  gets cited ~90% of the time, but precision (0.27) implies each answer cites roughly 3–4 of
  the 5 retrieved sources on average, regardless of which model is generating. Groundedness was
  also comparable (`qwen3:14b` 3.83/5, 70% rated 4–5, vs `llama3.1:8b`'s 3.89/5 baseline — well
  within run-to-run variance at n=30). **Practical implication:** the real lever for citation
  precision is `retrieval.top_k` (fewer candidates → structurally higher precision, at some
  recall cost) or a differently-shaped citation metric, not model choice or further prompt
  wording — a second, independently-worded prompt attempt already failed to move this number
  (see C5 below), and this cross-model replication is why that failure is now trusted rather
  than attributed to one particular model being stubborn. Jais-30B/Falcon-H1-Arabic remain
  untested (not pulled) — re-run `eval/rag_eval.py --model <name>` if/when tried.
- **C5 tuning — a real attempt, a real (negative) result, now replicated:** the citation-precision
  finding was evidence for a specific, single change (tighten the system prompt to say "cite only
  what a claim actually draws from," not "cite everything you were given"). Applied it, re-ran the
  identical 30-query sample on `llama3.1:8b`: citation precision was **unchanged** (0.2689 → 0.2689
  exactly). The `qwen3:14b` run above (same tuned prompt, different model) landed on the *same*
  0.2689 again — ruling out "this one model just didn't listen" and confirming the metric is
  structurally floored by `top_k`, not fixable by prompt wording. No chunking/top-k change was
  attempted — C2's retrieval numbers don't warrant one, and changing top_k trades away recall to
  move a metric that was never really about model quality.
- **NER (C4) F1 is likely undercounted:** reading `eval_reports/ner_v1_mismatches.md` by hand,
  a meaningful share of "errors" are surface-form mismatches, not real misses — e.g. `والعراق` vs
  `العراق` (attached conjunction), `جمال عبد الناصر` vs `جمال عبدالناصر` (compound-name spacing),
  trailing commas caught in a span. The script's own "normalized" mode (diacritics/alef-insensitive)
  doesn't address these, so the true 0.4712 F1 understates real quality on `PERSON`/`LOCATION`.
  `ORGANIZATION` recall (0.17) is a genuine, separate weak spot, not an artifact.
- **Reproduce:** `uv run python -m eval.ner_eval --gold eval/gold/ner_gold.json`;
  `uv run python -m eval.retrieval_eval`; `uv run python -m eval.rag_eval` (uses
  `configs/rag.yaml`'s `qwen3:14b` by default now that it's pulled; pass `--model llama3.1:8b`
  to compare against the smaller model). All write to `eval_reports/` — pass `--output` to keep
  more than one run side by side instead of overwriting the previous one at the default path.

### Track D — Multi-source expansion (P1/P2) — ✅ D1/D2/D3/D4/D5/D6/D7 all done

`BaseCollector` (`src/ingestion/base_collector.py`) is already the right abstraction —
every source below is a new subclass, not a redesign. That held for all four collectors
built in this track (Wikipedia extensions aside): D4, D2, D3 were each a new subclass,
zero changes to the base interface.

**Posture change, mid-track (2026-07-26):** D2/D3 were originally deferred on licensing
grounds (see the "deliberately deferred" note this section used to carry). The project
owner then clarified the corpus is **private research use** — shared with at most one
teammate, not published or redistributed — and asked to prioritize source coverage and
diversity over licensing enforcement, with licensing revisited only if/when a public
release is considered. In response:

- `scripts/export_to_hf.py` **no longer gates on `license_status` by default** — it
  exports everything, and `license_status` is kept accurate on every document purely as
  provenance so a `--clear-only` public-release subset stays possible later without
  re-collecting anything. See `_is_clear()`/`export_to_hf_dataset(clear_only=...)`.
- D2 (GDELT) and D3 (WAFA) were un-deferred and implemented for real (below).
- D4 (Semantic Scholar) was extended to fetch full open-access paper text, not just
  abstracts, per the same "richer content for downstream NLP/RAG/KG" request.
- Two exceptions were **kept in place regardless of the private-use reframing**, because
  neither is actually a licensing/redistribution question: **Palestine Remembered** stays
  `blocked` — that's a refusal to bypass live Cloudflare bot-detection, not a copyright
  judgment, and holds no matter how the corpus is used. **Nakba Archive** stays
  un-collected pending a direct answer from the project owner — its blocker is informed
  consent for reusing survivor-testimony content, which private-use framing doesn't
  resolve on its own.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. | Status |
|---|---|:--:|---|---|---|:--:|---|
| D1 | Licensing/rights gate — a per-source checklist (redistribution rights, attribution requirements) that must pass before a collector's output reaches the published corpus. Wikipedia (CC-BY-SA) already clears it; news/archive sources generally do not by default | P1 | — | Parallel | `docs/licensing_checklist.md` + a `license_status` field check in the export step | 1d | Done — later reframed as provenance-only, not export-gating (see posture change above) |
| D2 | GDELT collector (config already stubbed, disabled) | P2 | D1 | Parallel across sources | `src/ingestion/collectors/gdelt_collector.py` | 2d | Done — 13 real docs collected |
| D3 | WAFA / archive-source collectors (config already stubbed, disabled) | P2 | D1 | Parallel across sources | `src/ingestion/collectors/*_collector.py` | 2d each | Done — WAFA implemented and verified; Nakba Archive/Palestine Remembered still held (see posture change above) |
| D4 | Semantic Scholar collector (academic sources) | P2 | D1 | Parallel across sources | `src/ingestion/collectors/semantic_scholar_collector.py` | 2d | Done — extended to full OA text; 25 real docs collected |
| D5 | Collection-time relevance gating — reuse the existing seed-audit script's `RELEVANCE_KEYWORDS` as a live accept filter, not just a post-hoc audit (addresses the topical-drift finding: Jordanian/Lebanese/Syrian/Israeli cuisine leaking into seed-category traversal) | P1 | — | Parallel | Updated `wikipedia_collector.py` filter | 0.5d | Done |
| D6 | Persistent / incremental deduplication — the LSH index is currently in-memory-per-run and outputs open in overwrite mode, so nothing dedups against the existing corpus. This becomes a real blocker once D2–D4 land, not before | P1 | D2 (or first new source) | Sequential, gates further expansion | Persisted `DuplicationIndex` (e.g. serialized MinHash bands in Postgres alongside `Chunk`) | 1.5d | Done — landed alongside D4, as its own gate predicted |
| D7 | English pipeline — `pipeline.py` currently hardcodes `ar`; `sources.yaml` already has an `en` config block ready | P2 | — | Parallel | Parameterized `pipeline.py --language en` | 1d | Done — real run, 15 English docs |

**Real results:**

- **D1 licensing findings, with evidence, not assumptions:** checked `robots.txt` for the
  three archive/community sources as a legitimate, non-scraping diligence step (reading a
  site's own published crawl policy). WAFA's `robots.txt` explicitly allows crawling.
  Nakba Archive: inconclusive technically, but the real blocker is ethical
  (survivor-testimony consent), not copyright — held pending the project owner's answer.
  **Palestine Remembered's `robots.txt` request returned an active Cloudflare bot-detection
  JavaScript challenge, not a policy** — a clear, current signal against automated access.
  Marked `blocked`; this project will not attempt to bypass it, regardless of corpus use.
  Full assessment: `docs/licensing_checklist.md`.
- **D2 GDELT — real, working, with two corrections found by actually running it.**
  (1) The originally-configured `sourcecountry:PS` query returned genuinely empty results
  in testing (not rate-limiting) — GDELT's `sourcecountry` operator appears not to carry a
  distinct Palestine entry in its source-monitoring list. Switched to plain keyword queries
  (`"Palestinian culture"`, `"Palestinian heritage"`, `"Gaza culture"`, `"West Bank
  culture"`), confirmed to return real data. (2) A live 15-doc run collected 13 accepted
  documents (9,087 words); `seed_category_distribution` showed publishing-outlet countries
  (Spain, Argentina, Egypt, ...) rather than "Palestine" — this looked like a bug at first
  glance but is correct, intended behavior (`seed_category` reports the article's
  *publishing outlet's* country, by design). Spot-checking the actual collected documents
  confirmed genuinely on-topic, high-value content this project's other sources would never
  surface: international coverage of a Madrid ministerial conference on Palestinian
  culture, UNESCO reporting on Gaza heritage-site damage, and Egyptian press coverage of
  Palestinian author Liana Badr's new novel. The collector also has a real safety
  mechanism — a per-domain `robots.txt` check (bounded-timeout fetch, "absent robots.txt =
  allowed" per convention) before scraping any GDELT-linked article — and degraded
  correctly on one domain during the real run (an SSL certificate failure, retried 3x, then
  skipped without crashing the batch).
- **D3 WAFA — real, working, extraction pattern verified against live pages.** Discovers
  articles via WAFA's own daily sitemap (`wafa.ps/sitemap.xml`), walking backward up to
  `max_days_back` days. Title/body/date/category extraction was verified against real
  fetched pages, not guessed once and shipped — the category extractor in particular went
  through a real fix: an initial guess at a breadcrumb HTML element didn't exist on the
  actual page (confirmed by a live run showing `{"unknown": 17}`), so it was rebuilt to
  parse the label directly out of the same content block already used for title/body
  (`"الرئيسية <category> تاريخ النشر: ..."`), then re-verified against live data
  (`{"انتهاكات إسرائيلية": 14, "محلية": 3}`).
- **D4 Semantic Scholar — extended to full open-access text, verified per-document, not
  just by aggregate stats.** Previously title+abstract only; now attempts to fetch and
  extract the paper's actual PDF text (`pdfplumber`) whenever Semantic Scholar reports an
  `openAccessPdf` URL, with a content-type check and a minimum-length threshold, falling
  back to the abstract if extraction fails or comes back too short. A live 25-doc run
  collected all 25 (49,786 words, avg. 1,991 words/doc — far above the ~130–200 word/doc
  average from abstract-only runs). Spot-checking individual documents confirmed **7 of 25
  got real full text** (e.g. a 3,413-word and a 5,044-word paper — clearly full papers, not
  abstracts) and **18 of 25 correctly fell back to abstract-only**, which is the expected
  outcome (not every OA-flagged paper has an extractable PDF). Two real failure modes were
  observed and handled correctly rather than crashing the run: one PDF URL actually served
  HTML (content-type check caught it), and one Wiley URL returned `403 Forbidden` (retried,
  then fell back). Still rate-limited on the shared unauthenticated API tier in practice —
  same finding as the original abstract-only run, unchanged by this extension.
- **D6 landed together with D4, exactly as this table's own dependency note predicted.**
  `PersistentDuplicationIndex` loads existing MinHash signatures from Postgres on startup
  and persists new ones back — verified for real with a cross-instance test (register a
  doc, discard the index object, construct a fresh one against the same connection,
  confirm it still catches the duplicate) and in production by every source's stats
  (`"persistent": true`, rows visible in `ingestion_dedup_index`). `pipeline.py`'s
  accepted/rejected JSONL outputs are append-mode, not overwrite — re-running (or adding a
  source) grows the corpus instead of replacing it. Degrades to in-memory-only with a
  logged warning if Postgres isn't reachable, so ingestion still works standalone.
- **D5's filter is precisely scoped, not a broad keyword ban:** targets categories that
  pair a generic culture/heritage word (مطبخ/عمارة/أدب/...) with an*other* country's name
  — the exact pattern behind the originally-observed drift (`مطبخ أردني`, `مواقع أثرية في
  إسرائيل`). A category naming Palestine is never excluded regardless of what else it also
  mentions, and broader ambiguous regional categories (`مطبخ عربي`, `عمارة إسلامية`) are
  deliberately left alone — the evidence supported excluding specific-other-country
  pairings, not general regional overlap.
- **D7:** most of the work turned out to already exist — `WikipediaCollector` already
  handled `language="en"` internally (category prefixes, `Language.ENGLISH` mapping);
  the only real gap was `pipeline.py` hardcoding `ar` and no CLI flag. Fixed as part of
  generalizing `pipeline.py` for D6 (see below) — `--language en` now works, verified with
  a real 15-document run into `data/processed/wikipedia_en_documents.jsonl` (a separate
  file from the Arabic corpus, confirmed the existing 484-doc Arabic corpus was untouched).
- **A structural side-effect worth naming:** generalizing `pipeline.py` for D4/D6 meant
  extracting `run_collection_pipeline()` — the quality/dedup/write/stats loop every source
  shares — out of what was a Wikipedia-only function, plus a `_SIMPLE_SOURCE_REGISTRY` +
  `run_simple_source_pipeline()` so each new source (Semantic Scholar, WAFA, GDELT) is a
  registry entry and a thin wrapper, not a new copy of the orchestration loop. This is the
  "avoid future rewrites" principle actually paying off across three real sources, not just
  a hoped-for property.
- **Additional-sources research (scoping pass, no new collector yet):**
  `palestine-studies.org` explicitly disallows AI-training/`ClaudeBot` crawling in its
  `robots.txt` — respected, not pursued. UNRWA's site returned the same kind of active
  bot-detection challenge as Palestine Remembered — held for the same reason. A guessed
  `libraries.birzeit.edu` domain failed to resolve at all (not a real candidate). **B'Tselem
  (`btselem.org`) is a genuinely viable next candidate** — permissive `robots.txt`, no
  AI-specific disallow — documented here but not yet implemented.

### Track E — Knowledge layer (P1/P2) — ✅ E1–E5 all done, real end-to-end run

Builds directly on the `KGEntity`/`KGRelation` shapes from §2, which are themselves an
aggregation of what `entity_extractor.py` already emits.

| ID | Task | Priority | Depends on | Mode | Deliverable | Est. | Status |
|---|---|:--:|---|---|---|:--:|---|
| E1 | Entity canonicalization — aggregate the existing per-document `entities` output into corpus-scope `KGEntity` records (dedup by normalized name + type) | P1 | A1, C4 (trust NER first) | Sequential | `src/knowlegde_graph/canonicalize.py` | 1.5d | Done — 11,565 entities from 483 docs |
| E2 | Entity linking to Wikidata (mGENRE, or a simpler alias-table approach first — see note) | P2 | E1 | Sequential | `wikidata_qid` populated on `KGEntity` | 2–3d | Done — 847/11,565 linked (7.3%), two real bugs found and fixed |
| E3 | Relation extraction (LLM-prompted, using the Track B/C generator) | P2 | E1 | Parallel to E2 | `src/knowlegde_graph/relations.py` → `KGRelation` records | 2d | Done — 216 relations from 20 docs |
| E4 | KG store — **NetworkX in-process for the first working prototype** (matches the project's own iterative-validation practice: prove the graph is useful before standing up infrastructure for it); **migrate to Neo4j Community** (free, self-hosted, mature Cypher for the multi-hop queries the dashboard's KG explorer will need) once E1–E3 are validated | P2 | E1, E3 | Sequential | Neo4j-backed KG, loader script | 2d | Done (NetworkX stage) — 11,565 nodes / 211 edges, GraphML |
| E5 | **KG eval** — precision of extracted relations against a hand-checked sample; entity-linking accuracy against a Wikidata gold sample | P1 | E2, E3 | Sequential after E2/E3 | `eval/kg_eval.py` → `EvalReport` | 1d | Done — relation precision 0.60, entity-linking 100% on checked sample |

**Real results, including two real bugs found and fixed by actually running this at scale:**

- **E1 canonicalization — real, straightforward, one honest caveat.** A live run over the
  483-document Arabic Wikipedia NER corpus produced 11,565 unique `(normalized, type)`
  entities (`data/entities/kg_entities.jsonl`): 4,685 PERSON, 2,971 LOCATION, 1,877
  ORGANIZATION, 1,862 MISC, ~170 across the five HERITAGE_* types. One inherited artifact,
  not introduced here: `فلسطين` (1,709 mentions) and `فلسطين،` (228 mentions, trailing
  Arabic comma) canonicalize as two *different* entities — a pre-existing CAMeL NER
  span-boundary quirk (the tagger occasionally includes trailing punctuation in a span)
  flowing through as designed, since E1 aggregates NER output as-is rather than
  reinterpreting it. Not fixed here (out of Track E's scope); E2's fuzzy linking tier
  happens to reunite the two under the same Wikidata QID regardless.
- **E2 entity linking — real, and genuinely instructive about a naive alias table's failure
  modes.** First implementation: exact match + a `difflib`-based fuzzy fallback (0.90
  threshold) against a live-fetched Wikidata SPARQL dump of Palestine-related items
  (`wdt:P17`/`P27`/`P495` = Q219060), 4,319 items / 1,561 with aliases
  (`data/entities/wikidata_palestine_aliases.jsonl`). Three real problems surfaced only by
  actually running it at this corpus's real scale, not by design review:
  1. **Performance:** a naive per-pair `difflib.SequenceMatcher` scan (11,565 entities ×
     ~7,700 alias strings) didn't finish in 5 minutes even after a provably-lossless
     length-window prune (difflib's `ratio()` is mathematically bounded by the two string
     lengths, so aliases outside a computed window can never reach the threshold — safe to
     skip). The real fix was a character-bigram inverted index for candidate generation
     (`_candidate_indices` in `entity_linking.py`) — an approximation, not lossless like the
     length prune, but the standard technique for approximate string matching at this scale.
     Full run: **~35 seconds** for all 11,565 entities.
  2. **Wrong-entity collisions:** spot-checking the actual linked QIDs against live Wikidata
     found this corpus's *three highest-mention entities* linked to the wrong item entirely:
     `فلسطين` (1,709 mentions) → Q12231001, a **newspaper** named "Felesteen", not the
     country; `غزة` → Q1395229, a **Wikimedia disambiguation page**; `حيفا` → Q17004991, a
     **1996 film** titled "Haifa". Root cause: a same-label homonym plus a naive
     first-exact-match-wins collision policy. Fixed by excluding a small, evidence-based set
     of Wikidata classes (disambiguation pages, newspapers, films, books, ...) from the index
     — see `_EXCLUDED_INSTANCE_OF` in `entity_linking.py`.
  3. **Missing anchor entity:** `فلسطين` (Palestine) itself was never in its own alias dump —
     the SPARQL query only fetches items whose `P17`/`P27`/`P495` *points at* Q219060, and
     Q219060 never points at itself. Fixed by fetching the anchor item's own label/aliases as
     a fourth, separate query and merging it in (`wikidata_aliases.py`).
  After both fixes, re-verified directly against live Wikidata: `فلسطين` → Q219060 (State of
  Palestine, correct), `القدس` → Q1218 (Jerusalem, correct), `غزة` → Q39760 (Gaza Strip,
  correct), plus Hebron/Nablus/Ramallah/West Bank/Hamas/Bethlehem/Ahmed Yassin/dabke all
  independently verified correct. Final real run: **847/11,565 linked (7.3%)** — 439 exact,
  408 fuzzy. The fuzzy tier's real value was directly demonstrated: `فلسطين،` (the trailing-
  comma NER artifact from E1's own caveat above) fuzzy-matched to the *same* QID as `فلسطين`
  at score 0.923, correctly reuniting what E1 couldn't. **A real, named scope limitation, not
  a bug:** `اسرائيل`/`الاردن` (1,210 / 315 mentions) correctly abstain — out of scope by
  design, since the alias dump only contains Palestine-anchored items. More interestingly,
  `حيفا`/`يافا` also correctly abstain post-fix, for a different and more informative reason:
  both are real, historically Palestinian cities, but Wikidata's `P17` (country) reflects
  *current* sovereignty (Israel), not historical/cultural identity — a strict
  `P17`=Palestine query structurally cannot surface pre-1948 Palestinian population centers
  now administered by Israel. Closing that gap needs a supplementary, manually curated place
  list — not attempted here, named as a concrete next step rather than silently missed.
- **E3 relation extraction — real, and one real bug (not a design flaw) found by running it.**
  Sentence-level entity co-occurrence (capped at 4 distinct entities/sentence, i.e. ≤6 pairs)
  prompts `qwen3:14b` via the same Ollama config as the RAG generator (`configs/rag.yaml`) for
  a strict-JSON predicate + confidence, keeping predicates ≥0.5 confidence. First real attempt
  against the live model returned **empty content on every call** — `qwen3:14b` is a hybrid
  "thinking" model that emits a `<think>...</think>` block before its answer unless told not
  to, and the original 100-token budget was entirely consumed by that block. Fixed by passing
  Ollama's `think=False` for this call, which also cut per-call latency from ~7s (empty
  output) to ~1.4s (a real answer). Real run: 20 documents → **216 relations**, 456.77s
  (~2.1s/kept relation). Predicate distribution is genuinely diverse, not degenerate —
  `located_in` (93), `related_to` (17), `born_in` (11), `capital_of`, `signed_agreement_with`,
  `wrote_about`, and 50 more distinct predicates, mostly at count 1–4.
- **E5 KG eval — real, hand-built gold sets, honest numbers, not hidden behind aggregate
  stats.** Relation extraction: a random sample of 40 real extracted relations
  (`eval/gold/kg_relations_gold.json`) hand-checked against their `evidence_sentence` —
  **precision 0.60 (24/40)**. `located_in` (the dominant predicate, 19 of the 40 sampled) is
  the weak point at 9/19 (0.47), and the failures cluster into concrete, fixable patterns
  rather than random noise: (a) **backward relations** — subject/object swapped (e.g. "West
  Bank located_in Jerusalem", "Israel located_in Africa" — the latter a clear hallucination,
  not a swap); (b) **ternary "between X and Y" relations collapsed into one binary pair**,
  losing the second endpoint; (c) **truncated entity spans** (stray Arabic proclitics/commas
  inherited from NER boundaries) making an otherwise-correct relation hard to read on its own;
  (d) a few outright **wrong predicate choices** (e.g. "part_of" for a film screened *at* a
  festival, not part of it). Entity linking: **100% accuracy on the 13 checked linkable
  entities and 100% correct-abstention rate on 5 checked out-of-scope entities**
  (`eval/gold/kg_entity_linking_gold.json`) — expected and somewhat circular, since this gold
  set directly re-verifies the two bugs found and fixed above, not a claim that linking is
  perfect on unseen entities.
- **E4 graph store — real, small, honestly fragmented.** `build_graph()` produced **11,565
  nodes / 211 edges** (216 extracted relations minus 5 that collapsed onto identical
  `relation_id`s — expected `MultiDiGraph` behavior, not data loss) from the linked entities
  and the 20-document relation-extraction run, persisted as GraphML
  (`data/graph/kg_graph.graphml`). **11,420 of 11,565 nodes are isolated or nearly so**
  (weakly-connected components ≈ node count) — an honest, expected consequence of E3 having
  processed only 20 of 483 documents so far, not a flaw in the graph-building logic itself.
  Top-degree nodes are exactly what a Palestinian cultural KG should surface first: `فلسطين`
  (degree 25), `القدس`, `اسرائيل`, `غزة`, `الضفة الغربية`. Scaling E3 to the full corpus is
  the direct, obvious next step to make the graph genuinely connected and useful — a config
  change (`--max-docs`), not a redesign.
- **Multi-source extension — real, and one real correctness fix.** The numbers above were
  from Arabic Wikipedia alone (the only source with NER already run on it at the time). Once
  WAFA/GDELT/Semantic Scholar/English Wikipedia were collected (Track D), extending the KG to
  cover them exposed a real bug rather than being a config change: `scripts/run_ner.py` was
  blindly feeding every document to CAMeL Tools NER regardless of language. Directly testing
  this against real English Wikipedia text confirmed CAMeL's NER model is **Arabic-only** —
  it tagged "Israel" as PERSON and section headers like "Overview" as entities. Fixed by
  adding a language guard (`_ARABIC_LANGUAGES`) that skips non-Arabic documents (tagging them
  `ner_skipped_reason: "non_arabic_language"` rather than silently mis-tagging them), applied
  per-document so mixed-language sources (GDELT, Semantic Scholar) get partial, correct
  coverage rather than an all-or-nothing source-level decision. Real language split found:
  WAFA 106/106 Arabic (fully usable), GDELT 24/73 Arabic (49 skipped — mostly non-Palestinian
  outlets in Spanish/other, consistent with Track D's own finding that GDELT surfaces
  international coverage), Semantic Scholar 1/25 Arabic (academic papers are overwhelmingly
  English), English Wikipedia 0/97 (entirely skipped — not NER'd at all for the KG).
  `scripts/extract_kg_relations.py` was also extended to accept multiple `--input` files, with
  `--max-docs` applying **per file** rather than to the combined total — otherwise Arabic
  Wikipedia's much larger file (581 docs) would starve every other source of any documents at
  all when files are processed in sequence. A real combined run (15 docs/source cap, 4
  sources) produced **189 relations from 46 documents** spanning all three Arabic-capable
  sources (Wikipedia AR, WAFA, GDELT, one Semantic Scholar paper) — E1's entity count grew to
  **13,063** (711 documents) and E2's linked count to **950 (7.27%)**, both re-verified to hold
  the same correct QIDs as the single-source run.
- **Reproduce, in order (multi-source):**
  ```powershell
  uv run python scripts/run_ner.py                                                    # refresh AR Wikipedia NER
  uv run python scripts/run_ner.py --input data/processed/wafa_documents.jsonl --output data/processed/wafa_documents.ner.jsonl
  uv run python scripts/run_ner.py --input data/processed/gdelt_documents.jsonl --output data/processed/gdelt_documents.ner.jsonl
  uv run python scripts/run_ner.py --input data/processed/semantic_scholar_documents.jsonl --output data/processed/semantic_scholar_documents.ner.jsonl
  uv run python scripts/build_kg_entities.py --input data/processed/wikipedia_ar_documents.ner.jsonl `
      data/processed/wafa_documents.ner.jsonl data/processed/gdelt_documents.ner.jsonl data/processed/semantic_scholar_documents.ner.jsonl   # E1
  uv run python scripts/fetch_wikidata_aliases.py --limit 4000       # E2 (fetch dump — occasional, not per-run)
  uv run python scripts/link_kg_entities.py                         # E2 (link)
  uv run python scripts/extract_kg_relations.py `
      --input data/processed/wikipedia_ar_documents.ner.jsonl data/processed/wafa_documents.ner.jsonl `
              data/processed/gdelt_documents.ner.jsonl data/processed/semantic_scholar_documents.ner.jsonl `
      --max-docs 15 --max-pairs-per-doc 15                          # E3 (needs Ollama running; --max-docs is PER FILE)
  uv run python scripts/build_kg_graph.py                            # E4
  uv run python -m eval.kg_eval                                      # E5 (gold sets are a fixed hand-checked snapshot,
                                                                       #     not re-derived from the live relations file)
  ```

*Note on E2: started with a cheap alias-table linker (exact/fuzzy match against a pre-pulled
Wikidata Palestine-entity SPARQL dump) before reaching for mGENRE's full model — consistent
with "validate cheap first," and the real bugs found above (performance, wrong-entity
collisions, missing anchor entity) were each fixable within that cheap approach. Escalate to
mGENRE only if E5's precision, re-measured on a larger sample later, doesn't hold up.*

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
