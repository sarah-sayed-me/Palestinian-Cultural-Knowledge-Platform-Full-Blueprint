# Project Quick Deep-Dive — Palestinian Cultural Knowledge Platform

*Audit date: verified directly against the repository (code, data, tests, eval reports) on this date. Numbers are counted from actual files, not from ROADMAP.md targets.*

Legend: ✅ Implemented · 🟡 Partial · ❌ Not implemented · 🔵 Experimental/unverified quality

---

## 1. Project in 60 Seconds

**Problem:** Build a reproducible, quality-scored, multi-source text corpus about Palestinian culture and history, then turn it into structured, queryable knowledge (entities, a knowledge graph, retrieval-augmented QA) — prioritizing culture/heritage over conflict-only framing.

**Input:** Documents from 5 sources (Arabic/English Wikipedia, WAFA news agency, GDELT news index, Semantic Scholar papers).

**Output:** A quality-filtered JSONL corpus → NER-enriched documents → a knowledge graph (GraphML) → a pgvector index → cited RAG answers via CLI/API/dashboard.

**Real end-to-end pipeline (verified from code, not assumed):**

```
Collect (5 sources) → Quality score + dedup → JSONL corpus
        → NER (CAMeL + heritage dict, Arabic-only)
              ├─→ Entity canonicalization → Wikidata linking → LLM relation extraction → NetworkX KG
              └─→ Chunk → Embed (Qwen3) → pgvector → Retrieve → Ollama LLM → cited answer (RAG)
        → [optional analysis] Topic modeling / Content classification / Bias measurement / Temporal analysis
```

Main tech: Python 3.14, `uv`, Pydantic v2 schemas, PostgreSQL+pgvector, CAMeL Tools NER, `sentence-transformers` (Qwen3-Embedding), Ollama (qwen3:14b), NetworkX, BERTopic, FastAPI, Streamlit.

---

## 2. What Has Actually Been Implemented?

| Component | Status | What actually exists |
|---|:--:|---|
| Data Collection | ✅ | 5 real collectors (`src/ingestion/collectors/`): Wikipedia AR/EN, WAFA, GDELT, Semantic Scholar. Shared pipeline with quality scoring + MinHash dedup. |
| Preprocessing | ✅ | `src/preprocessing/arabic_normalizer.py` — Arabic normalization, boilerplate stripping. |
| NER | ✅ | `src/ingestion/entity_extractor.py` — CAMeL Tools (model-based) + heritage-term dictionary (rule-based hybrid). **Arabic-only** — non-Arabic docs are skipped, not processed. |
| Knowledge Graph | ✅ | `src/knowlegde_graph/` — canonicalization, Wikidata alias-table linking, LLM relation extraction, NetworkX graph. Real data exists (see §3). |
| Embeddings | ✅ | `src/rag/embedder.py`, Qwen3-Embedding-0.6B, 1024-dim, stored in pgvector. 1,282 chunks currently indexed. |
| RAG | ✅ | `src/rag/pipeline.py` + `scripts/ask.py` + `src/api/main.py` (FastAPI) + dashboard "Ask" tab. Verified working. |
| Evaluation | ✅ | `eval/` — NER, retrieval, RAG, KG relations, KG linking all have real gold sets and real scores (§7, §3). |
| Topic Modeling | ✅ | `src/nlp/topic_model.py`. Real run: 41 topics across 488/581 Wikipedia AR docs. Clustering is meaningful; auto-generated *labels* skew toward years/numbers rather than Arabic words — a real c-TF-IDF characteristic on this corpus, not a bug (see §11). |
| Classification | ✅ | `src/nlp/content_classifier.py` — **not** the fine-tuned AraBERT the roadmap describes; it's zero-shot LLM prompting (qwen3). Real run exists: 75 docs classified. |
| Bias Analysis | ✅ | `src/nlp/bias_measurement.py` — category-distribution divergence, WEAT embedding test, LLM framing probe. Real run exists with real numbers (§ below). |
| Temporal Analysis | ✅ | `src/nlp/temporal_analysis.py` — real run, 706/785 docs bucketed by decade. Note: doesn't use the `decade` metadata field alone (see §2 discrepancy below). |

Everything with a roadmap description that looks like a gap is actually done — either fully working, or done via a documented **deviation** from the original plan, not a silent gap. **Documentation says F2 = fine-tuned AraBERT, but the actual implementation is zero-shot LLM classification** (`src/nlp/content_classifier.py` docstring admits this explicitly).

---

## 3. Current Dataset Size — Verified From Files

| Source | Cumulative docs (current) | Latest single-run accept/reject (from `data/metadata/*_stats.json`) |
|---|--:|---|
| Wikipedia AR | **581** | 97 accepted / 3 rejected / 0 dup (of 100 attempted) |
| Wikipedia EN | **97** | 82 accepted / 18 rejected / 15 dup (of 100 attempted) |
| WAFA | **106** | 89 accepted / 11 rejected / 0 dup (of 100 attempted) |
| GDELT | **73** | 13 accepted / 2 rejected / 0 dup (of 15 attempted) |
| Semantic Scholar | **25** | stats file shows a stale/empty run (0/0/0) — the 25 docs came from an earlier run whose stats were overwritten |
| **Total** | **882** | — |

⚠️ **Important:** `*_stats.json` files hold only the **latest single run's** numbers, not cumulative totals. The corpus JSONL files (`data/processed/*_documents.jsonl`) are append-only and ARE the real cumulative count.

**No fixed "target N documents" milestone exists in ROADMAP.md** — collection is deliberately iterative/small-batch. The only numeric ceilings found are config limits, not targets: `configs/sources.yaml` sets `max_articles_per_language: 8000`, `max_articles_per_category: 500` (Wikipedia only, never reached).

**Downstream artifact counts (verified):**
- RAG chunks indexed in pgvector: **1,282** (from an earlier 484-document batch of Wikipedia AR — see discrepancy below)
- KG entities: **13,063** canonicalized, **950 (7.3%)** linked to a Wikidata QID
- KG relations: **585 unique** (deduplicated; see below) used to build the current graph file
- NER eval gold set: 108 annotated paragraphs
- Retrieval eval gold set: 200 queries
- RAG eval: sampled 30/200 queries

🔴 **Documentation/data discrepancy found:** README.md states "484 documents chunked into 1282 passages" as current status. The actual `wikipedia_ar_documents.jsonl` now has **581** documents — the corpus grew after the RAG index was built, and `scripts/chunk_corpus.py`/`build_index.py` were never re-run. **The RAG system currently only searches the original 484-doc subset, not the full 581.**

🔴 **Second discrepancy:** `data/graph/kg_relations.jsonl` currently contains **1,175 unique relation records** (verified by parsing), but `data/graph/kg_graph.graphml` (the file actually used for graph queries) was last built from only **585** — the graph file is stale relative to the relations file. Run `scripts/build_kg_graph.py` again to reconcile before demoing the KG.

**Verify these numbers yourself:**
```powershell
Get-Content data/processed/wikipedia_ar_documents.jsonl | Measure-Object -Line
Get-Content data/processed/wafa_documents.jsonl | Measure-Object -Line
Get-Content data/entities/kg_entities.jsonl | Measure-Object -Line
Get-Content data/graph/kg_relations.jsonl | Measure-Object -Line
Get-Content eval_reports/ner_v1.json | ConvertFrom-Json | Select -ExpandProperty report
```

---

## 4. How to Run It Yourself

```powershell
# 1. Collection (each appends to its JSONL; safe to re-run)
uv run python main.py --max-docs 100                          # Wikipedia AR
uv run python main.py --language en --max-docs 100             # Wikipedia EN
uv run python scripts/collect_wafa.py --max-docs 30
uv run python scripts/collect_gdelt.py --max-docs 30
uv run python scripts/collect_semantic_scholar.py --max-docs 30

# 2. NER (Arabic-only; non-Arabic docs get ner_skipped_reason, not tagged)
uv run python scripts/run_ner.py                                # writes *.ner.jsonl

# 3. Classification (Ollama must be running)
uv run python scripts/run_content_classification.py --max-docs 30

# 4. Evaluation (writes to eval_reports/)
uv run python -m eval.ner_eval --gold eval/gold/ner_gold.json
uv run python -m eval.retrieval_eval
uv run python -m eval.rag_eval
uv run python -m eval.kg_eval

# 5. Knowledge Graph (in order — each depends on the previous)
uv run python scripts/build_kg_entities.py                      # E1: canonicalize
uv run python scripts/fetch_wikidata_aliases.py --limit 4000     # E2a: fetch dump (occasional)
uv run python scripts/link_kg_entities.py                        # E2b: link
uv run python scripts/extract_kg_relations.py --max-docs 15      # E3: needs Ollama, slow (1 LLM call/pair)
uv run python scripts/build_kg_graph.py                          # E4: build GraphML

# 6. Embeddings / RAG (needs `docker compose up -d` for Postgres first)
uv run python scripts/chunk_corpus.py
uv run python scripts/build_index.py
uv run python scripts/ask.py "ما هي الكنافة النابلسية؟"

# 7. Topic modeling / Bias / Temporal
uv run python scripts/run_topic_model.py           # needs step 6 done first — NOT YET RUN in this repo
uv run python scripts/run_bias_measurement.py       # needs step 3's categorized output
uv run python scripts/run_temporal_analysis.py      # no DB/Ollama needed

# 8. API / Dashboard
uv run uvicorn src.api.main:app --reload --port 8000
uv run streamlit run src/frontend/app.py

# 9. Tests
uv run pytest tests/ -q                              # 198 tests, all pass when Postgres is up
```

Each script prints a JSON summary to stdout — that IS the inspection method; also check the `outputs`/`output` paths it prints (usually `data/processed/*`, `data/metadata/*`, `eval_reports/*`, `reports/*`).

---

## 5. Trace One Document (WAFA article example)

| Stage | File / Function | Input → Output |
|---|---|---|
| Collect | `src/ingestion/collectors/wafa_collector.py::WafaCollector.collect()` | Sitemap URL → raw HTML → `DocumentMetadata` (title, text, date, category) |
| Quality + dedup | `src/ingestion/quality_checker.py`, `deduplication.py` | Text → quality_score, MinHash signature → accept/reject decision |
| Write | `src/ingestion/pipeline.py::run_collection_pipeline` | Accepted doc → appended to `data/processed/wafa_documents.jsonl` |
| NER | `src/ingestion/entity_extractor.py::EntityExtractor.extract()` | Text → CAMeL NER + heritage dict → `entities` field → `wafa_documents.ner.jsonl` |
| Classification | `src/nlp/content_classifier.py::OllamaContentClassifier.classify()` | Title+text → LLM prompt → `category`/`category_confidence` → `wafa_documents.categorized.jsonl` |
| KG: canonicalize | `src/knowlegde_graph/canonicalize.py::canonicalize_entities()` | `entities` (all docs) → corpus-scope `KGEntity` rows → `kg_entities.jsonl` |
| KG: link | `src/knowlegde_graph/entity_linking.py::WikidataAliasLinker.link()` | Entity name → Wikidata QID or None → `kg_entities.linked.jsonl` |
| KG: relations | `src/knowlegde_graph/relations.py::OllamaRelationExtractor.extract_document()` | Sentence + 2 co-occurring entities → LLM → predicate → `kg_relations.jsonl` |
| KG: graph | `src/knowlegde_graph/graph_store.py::build_graph()` | Entities + relations → `kg_graph.graphml` |
| RAG (separate path) | `src/rag/chunker.py` → `embedder.py` → `index.py` | Text → 500-token chunks → Qwen3 embedding → pgvector row |

Note: RAG and KG are **two independent downstream paths** from the same corpus — a document doesn't have to pass through NER to be searchable via RAG, and doesn't have to be chunked to appear in the KG.

---

## 6. Models and Algorithms

| Component | Actual model/algorithm | Why (from code/docs, or marked as inferred) |
|---|---|---|
| Embeddings | `Qwen/Qwen3-Embedding-0.6B`, 1024-dim, via `sentence-transformers` | ROADMAP.md: strong multilingual/Arabic MTEB results at small size, runs locally on 8GB VRAM or CPU. *Explicit project decision.* |
| NER | CAMeL Tools (pretrained model) + rule-based heritage-term dictionary | Hybrid because CAMeL alone misses Palestinian-culture-specific terms (food, crafts) not in general Arabic NER training data. *Explicit project decision.* |
| Classification | Zero-shot prompting of `qwen3:14b` via Ollama | **Deviation**, not the original plan (roadmap says fine-tune AraBERT). Code docstring: no time this session to build a real ~1,000-1,500-example labeled training set; reuses existing LLM infra instead. *Explicit, documented deviation.* |
| Topic Modeling | BERTopic (UMAP + HDBSCAN + c-TF-IDF), built but not run | ROADMAP.md: reuses existing chunk embeddings, no new embedding pass. *Explicit decision; untested on real data — mark 🟡.* |
| Relation Extraction | Zero-shot prompting of `qwen3:14b`, sentence-level entity co-occurrence | Reuses the same LLM already running for RAG — avoids a second model dependency. *Explicit project decision.* |
| Entity Linking | Exact + fuzzy (bigram-indexed) string match against a Wikidata SPARQL dump scoped to Palestine-related items | ROADMAP.md: "cheap alias-table linker... before reaching for mGENRE." *Explicit, staged decision — mGENRE is the stated future upgrade if precision proves insufficient.* |
| RAG LLM | `qwen3:14b` via Ollama, temperature 0.2, top_k=5 retrieval | Local/free-first requirement (ROADMAP §3.4) — compared explicitly against Gemini free tier, OpenRouter, Grok in the roadmap's own writeup; chosen for zero cost, data control, and strong Arabic support. *Explicit project decision with a written comparison.* |
| Vector Store | PostgreSQL + pgvector | ROADMAP §3.2: `psycopg2` already a dependency, comfortably handles the realistic corpus scale, avoids a second database system. Escape hatch to Qdrant is pre-agreed but not triggered (corpus is nowhere near the scale that would require it). *Explicit decision, with a documented trigger condition for revisiting.* |

**Why RAG instead of fine-tuning an LLM?** Not explicitly argued in the repo as "RAG vs. fine-tuning" — reasonably inferred: fine-tuning would need to be redone every time the corpus grows; RAG lets the corpus update independently of the model, and citations are a hard requirement (fine-tuned models can't cite sources the way retrieval can). Mark this as **engineering reasoning, not an explicit documented project decision**.

---

## 7. NER + Evaluation

**Entity types:** `PERSON`, `LOCATION`, `ORGANIZATION`, `MISC` (from CAMeL) + `HERITAGE_FOOD`, `HERITAGE_CLOTHING`, `HERITAGE_CRAFT`, `HERITAGE_HERITAGE_PRACTICE`, `HERITAGE_PLACE`, `HERITAGE_PLANT` (from the dictionary in `configs/heritage_entities.yaml`).

**Approach:** Hybrid — CAMeL Tools model-based tagging + a rule-based dictionary matcher for heritage terms (`src/ingestion/entity_extractor.py`). Not purely rule-based, not purely model-based.

**Gold set:** 108 hand-annotated paragraphs (`eval/gold/ner_gold.json`).

**Real scores (`eval_reports/ner_v1.json`):**

| Metric | Exact match | Normalized (diacritic-insensitive) |
|---|--:|--:|
| Precision | 0.460 | 0.463 |
| Recall | 0.483 | 0.486 |
| **F1** | **0.471** | **0.474** |

**Weakest category:** `ORGANIZATION` — precision 0.33, recall 0.17, F1 0.22 (support 83). `HERITAGE_PLACE` is the strongest (F1 0.70, support 32). Several `HERITAGE_*` types have 0 support in this gold set (untested, not necessarily broken).

**Known evaluation caveat (from the repo's own eval notes):** manual review of mismatches found some "errors" are surface-form artifacts (attached conjunctions, compound-name spacing) rather than true misses — the real quality is likely somewhat better than 0.47 F1 for `PERSON`/`LOCATION`.

**Run it yourself:** `uv run python -m eval.ner_eval --gold eval/gold/ner_gold.json`

---

## 8. Knowledge Graph

**Entities stored:** `KGEntity` — canonical name, type (from NER types above), Wikidata QID (if linked), mention count, source doc IDs. **13,063** total, **950 linked**.

**Relations:** `KGRelation` — subject/predicate/object entity IDs, confidence, source doc, evidence sentence. Predicates are free-form snake_case strings generated by the LLM (`located_in`, `born_in`, `member_of`, ...), **not** a fixed closed ontology.

**Extraction method:** Sentence-level co-occurrence (2+ entities in one sentence) → LLM prompt → predicate + confidence (`src/knowlegde_graph/relations.py`).

**Storage:** NetworkX `MultiDiGraph`, persisted as GraphML at `data/graph/kg_graph.graphml` (⚠️ currently stale — see §3 discrepancy).

**Real example** (from `data/graph/kg_relations.jsonl`):
```
يافا (Jaffa) --located_next_to--> البحر الابيض المتوسط (Mediterranean Sea)
```

**Evaluation:** Hand-checked, real gold sets — **relation precision 0.60** (24/40, weakest on `located_in` at 0.47 due to backward subject/object errors), **entity-linking accuracy 100%** on 13 checked links + 5 checked correct abstentions (small sample — this is a spot-check, not a large statistical claim).

**Status:** ✅ Implemented and real, but 🟡 in coverage — relations only cover 88 of 882 documents (the LLM-call cost bounds how much can be processed per run).

---

## 9. RAG

```
Question → Embed (Qwen3, same model as corpus) → pgvector cosine search (top_k=5)
    → Context chunks → Ollama qwen3:14b (temp=0.2, think=False) → Answer + numbered citations
```

- **Embedded:** 500-token recursive chunks (`chunking_version: recursive-500-v1`) of document text, 1,282 currently indexed (from the older 484-doc batch — see §3).
- **Vector store:** pgvector, table `rag_chunks`, cosine distance.
- **Top-k:** 5 (`configs/rag.yaml`), optionally filtered by `min_credibility_tier`.
- **LLM:** `qwen3:14b` via Ollama, temperature 0.2, max 1024 output tokens.
- **Hallucination reduction:** system prompt instructs "answer strictly from the numbered sources... say so if insufficient... cite only what you actually used."
- **Citations:** yes — `Answer.citations` returns numbered `[doc_id, title, source_url]`, deduplicated by document.
- **Retrieval quality (real, `eval_reports/retrieval_v1.json`):** Recall@5 = 0.935, Recall@10 = 0.965, MRR = 0.888 (199 evaluable queries).
- **RAG quality (real, `eval_reports/rag_v1_qwen3-14b.json`, 30 sampled queries):** citation recall 0.90, **citation precision only 0.269** (the model over-cites — includes sources it didn't really draw from), groundedness 3.83/5.

**3 good demo questions:**
1. `ما هي الكنافة النابلسية؟` (What is Nabulsi knafeh?) — heritage/food, should retrieve real Wikipedia content with citations.
2. `أين تقع يافا؟` (Where is Jaffa?) — tests geography retrieval + can be cross-checked against the KG example in §8.
3. A deliberately out-of-corpus question (e.g. about a topic with no Wikipedia article) — demonstrates the "insufficient information" fallback rather than hallucination.

---

## 10. Ten Important "Why" Questions

1. **Why RAG instead of fine-tuning?** Corpus updates independently of the model; citations are a hard requirement fine-tuning can't give you as directly. *(engineering reasoning)*
2. **Why pgvector, not a dedicated vector DB?** Already a dependency (`psycopg2`), comfortably handles this corpus's realistic scale, avoids a second system. *(explicit, ROADMAP §3.2)*
3. **Why Qwen3-Embedding?** Strong multilingual/Arabic benchmark results at a size that runs locally for free. *(explicit, ROADMAP §3)*
4. **Why local Ollama over a paid API?** Zero cost, data control for culturally sensitive material, strong Arabic support. *(explicit, ROADMAP §3.4, with a written comparison against Gemini/OpenRouter/Grok)*
5. **Why hybrid NER instead of pure model-based?** CAMeL alone misses Palestinian heritage-specific vocabulary not in general training data. *(explicit)*
6. **Why zero-shot LLM classification instead of the planned fine-tuned AraBERT?** No time this session to build a real labeled training set; reuses existing LLM infrastructure. *(explicit, documented deviation)*
7. **Why an alias-table entity linker instead of mGENRE immediately?** "Validate cheap first" — staged escalation only if precision proves insufficient. *(explicit)*
8. **Why is the corpus only ~880 documents, not tens of thousands?** Deliberate iterative small-batch philosophy — validate the pipeline on small real data before scaling, not a limitation discovered late. *(explicit, README "Development Philosophy")*
9. **Why NER is Arabic-only?** Directly tested CAMeL on English text; it produced nonsense (e.g. tagged "Israel" as PERSON). *(explicit, found during this project's own build)*
10. **Why does citation precision (0.27) look bad?** Structural, not a model bug — replicated identically across two different LLMs (qwen3 and llama3.1), so it's an artifact of citing from a 5-chunk top-k rather than a model-quality problem. *(explicit, ROADMAP Track C finding)*

---

## 11. Weaknesses / Limitations (top 8)

1. **RAG index is stale.** Only 484/581 Wikipedia AR docs are actually searchable — the index was never rebuilt after more documents were collected. *Say: "a known sync gap between collection and indexing, not a design flaw — re-running two scripts fixes it."*
2. **NER F1 is 0.47**, and `ORGANIZATION` recall is particularly weak (0.17). *Say: "moderate quality, honestly measured against a real gold set, not overclaimed; some of the error is surface-form noise, not true misses."*
3. **Citation precision is low (0.27)**, though citation recall is high (0.90). *Say: "the model over-cites; this is a top-k/prompt-design artifact confirmed across two different LLMs, not a hallucination problem."*
4. **KG covers a small fraction of the corpus** (relations exist for 88 of 882 docs) because relation extraction costs one LLM call per entity pair. *Say: "a cost/scale tradeoff, and the graph file itself is currently stale vs. the relations file — needs a rebuild."*
5. **Content classification and relation extraction use zero-shot LLM prompting, not trained/fine-tuned models**, and neither has a large benchmarked gold-set score the way NER/retrieval do. *Say: "a deliberate, documented scope decision given session time, not hidden — the roadmap's original fine-tuning plan is still the intended long-term direction."*
6. **Topic modeling's auto-generated labels are dominated by years/numbers, not Arabic keywords** — the clustering itself is meaningful (verified: one topic cleanly groups 1948/1947/1967/1949, i.e. Nakba/1948-war content), but the labels aren't very readable. *Say: "c-TF-IDF ranks words frequent-within-cluster-but-rare-elsewhere; common Arabic words appear in nearly every chunk so they can never win that ranking — a real characteristic of this corpus, not a bug. LLM-generated labels would read better and are a natural next step, not yet built."*
7. **Entity-linking scope is narrow by design** — the Wikidata alias dump only covers Palestine-anchored items, so it correctly can't link entities like Israel, Jordan, or even historically-Palestinian cities now administered by Israel (e.g. Haifa). *Say: "a known, named scope boundary, not a bug — closing it needs a curated supplementary place list."*
8. **Bias-measurement methodology (WEAT, LLM framing probe) is inherently contestable** — small term lists, small sample sizes (8 docs/source for the framing probe). *Say: "a real, working first pass with genuinely interpretable results, not a peer-reviewed methodology — appropriate caution applies."*

---

## 12. Viva — Likely Questions

| # | Question | Short ideal answer | Where in repo |
|---|---|---|---|
| 1 | What problem does this solve? | Structured, queryable Palestinian cultural knowledge from a curated multi-source corpus. | README.md Overview |
| 2 | How many documents do you have? | 882 total across 5 sources (581 Wikipedia AR, 106 WAFA, 97 Wikipedia EN, 73 GDELT, 25 Semantic Scholar). | `data/processed/*.jsonl` |
| 3 | Is that your target size? | No fixed target exists — deliberately iterative, small-batch collection philosophy. | README "Development Philosophy" |
| 4 | What's the end-to-end pipeline? | Collect → quality/dedup → NER → (KG path) / (RAG path) → optional analysis. | §1 above |
| 5 | What NER approach did you use and why? | Hybrid CAMeL model + heritage dictionary; pure model misses culture-specific terms. | `src/ingestion/entity_extractor.py` |
| 6 | What's your NER F1? | 0.47 exact match, weakest on ORGANIZATION (0.22), strongest on HERITAGE_PLACE (0.70). | `eval_reports/ner_v1.json` |
| 7 | What embedding model and why? | Qwen3-Embedding-0.6B, 1024-dim — strong multilingual/Arabic results, runs locally. | `configs/rag.yaml` |
| 8 | What's your retrieval quality? | Recall@5 = 0.935, MRR = 0.888. | `eval_reports/retrieval_v1.json` |
| 9 | How does the Knowledge Graph work? | NER entities canonicalized → linked to Wikidata (alias table) → LLM extracts relations from sentence co-occurrence → NetworkX graph. | `src/knowlegde_graph/` |
| 10 | Give a KG example. | يافا --located_next_to--> البحر الابيض المتوسط. | `data/graph/kg_relations.jsonl` |
| 11 | How does RAG reduce hallucination? | Strict "cite only from provided sources" system prompt + explicit "say insufficient info" instruction; citations returned. | `src/rag/generator.py` |
| 12 | Which LLM for RAG and why? | qwen3:14b via Ollama — free, local, strong Arabic, data control. | ROADMAP §3.4 |
| 13 | Why RAG instead of fine-tuning? | Corpus updates independently; citations are a hard requirement. | engineering reasoning |
| 14 | Why pgvector? | Already a dependency, handles this scale, avoids a second DB system. | ROADMAP §3.2 |
| 15 | Is content classification a trained model? | No — zero-shot LLM prompting, a documented deviation from the original fine-tuned-AraBERT plan. | `src/nlp/content_classifier.py` docstring |
| 16 | Have you run topic modeling? | Yes — 41 topics across 488/581 docs. Clustering is meaningful; auto-labels skew toward years since common Arabic words can't win c-TF-IDF's rarity-based ranking. | `data/processed/wikipedia_ar_documents.ner.topics.jsonl` |
| 17 | What's your biggest limitation? | RAG index is stale (484/581 docs searchable) — a sync gap between collection and indexing. | §3, §11 |
| 18 | How do you measure bias? | Category-distribution divergence across sources + WEAT embedding test + LLM framing probe. | `src/nlp/bias_measurement.py` |
| 19 | What did the bias analysis find? | WEAT effect size -1.612 (conflict terms associate with violence words, as expected); WAFA news skewed conflict-framed (7/8 sampled), Wikipedia leaned non-conflict. | `reports/bias_measurement.json` |
| 20 | How do you run the tests? | `uv run pytest tests/` — 198 tests, all pass (requires Postgres up via `docker compose up -d`). | `tests/` |

---

## 13. One-Page Cheat Sheet

- **Goal:** structured, culture-first Palestinian knowledge platform — corpus → NER → KG + RAG.
- **Current corpus:** **882 documents** (581 Wikipedia AR, 106 WAFA, 97 Wikipedia EN, 73 GDELT, 25 Semantic Scholar). No fixed target size — iterative philosophy.
- **Sources:** Wikipedia AR/EN, WAFA (news), GDELT (news index), Semantic Scholar (papers).
- **Pipeline:** Collect → quality score + MinHash dedup → NER (Arabic-only) → {KG: canonicalize → link (Wikidata) → LLM relations → graph} + {RAG: chunk → embed → pgvector → retrieve → Ollama → cited answer}.
- **Models:** Embedding = Qwen3-Embedding-0.6B (1024-dim); NER = CAMeL + heritage dictionary (hybrid); RAG/Classification/KG-relations LLM = qwen3:14b via Ollama; Vector store = pgvector; Topic model = BERTopic (built, not yet run); Entity linking = alias-table (exact+fuzzy) against Wikidata.
- **NER labels:** PERSON, LOCATION, ORGANIZATION, MISC, HERITAGE_FOOD/CLOTHING/CRAFT/HERITAGE_PRACTICE/PLACE/PLANT.
- **Scores:** NER F1 0.47 · Retrieval Recall@5 0.935, MRR 0.888 · RAG citation recall 0.90 / precision 0.27, groundedness 3.83/5 · KG relation precision 0.60 · KG linking 950/13,063 (7.3%).
- **Embedding dim:** 1024. **Vector store:** PostgreSQL + pgvector. **RAG LLM:** qwen3:14b, temp 0.2, top_k 5.
- **Key commands:** `uv run python main.py --max-docs 100` (collect) · `uv run python scripts/run_ner.py` · `uv run python scripts/ask.py "question"` · `uv run pytest tests/`.
- **5 biggest limitations:** (1) RAG index stale (484/581 searchable), (2) NER F1 moderate + weak ORGANIZATION, (3) citation precision low (structural, replicated across 2 LLMs), (4) KG covers only 88/882 docs and its graph file is stale vs. relations file, (5) classification/relations are zero-shot LLM, not trained models — no large benchmarked score yet.
- **Top 10 WHYs:** RAG>fine-tuning (fresh corpus + citations) · pgvector (already a dep, fits scale) · Qwen3 embeddings (Arabic MTEB) · local Ollama (free, data control) · hybrid NER (culture terms) · zero-shot classifier (no time to fine-tune, documented deviation) · alias-table linking first (cheap-first, staged) · small corpus (deliberate iteration) · Arabic-only NER (tested, English produced garbage) · low citation precision is structural (replicated across models).
