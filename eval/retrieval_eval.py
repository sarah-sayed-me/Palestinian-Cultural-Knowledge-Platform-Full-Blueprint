"""
eval/retrieval_eval.py — evaluate the real retriever (real pgvector index, real
Qwen3-Embedding-0.6B) against a hand-built query -> relevant-document gold set.
Covers both ROADMAP.md Track C1 (embedding-quality signal) and C2 (Recall@k, MRR)
from one run, since both need the same query/gold-doc pairs.

Granularity note (read before trusting a paragraph-level number from this data):
the gold set's `relevant_para_ids` use a `<doc_id>_p<N>` PARAGRAPH numbering that
does not match this project's chunker (recursive/overlapping, different
boundaries) and, on spot-checking, does not reliably reconstruct even from the
raw corpus text for long/frequently-edited articles (see git history / PR
discussion). Cross-referencing against eval/gold/ner_gold.json's text_id field
(the only source of exact paragraph text) found overlap for only 2 of 328
references — not enough to support a paragraph-level metric. This script
therefore evaluates at DOCUMENT level: does the retriever surface a chunk from
the correct source document, not the exact paragraph. That is the honestly
supportable granularity for this gold set; do not over-interpret document-level
recall as chunk-level recall.

Usage:
    python -m eval.retrieval_eval \
        --queries eval/gold/retrieval_queries.json \
        --output eval_reports/retrieval_v1.json \
        --misses eval_reports/retrieval_v1_misses.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.schemas import EvalReport  # noqa: E402
from src.rag.config import RagConfig  # noqa: E402
from src.rag.db import get_connection  # noqa: E402
from src.rag.embedder import Embedder  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402

DEFAULT_QUERIES = Path("eval/gold/retrieval_queries.json")
K_VALUES = (5, 10)
MAX_K = max(K_VALUES)


def doc_id_of(para_id: str) -> str:
    return para_id.rsplit("_p", 1)[0]


def load_indexed_doc_ids(conn) -> set[str]:
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT doc_id FROM rag_chunks;")
        return {row[0] for row in cur.fetchall()}


def evaluate_query(
    query: dict[str, Any], retriever: Retriever, indexed_doc_ids: set[str]
) -> dict[str, Any]:
    gold_doc_ids = {doc_id_of(pid) for pid in query["relevant_para_ids"]}
    resolvable = gold_doc_ids & indexed_doc_ids
    unresolvable = gold_doc_ids - indexed_doc_ids

    result: dict[str, Any] = {
        "query_id": query["query_id"],
        "query_type": query["query_type"],
        "gold_doc_ids": sorted(gold_doc_ids),
        "unresolvable_doc_ids": sorted(unresolvable),
        "evaluable": bool(resolvable),
    }
    if not resolvable:
        return result

    retrieved = retriever.retrieve(query["query"], top_k=MAX_K)
    result["top1_score"] = retrieved[0].score if retrieved else None

    hit_rank = None
    hit_score = None
    distinct_gold_found: set[str] = set()
    for rank, item in enumerate(retrieved, start=1):
        if item.chunk.doc_id in resolvable:
            distinct_gold_found.add(item.chunk.doc_id)
            if hit_rank is None:
                hit_rank = rank
                hit_score = item.score

    result["hit_rank"] = hit_rank
    result["hit_score"] = hit_score
    result["reciprocal_rank"] = (1.0 / hit_rank) if hit_rank else 0.0
    result["distinct_gold_docs_found"] = len(distinct_gold_found)
    result["distinct_gold_docs_total"] = len(resolvable)
    for k in K_VALUES:
        result[f"hit_at_{k}"] = bool(hit_rank and hit_rank <= k)
    return result


def aggregate(results: list[dict[str, Any]]) -> dict[str, float]:
    evaluable = [r for r in results if r["evaluable"]]
    metrics: dict[str, float] = {"evaluable_queries": len(evaluable), "skipped_queries": len(results) - len(evaluable)}
    if not evaluable:
        return metrics

    metrics["mrr"] = round(sum(r["reciprocal_rank"] for r in evaluable) / len(evaluable), 4)
    for k in K_VALUES:
        hits = sum(1 for r in evaluable if r[f"hit_at_{k}"])
        metrics[f"recall_at_{k}"] = round(hits / len(evaluable), 4)

    top1_scores = [r["top1_score"] for r in evaluable if r["top1_score"] is not None]
    if top1_scores:
        metrics["avg_top1_score"] = round(sum(top1_scores) / len(top1_scores), 4)
    hit_scores = [r["hit_score"] for r in evaluable if r.get("hit_score") is not None]
    if hit_scores:
        metrics["avg_hit_score"] = round(sum(hit_scores) / len(hit_scores), 4)

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--output", type=Path, default=Path("eval_reports/retrieval_v1.json"))
    parser.add_argument("--misses", type=Path, default=Path("eval_reports/retrieval_v1_misses.md"))
    args = parser.parse_args()

    with args.queries.open(encoding="utf-8") as f:
        queries = json.load(f)

    config = RagConfig.load(args.config)
    embedder = Embedder(config.embedding)
    conn = get_connection()
    try:
        indexed_doc_ids = load_indexed_doc_ids(conn)
        retriever = Retriever(conn, embedder, config)

        results = [evaluate_query(q, retriever, indexed_doc_ids) for q in queries]
    finally:
        conn.close()

    overall = aggregate(results)
    by_type: dict[str, dict[str, float]] = {}
    for qtype in sorted({r["query_type"] for r in results}):
        by_type[qtype] = aggregate([r for r in results if r["query_type"] == qtype])

    drift_affected = [r for r in results if r["unresolvable_doc_ids"]]
    fully_unevaluable = [r for r in results if not r["evaluable"]]
    misses = [r for r in results if r["evaluable"] and r.get("hit_rank") is None]

    report = EvalReport(
        eval_name="retrieval_v1",
        dataset_size=len(queries),
        metrics=overall,
        notes=(
            "Document-level Recall@k/MRR against the real pgvector retriever — see module "
            "docstring for why paragraph-level scoring isn't supported by this gold set. "
            f"{len(drift_affected)} queries reference at least one doc_id no longer in the "
            f"indexed corpus (corpus drift since the gold set was built); "
            f"{len(fully_unevaluable)} of those have NO resolvable gold doc and were excluded "
            "from the metrics entirely (not counted as misses)."
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump(
            {
                "report": report.to_json_dict(),
                "by_query_type": by_type,
                "corpus_drift": {
                    "queries_with_unresolvable_docs": len(drift_affected),
                    "queries_fully_unevaluable": len(fully_unevaluable),
                    "unevaluable_query_ids": [r["query_id"] for r in fully_unevaluable],
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    args.misses.parent.mkdir(parents=True, exist_ok=True)
    with args.misses.open("w", encoding="utf-8") as f:
        f.write("# Retrieval misses (evaluable queries where no relevant document was found in top-{})\n\n".format(MAX_K))
        f.write(f"{len(misses)} of {len(results) - len(fully_unevaluable)} evaluable queries missed entirely.\n\n---\n\n")
        for r in misses:
            q = next(q for q in queries if q["query_id"] == r["query_id"])
            f.write(f"- [{r['query_id']}] \"{q['query']}\" — gold doc_ids: {r['gold_doc_ids']}\n")

    print(json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2))
    print("\nBy query_type:")
    for qtype, m in by_type.items():
        print(f"  {qtype}: {m}")
    print(f"\nCorpus drift: {len(drift_affected)} queries affected, {len(fully_unevaluable)} fully unevaluable")
    print(f"Saved report: {args.output}")
    print(f"Saved misses: {args.misses}")


if __name__ == "__main__":
    main()
