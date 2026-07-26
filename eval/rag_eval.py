"""
eval/rag_eval.py — end-to-end RAG evaluation (ROADMAP.md Track C3): runs the
real RAGPipeline (real retriever + real generator) over a sample of the gold
queries and measures two things a retrieval-only eval can't:

  1. Citation-level precision/recall — of the documents the GENERATOR actually
     chose to cite, how many are truly relevant (precision) and how many of
     the relevant documents got cited (recall)? This can diverge from raw
     retrieval recall: retrieval can surface the right document while the
     generator still doesn't cite it, or cites an irrelevant one instead.
  2. Groundedness — an LLM-judge pass asking whether the generated answer's
     claims are actually supported by the retrieved passages, independent of
     whether the citations happen to point at the right documents. A model
     can cite the correct source and still say something the source doesn't
     support.

No gold ANSWERS exist for this query set (see eval/retrieval_eval.py's
docstring on the granularity limits of the gold set) — this script does not
attempt answer-correctness scoring, only groundedness and citation accuracy,
which are honestly supportable without gold answers.

Usage:
    python -m eval.rag_eval \
        --queries eval/gold/retrieval_queries.json \
        --sample 30 \
        --model llama3.1:8b \
        --output eval_reports/rag_v1.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.retrieval_eval import doc_id_of, load_indexed_doc_ids  # noqa: E402
from eval.schemas import EvalReport  # noqa: E402
from src.rag.config import RagConfig  # noqa: E402
from src.rag.db import get_connection  # noqa: E402
from src.rag.embedder import Embedder  # noqa: E402
from src.rag.generator import OllamaGenerator  # noqa: E402
from src.rag.pipeline import RAGPipeline  # noqa: E402
from src.rag.retriever import Retriever  # noqa: E402

DEFAULT_QUERIES = Path("eval/gold/retrieval_queries.json")

JUDGE_PROMPT = """You are a strict fact-checker. Given a question, a set of numbered \
source passages, and a generated answer, decide whether every factual claim in the \
answer is actually supported by the passages (not outside knowledge, not invented).

Question: {question}

Sources:
{sources}

Answer to check:
{answer}

Respond with ONLY a single digit 1-5:
5 = fully grounded, every claim traceable to a source
4 = grounded with minor unsupported detail
3 = partially grounded, some claims unsupported
2 = mostly unsupported
1 = not grounded / contradicts the sources"""

_DIGIT_RE = re.compile(r"[1-5]")


def evenly_spaced_sample(items: list[Any], n: int) -> list[Any]:
    if n >= len(items):
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def judge_groundedness(client, judge_model: str, question: str, answer_text: str, context_chunks) -> int | None:
    sources = "\n\n".join(
        f"[{i}] {c.chunk.title or '(untitled)'}: {c.chunk.text}" for i, c in enumerate(context_chunks, start=1)
    )
    prompt = JUDGE_PROMPT.format(question=question, sources=sources, answer=answer_text)
    response = client.chat(model=judge_model, messages=[{"role": "user", "content": prompt}])
    match = _DIGIT_RE.search(response.message.content or "")
    return int(match.group()) if match else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--config", type=Path, default=Path("configs/rag.yaml"))
    parser.add_argument("--sample", type=int, default=30, help="Evenly-spaced sample size (LLM calls are the slow part)")
    parser.add_argument("--model", help="Override configs/rag.yaml's generation.model for this run")
    parser.add_argument("--no-judge", action="store_true", help="Skip the LLM-judge groundedness pass (citation metrics only)")
    parser.add_argument("--output", type=Path, default=Path("eval_reports/rag_v1.json"))
    args = parser.parse_args()

    with args.queries.open(encoding="utf-8") as f:
        all_queries = json.load(f)
    queries = evenly_spaced_sample(all_queries, args.sample)

    config = RagConfig.load(args.config)
    if args.model:
        config = replace(config, generation=replace(config.generation, model=args.model))

    embedder = Embedder(config.embedding)
    conn = get_connection()
    try:
        indexed_doc_ids = load_indexed_doc_ids(conn)
        retriever = Retriever(conn, embedder, config)
        generator = OllamaGenerator(config.generation)
        pipeline = RAGPipeline(retriever, generator, config)

        per_query: list[dict[str, Any]] = []
        for q in queries:
            gold_doc_ids = {doc_id_of(pid) for pid in q["relevant_para_ids"]}
            resolvable_gold = gold_doc_ids & indexed_doc_ids
            if not resolvable_gold:
                continue

            retrieved = retriever.retrieve(q["query"], top_k=config.retrieval.top_k)
            if not retrieved:
                per_query.append(
                    {"query_id": q["query_id"], "insufficient_context": True, "citation_precision": None,
                     "citation_recall": None, "groundedness": None}
                )
                continue

            generated_text = generator.generate(q["query"], retrieved)
            from src.rag.answer import assemble_answer

            answer = assemble_answer(generated_text, retrieved)
            cited_doc_ids = {c.doc_id for c in answer.citations}

            precision = len(cited_doc_ids & resolvable_gold) / len(cited_doc_ids) if cited_doc_ids else 0.0
            recall = len(cited_doc_ids & resolvable_gold) / len(resolvable_gold)

            groundedness = None
            if not args.no_judge:
                groundedness = judge_groundedness(
                    generator.client, config.generation.model, q["query"], answer.text, retrieved
                )

            per_query.append(
                {
                    "query_id": q["query_id"],
                    "insufficient_context": False,
                    "citation_precision": round(precision, 4),
                    "citation_recall": round(recall, 4),
                    "groundedness": groundedness,
                }
            )
    finally:
        conn.close()

    scored = [r for r in per_query if not r["insufficient_context"]]
    metrics: dict[str, float] = {
        "sampled_queries": len(queries),
        "scored_queries": len(scored),
        "insufficient_context_count": len(per_query) - len(scored),
    }
    if scored:
        metrics["avg_citation_precision"] = round(sum(r["citation_precision"] for r in scored) / len(scored), 4)
        metrics["avg_citation_recall"] = round(sum(r["citation_recall"] for r in scored) / len(scored), 4)
        grounded_scores = [r["groundedness"] for r in scored if r["groundedness"] is not None]
        if grounded_scores:
            metrics["avg_groundedness_1to5"] = round(sum(grounded_scores) / len(grounded_scores), 4)
            metrics["groundedness_4_or_5_rate"] = round(sum(1 for s in grounded_scores if s >= 4) / len(grounded_scores), 4)

    report = EvalReport(
        eval_name="rag_v1",
        dataset_size=len(all_queries),
        metrics=metrics,
        notes=(
            f"Evenly-spaced sample of {len(queries)}/{len(all_queries)} gold queries, run through the "
            f"real end-to-end pipeline (model={config.generation.model}). Citation precision/recall "
            "compares Answer.citations' doc_ids to gold-relevant doc_ids. Groundedness is an LLM-judge "
            "1-5 score (5=fully grounded); no gold answers exist for this query set, so answer-content "
            "correctness itself is not scored, only citation accuracy and groundedness — see module docstring."
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"report": report.to_json_dict(), "per_query": per_query}, f, ensure_ascii=False, indent=2)

    print(json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2))
    print(f"\nSaved report: {args.output}")


if __name__ == "__main__":
    main()
