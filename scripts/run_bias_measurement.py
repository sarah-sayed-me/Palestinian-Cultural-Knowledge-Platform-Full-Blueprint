"""Bias measurement over the corpus (Track F3): category-distribution comparison
across sources, a WEAT-style embedding association test, and an LLM framing probe.

Usage:
    uv run python scripts/run_bias_measurement.py
    uv run python scripts/run_bias_measurement.py --skip-framing-probe   # skip the Ollama-dependent part

Requires the corpus to already have `category` labels from
scripts/run_content_classification.py for the category-distribution part.
The WEAT test needs the local embedding model (no Postgres/Ollama needed).
The framing probe needs Ollama running (same model as configs/rag.yaml).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Iterator, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.nlp.bias_measurement import (
    OllamaFramingProbe,
    category_distribution_by_source,
    total_variation_distance,
    weat_effect_size,
)
from src.rag.config import RagConfig
from src.rag.embedder import Embedder

DEFAULT_INPUTS = [
    Path("data/processed/wikipedia_ar_documents.categorized.jsonl"),
    Path("data/processed/wafa_documents.categorized.jsonl"),
    Path("data/processed/gdelt_documents.categorized.jsonl"),
]
DEFAULT_OUTPUT = Path("reports/bias_measurement.json")

# A starting term set for the WEAT test — Palestine-culture terms vs.
# Palestine-conflict terms as the two TARGET sets, against conflict-coded
# vs. culture/peace-coded ATTRIBUTE terms. Not an exhaustive or validated
# psycholinguistic word list — a first, inspectable pass; see ROADMAP.md for
# the honest caveats on WEAT's methodological fragility (R4).
TARGET_SET_CULTURE = ["تراث", "فولكلور", "تطريز", "دبكة", "مطبخ", "حرفة", "موسيقى", "أدب"]
TARGET_SET_CONFLICT = ["احتلال", "مقاومة", "اشتباك", "حصار", "غارة", "اعتقال", "استيطان", "انتفاضة"]
ATTRIBUTE_SET_NEGATIVE = ["عنف", "دمار", "قتل", "حرب", "خطر", "معاناة"]
ATTRIBUTE_SET_POSITIVE = ["سلام", "جمال", "فرح", "أمل", "ازدهار", "تعاون"]


def _read_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _run_weat(embedder: Embedder) -> dict:
    def embed_terms(terms: List[str]):
        import numpy as np

        return [np.array(v) for v in embedder.embed_documents(terms)]

    target_x = embed_terms(TARGET_SET_CULTURE)
    target_y = embed_terms(TARGET_SET_CONFLICT)
    attr_a = embed_terms(ATTRIBUTE_SET_NEGATIVE)
    attr_b = embed_terms(ATTRIBUTE_SET_POSITIVE)

    effect_size, associations = weat_effect_size(target_x, target_y, attr_a, attr_b)
    return {
        "target_set_culture_terms": TARGET_SET_CULTURE,
        "target_set_conflict_terms": TARGET_SET_CONFLICT,
        "attribute_set_negative_terms": ATTRIBUTE_SET_NEGATIVE,
        "attribute_set_positive_terms": ATTRIBUTE_SET_POSITIVE,
        "effect_size": effect_size,
        "interpretation": (
            "Positive effect_size: culture terms associate more with the NEGATIVE attribute set "
            "than conflict terms do (surprising, worth investigating). Negative effect_size: "
            "conflict terms associate more with the NEGATIVE attribute set (the intuitive "
            "direction). Magnitude near 0: no strong differential association found."
        ),
        "per_term_associations": associations,
    }


def _run_framing_probe(documents: List[dict], *, max_docs_per_source: int, model: str | None) -> dict:
    from collections import Counter, defaultdict
    from dataclasses import replace

    from src.rag.config import RagConfig as _RagConfig

    generation_config = _RagConfig.load().generation
    if model:
        generation_config = replace(generation_config, model=model)
    probe = OllamaFramingProbe(config=generation_config)

    per_source_counts: Dict[str, Counter] = defaultdict(Counter)
    per_source_seen: Dict[str, int] = defaultdict(int)
    for doc in documents:
        source_id = doc.get("source_id", "unknown")
        if per_source_seen[source_id] >= max_docs_per_source:
            continue
        per_source_seen[source_id] += 1
        result = probe.probe(doc.get("text", ""))
        if result is not None:
            framing, _ = result
            per_source_counts[source_id][framing] += 1

    return {
        source_id: dict(counts) for source_id, counts in per_source_counts.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure bias signals across the corpus.")
    parser.add_argument("--input", type=Path, nargs="+", default=DEFAULT_INPUTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--skip-framing-probe", action="store_true", help="Skip the Ollama-dependent LLM probe.")
    parser.add_argument("--framing-max-docs-per-source", type=int, default=10)
    parser.add_argument("--model", type=str, default=None)
    args = parser.parse_args()

    existing_inputs = [p for p in args.input if p.exists()]
    if not existing_inputs:
        raise FileNotFoundError(
            f"None of the input files exist: {args.input}. Run scripts/run_content_classification.py first "
            "(it produces the *.categorized.jsonl files this script reads by default)."
        )

    documents = [doc for path in existing_inputs for doc in _read_jsonl(path)]

    distributions = category_distribution_by_source(documents)
    source_ids = list(distributions)
    divergences = {}
    for i in range(len(source_ids)):
        for j in range(i + 1, len(source_ids)):
            a, b = source_ids[i], source_ids[j]
            divergences[f"{a}_vs_{b}"] = total_variation_distance(distributions[a], distributions[b])

    config = RagConfig.load()
    embedder = Embedder(config.embedding)
    weat_result = _run_weat(embedder)

    framing_result = None
    if not args.skip_framing_probe:
        framing_result = _run_framing_probe(
            documents, max_docs_per_source=args.framing_max_docs_per_source, model=args.model
        )

    report = {
        "inputs": [str(p) for p in existing_inputs],
        "total_documents": len(documents),
        "documents_with_category": sum(1 for d in documents if d.get("category")),
        "category_distribution_by_source": distributions,
        "category_distribution_divergence": divergences,
        "weat": weat_result,
        "llm_framing_probe_by_source": framing_result,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, ensure_ascii=False, indent=2)

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
