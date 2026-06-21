"""Phase 1 ingestion pipeline orchestration."""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import yaml

from src.ingestion.collectors.wikipedia_collector import WikipediaCollector
from src.ingestion.deduplication import DuplicationIndex
from src.ingestion.quality_checker import check_document
from src.ingestion.schemas import QualityDecision

DEFAULT_SOURCES_CONFIG = Path("configs/sources.yaml")
DEFAULT_QUALITY_CONFIG = Path("configs/quality_thresholds.yaml")
DEFAULT_ACCEPTED_PATH = Path("data/processed/wikipedia_ar_documents.jsonl")
DEFAULT_REJECTED_PATH = Path("data/metadata/wikipedia_ar_rejected.jsonl")
DEFAULT_STATS_PATH = Path("data/metadata/wikipedia_ar_stats.json")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def run_wikipedia_arabic_pipeline(
    *,
    max_docs: int = 100,
    sources_path: Path = DEFAULT_SOURCES_CONFIG,
    quality_path: Path = DEFAULT_QUALITY_CONFIG,
    accepted_path: Path = DEFAULT_ACCEPTED_PATH,
    rejected_path: Path = DEFAULT_REJECTED_PATH,
    stats_path: Path = DEFAULT_STATS_PATH,
) -> dict[str, Any]:
    """Run the Arabic Wikipedia collection, quality, dedup, and storage path."""
    started = time.time()
    sources_config = load_yaml(sources_path)
    quality_config = load_yaml(quality_path)
    wikipedia_config = sources_config.get("wikipedia", {})
    if not wikipedia_config.get("enabled", False):
        raise RuntimeError("Wikipedia collection is disabled in configs/sources.yaml")

    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    dedup_config = quality_config.get("deduplication", {})
    dedup_index = DuplicationIndex(
        threshold=float(dedup_config.get("threshold", 0.80)),
        num_perm=int(dedup_config.get("num_perm", 128)),
    )
    collector = WikipediaCollector(
        language="ar",
        source_config=wikipedia_config,
        credibility_map=sources_config.get("credibility_map", {}),
        max_docs=max_docs,
        request_delay=float(wikipedia_config.get("rate_limit_delay", 1.0)),
    )

    stats: dict[str, Any] = {
        "source_id": "wikipedia-ar",
        "attempted_documents": 0,
        "accepted_documents": 0,
        "rejected_documents": 0,
        "duplicate_documents": 0,
        "total_words": 0,
        "average_document_length": 0.0,
        "category_distribution": {},
        "quality_decision_distribution": {},
        "deduplication": {},
        "outputs": {
            "accepted_jsonl": str(accepted_path),
            "rejected_jsonl": str(rejected_path),
            "stats_json": str(stats_path),
        },
    }
    categories: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    rejection_reasons: Counter[str] = Counter()
    seed_categories_counter: Counter[str] = Counter()

    with accepted_path.open("w", encoding="utf-8") as accepted, rejected_path.open(
        "w", encoding="utf-8"
    ) as rejected:
        for document in collector.collect():
            stats["attempted_documents"] += 1
            quality = check_document(document)
            document.quality_score = quality.quality_score
            document.quality_decision = quality.decision
            document.rejection_reason = quality.rejection_reason
            decisions[quality.decision.value] += 1

            if quality.decision in {QualityDecision.REJECT, QualityDecision.HARD_REJECT}:
                stats["rejected_documents"] += 1
                rejection_reasons[quality.rejection_reason or "unknown"] += 1
                _write_jsonl(rejected, document.model_dump(mode="json"))
                continue

            duplicate = dedup_index.check_and_register(document.doc_id, document.text)
            if duplicate.is_duplicate:
                document.has_duplicate = True
                document.is_duplicate_of = duplicate.canonical_id
                document.rejection_reason = "duplicate"
                stats["duplicate_documents"] += 1
                stats["rejected_documents"] += 1
                rejection_reasons["duplicate"] += 1
                _write_jsonl(rejected, document.model_dump(mode="json"))
                continue

            stats["accepted_documents"] += 1
            stats["total_words"] += document.word_count
            categories.update(document.wikipedia_categories)
            seed_categories_counter.update([document.seed_category or "unknown"])
            _write_jsonl(accepted, document.model_dump(mode="json"))

    accepted_count = stats["accepted_documents"]
    if accepted_count:
        stats["average_document_length"] = round(stats["total_words"] / accepted_count, 2)
    stats["category_distribution"] = dict(categories.most_common(50))
    stats["quality_decision_distribution"] = dict(decisions)
    stats["rejection_reason_distribution"] = dict(rejection_reasons.most_common(50))
    stats["seed_category_distribution"] = dict(seed_categories_counter.most_common(50))
    stats["deduplication"] = dedup_index.stats()
    stats["duration_seconds"] = round(time.time() - started, 2)

    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(stats, handle, ensure_ascii=False, indent=2)
    return stats


def _write_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Phase 1 Arabic Wikipedia ingestion.")
    parser.add_argument("--max-docs", type=int, default=100)
    args = parser.parse_args()
    stats = run_wikipedia_arabic_pipeline(max_docs=args.max_docs)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
