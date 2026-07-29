"""Ingestion pipeline orchestration — collector-agnostic since Track D.

run_collection_pipeline() holds the part every source shares (quality check,
dedup, write accepted/rejected, stats) so adding a new source is a new
collector + a thin wrapper function (see run_wikipedia_arabic_pipeline,
run_semantic_scholar_pipeline below), not a duplicated orchestration loop.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from pathlib import Path
from typing import Any, Optional

import yaml

from src.ingestion.base_collector import BaseCollector
from src.ingestion.collectors.gdelt_collector import GdeltCollector
from src.ingestion.collectors.semantic_scholar_collector import SemanticScholarCollector
from src.ingestion.collectors.wafa_collector import WafaCollector
from src.ingestion.collectors.wikipedia_collector import WikipediaCollector
from src.ingestion.deduplication import DuplicationIndex, PersistentDuplicationIndex
from src.ingestion.quality_checker import QualityConfig, check_document
from src.ingestion.schemas import QualityDecision

DEFAULT_SOURCES_CONFIG = Path("configs/sources.yaml")
DEFAULT_QUALITY_CONFIG = Path("configs/quality_thresholds.yaml")
DEFAULT_ACCEPTED_PATH = Path("data/processed/wikipedia_ar_documents.jsonl")
DEFAULT_REJECTED_PATH = Path("data/metadata/wikipedia_ar_rejected.jsonl")
DEFAULT_STATS_PATH = Path("data/metadata/wikipedia_ar_stats.json")


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def _build_dedup_index(threshold: float, num_perm: int) -> DuplicationIndex:
    """PersistentDuplicationIndex when Postgres is reachable, else a plain
    in-memory DuplicationIndex with a clear warning — ingestion still runs
    without Postgres, just without cross-run dedup (see ROADMAP.md Track D6).
    """
    try:
        from src.rag.db import get_connection

        conn = get_connection()
        return PersistentDuplicationIndex(threshold=threshold, num_perm=num_perm, conn=conn)
    except Exception as exc:
        import logging

        logging.getLogger(__name__).warning(
            "Postgres not reachable (%s) — deduplication will be in-memory-only for "
            "this run, and will NOT catch duplicates against previously-collected "
            "documents. Run `docker compose up -d` for persistent cross-run dedup.",
            exc,
        )
        return DuplicationIndex(threshold=threshold, num_perm=num_perm)


def run_collection_pipeline(
    *,
    collector: BaseCollector,
    source_label: str,
    accepted_path: Path,
    rejected_path: Path,
    stats_path: Path,
    quality_thresholds: QualityConfig,
    dedup_index: DuplicationIndex,
) -> dict[str, Any]:
    """Run any BaseCollector through quality checking, dedup, and storage.

    accepted_path/rejected_path are opened in APPEND mode: re-running (or
    running a second source into the same files) grows the corpus instead of
    overwriting it. stats_path reflects only THIS run's activity, not the
    cumulative corpus — the JSONL files are the cumulative state.
    """
    started = time.time()
    accepted_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.parent.mkdir(parents=True, exist_ok=True)

    stats: dict[str, Any] = {
        "source_id": source_label,
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

    with accepted_path.open("a", encoding="utf-8") as accepted, rejected_path.open(
        "a", encoding="utf-8"
    ) as rejected:
        for document in collector.collect():
            stats["attempted_documents"] += 1
            quality = check_document(document, quality_thresholds)
            document.quality_score = quality.quality_score
            document.quality_decision = quality.decision
            document.rejection_reason = quality.rejection_reason
            decisions[quality.decision.value] += 1

            if quality.decision in {QualityDecision.REJECT, QualityDecision.HARD_REJECT}:
                stats["rejected_documents"] += 1
                rejection_reasons[quality.rejection_reason or "unknown"] += 1
                _write_jsonl(rejected, document.model_dump(mode="json"))
                continue

            duplicate = dedup_index.check_and_register(
                document.doc_id, document.text, source_id=document.source_id
            )
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


def run_wikipedia_arabic_pipeline(
    *,
    max_docs: int = 100,
    language: str = "ar",
    sources_path: Path = DEFAULT_SOURCES_CONFIG,
    quality_path: Path = DEFAULT_QUALITY_CONFIG,
    accepted_path: Optional[Path] = None,
    rejected_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run the Wikipedia collection, quality, dedup, and storage path for one language."""
    sources_config = load_yaml(sources_path)
    wikipedia_config = sources_config.get("wikipedia", {})
    if not wikipedia_config.get("enabled", False):
        raise RuntimeError("Wikipedia collection is disabled in configs/sources.yaml")

    accepted_path = accepted_path or Path(f"data/processed/wikipedia_{language}_documents.jsonl")
    rejected_path = rejected_path or Path(f"data/metadata/wikipedia_{language}_rejected.jsonl")
    stats_path = stats_path or Path(f"data/metadata/wikipedia_{language}_stats.json")

    quality_config = load_yaml(quality_path)
    dedup_config = quality_config.get("deduplication", {})
    dedup_index = _build_dedup_index(
        threshold=float(dedup_config.get("threshold", 0.80)),
        num_perm=int(dedup_config.get("num_perm", 128)),
    )
    quality_thresholds = QualityConfig.load(quality_path)
    collector = WikipediaCollector(
        language=language,
        source_config=wikipedia_config,
        credibility_map=sources_config.get("credibility_map", {}),
        max_docs=max_docs,
        request_delay=float(wikipedia_config.get("rate_limit_delay", 1.0)),
        output_dir=f"data/raw/wikipedia/{language}",
    )

    return run_collection_pipeline(
        collector=collector,
        source_label=f"wikipedia-{language}",
        accepted_path=accepted_path,
        rejected_path=rejected_path,
        stats_path=stats_path,
        quality_thresholds=quality_thresholds,
        dedup_index=dedup_index,
    )


# Every non-Wikipedia collector so far shares one constructor shape:
# (*, source_config, credibility_map, max_docs, request_delay). Wikipedia is
# kept as its own function above because language is a real extra dimension
# (different output paths, different collector arg) — everything else is a
# registry entry here, so a new source is a new collector class + one line
# below, not a fourth copy-pasted wrapper function.
_SIMPLE_SOURCE_REGISTRY: dict[str, dict[str, Any]] = {
    "semantic_scholar": {
        "collector_cls": SemanticScholarCollector,
        "source_label": "semantic-scholar",
        "default_request_delay": 3.0,
        "default_accepted": Path("data/processed/semantic_scholar_documents.jsonl"),
        "default_rejected": Path("data/metadata/semantic_scholar_rejected.jsonl"),
        "default_stats": Path("data/metadata/semantic_scholar_stats.json"),
    },
    "wafa": {
        "collector_cls": WafaCollector,
        "source_label": "wafa-news",
        "default_request_delay": 2.0,
        "default_accepted": Path("data/processed/wafa_documents.jsonl"),
        "default_rejected": Path("data/metadata/wafa_rejected.jsonl"),
        "default_stats": Path("data/metadata/wafa_stats.json"),
    },
    "gdelt": {
        "collector_cls": GdeltCollector,
        "source_label": "gdelt",
        "default_request_delay": 5.0,
        "default_accepted": Path("data/processed/gdelt_documents.jsonl"),
        "default_rejected": Path("data/metadata/gdelt_rejected.jsonl"),
        "default_stats": Path("data/metadata/gdelt_stats.json"),
    },
}


def run_simple_source_pipeline(
    source_key: str,
    *,
    max_docs: int = 100,
    sources_path: Path = DEFAULT_SOURCES_CONFIG,
    quality_path: Path = DEFAULT_QUALITY_CONFIG,
    accepted_path: Optional[Path] = None,
    rejected_path: Optional[Path] = None,
    stats_path: Optional[Path] = None,
) -> dict[str, Any]:
    """Run any source registered in _SIMPLE_SOURCE_REGISTRY end to end."""
    if source_key not in _SIMPLE_SOURCE_REGISTRY:
        raise ValueError(f"Unknown source {source_key!r}; expected one of {list(_SIMPLE_SOURCE_REGISTRY)}")
    entry = _SIMPLE_SOURCE_REGISTRY[source_key]

    sources_config = load_yaml(sources_path)
    source_config = sources_config.get(source_key, {})
    if not source_config.get("enabled", False):
        raise RuntimeError(f"{source_key} collection is disabled in configs/sources.yaml")

    quality_config = load_yaml(quality_path)
    dedup_config = quality_config.get("deduplication", {})
    dedup_index = _build_dedup_index(
        threshold=float(dedup_config.get("threshold", 0.80)),
        num_perm=int(dedup_config.get("num_perm", 128)),
    )
    quality_thresholds = QualityConfig.load(quality_path)
    collector = entry["collector_cls"](
        source_config=source_config,
        credibility_map=sources_config.get("credibility_map", {}),
        max_docs=max_docs,
        request_delay=float(source_config.get("rate_limit_delay", entry["default_request_delay"])),
    )

    return run_collection_pipeline(
        collector=collector,
        source_label=entry["source_label"],
        accepted_path=accepted_path or entry["default_accepted"],
        rejected_path=rejected_path or entry["default_rejected"],
        stats_path=stats_path or entry["default_stats"],
        quality_thresholds=quality_thresholds,
        dedup_index=dedup_index,
    )


def run_semantic_scholar_pipeline(*, max_docs: int = 100, **kwargs: Any) -> dict[str, Any]:
    """Thin, backward-compatible alias — see run_simple_source_pipeline."""
    return run_simple_source_pipeline("semantic_scholar", max_docs=max_docs, **kwargs)


def run_wafa_pipeline(*, max_docs: int = 100, **kwargs: Any) -> dict[str, Any]:
    """Thin, backward-compatible alias — see run_simple_source_pipeline."""
    return run_simple_source_pipeline("wafa", max_docs=max_docs, **kwargs)


def run_gdelt_pipeline(*, max_docs: int = 100, **kwargs: Any) -> dict[str, Any]:
    """Thin, backward-compatible alias — see run_simple_source_pipeline."""
    return run_simple_source_pipeline("gdelt", max_docs=max_docs, **kwargs)


def _write_jsonl(handle, payload: dict[str, Any]) -> None:
    handle.write(json.dumps(payload, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Wikipedia ingestion.")
    parser.add_argument("--max-docs", type=int, default=100)
    parser.add_argument(
        "--language",
        default="ar",
        choices=["ar", "en"],
        help="Wikipedia language edition to collect (configs/sources.yaml must have a matching seed_categories block).",
    )
    args = parser.parse_args()
    stats = run_wikipedia_arabic_pipeline(max_docs=args.max_docs, language=args.language)
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
