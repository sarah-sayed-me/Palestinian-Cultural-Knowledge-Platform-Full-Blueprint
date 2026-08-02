"""Pipeline health report (Track H2) — current snapshot + drift over time + anomalies.

Works even before any scheduled run has happened (falls back to each
source's latest *_stats.json as a single current snapshot); becomes more
useful once data/metadata/scheduled_run_log.jsonl has real history from
scripts/run_scheduled_collection.py.

Usage:
    uv run python scripts/collection_health_report.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.monitoring.health_report import build_health_report

DEFAULT_RUN_LOG = Path("data/metadata/scheduled_run_log.jsonl")
DEFAULT_STATS_PATHS = {
    "wikipedia_ar": Path("data/metadata/wikipedia_ar_stats.json"),
    "wikipedia_en": Path("data/metadata/wikipedia_en_stats.json"),
    "semantic_scholar": Path("data/metadata/semantic_scholar_stats.json"),
    "wafa": Path("data/metadata/wafa_stats.json"),
    "gdelt": Path("data/metadata/gdelt_stats.json"),
}


def main() -> None:
    report = build_health_report(DEFAULT_RUN_LOG, DEFAULT_STATS_PATHS)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
