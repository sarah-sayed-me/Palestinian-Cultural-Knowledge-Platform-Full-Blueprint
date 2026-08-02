r"""Run one full collection cycle across every source (Track H1).

Deliberately NOT a long-running daemon — it runs every enabled source once
and exits. Recurrence is the OS scheduler's job (Windows Task Scheduler /
cron), not something to reimplement in Python: the OS scheduler already
handles machine reboots, missed runs, and logging failures better than a
custom always-on process would, and needs no extra dependency (no
APScheduler) for something this infrequent (a collection run every few
hours/days, not every few seconds).

Every source's real collector already degrades gracefully on its own
per-source failures (rate limits, robots.txt, network errors — see Track D)
so one source failing doesn't need to be caught here; each is still wrapped
individually so one source's total failure (e.g. Postgres unreachable) never
prevents the others from running.

Every run appends one JSON line to data/metadata/scheduled_run_log.jsonl —
this is what scripts/collection_health_report.py (Track H2) reads.

Usage (run once, manually or from a scheduler):
    uv run python scripts/run_scheduled_collection.py
    uv run python scripts/run_scheduled_collection.py --max-docs 50 --sources wafa gdelt

Windows Task Scheduler (run daily at 03:00, adjust the two paths for your machine):
    schtasks /create /tn "PalestinianKnowledgePlatform-DailyCollection" /sc daily /st 03:00 ^
        /tr "\"C:\path\to\.venv\Scripts\python.exe\" \"C:\path\to\scripts\run_scheduled_collection.py\"" ^
        /ru "%USERNAME%"
    schtasks /query /tn "PalestinianKnowledgePlatform-DailyCollection"     # verify it registered
    schtasks /delete /tn "PalestinianKnowledgePlatform-DailyCollection"   # remove it later

cron (Linux/Mac, daily at 03:00):
    0 3 * * * cd /path/to/project && /path/to/.venv/bin/python scripts/run_scheduled_collection.py >> logs/cron.log 2>&1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from src.ingestion.pipeline import (
    run_gdelt_pipeline,
    run_semantic_scholar_pipeline,
    run_wafa_pipeline,
    run_wikipedia_arabic_pipeline,
)

DEFAULT_LOG_PATH = Path("data/metadata/scheduled_run_log.jsonl")

_RUNNERS = {
    "wikipedia_ar": lambda max_docs: run_wikipedia_arabic_pipeline(max_docs=max_docs, language="ar"),
    "wikipedia_en": lambda max_docs: run_wikipedia_arabic_pipeline(max_docs=max_docs, language="en"),
    "semantic_scholar": lambda max_docs: run_semantic_scholar_pipeline(max_docs=max_docs),
    "wafa": lambda max_docs: run_wafa_pipeline(max_docs=max_docs),
    "gdelt": lambda max_docs: run_gdelt_pipeline(max_docs=max_docs),
}


def run_cycle(*, sources: List[str], max_docs: int, log_path: Path) -> Dict[str, Any]:
    started = time.time()
    results: Dict[str, Any] = {}
    for source in sources:
        runner = _RUNNERS[source]
        source_started = time.time()
        try:
            stats = runner(max_docs)
            results[source] = {"status": "ok", "duration_seconds": round(time.time() - source_started, 2), "stats": stats}
        except Exception as exc:  # noqa: BLE001 - one source's failure must not abort the cycle
            results[source] = {
                "status": "error",
                "duration_seconds": round(time.time() - source_started, 2),
                "error": f"{type(exc).__name__}: {exc}",
            }

    cycle_summary = {
        "run_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "sources_attempted": sources,
        "sources_succeeded": [s for s, r in results.items() if r["status"] == "ok"],
        "sources_failed": [s for s, r in results.items() if r["status"] == "error"],
        "total_duration_seconds": round(time.time() - started, 2),
        "results": results,
    }

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(cycle_summary, ensure_ascii=False) + "\n")

    return cycle_summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one full collection cycle across every enabled source.")
    parser.add_argument("--sources", nargs="+", choices=list(_RUNNERS), default=list(_RUNNERS))
    parser.add_argument("--max-docs", type=int, default=50, help="Per-source cap for this cycle.")
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG_PATH)
    args = parser.parse_args()

    summary = run_cycle(sources=args.sources, max_docs=args.max_docs, log_path=args.log)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
