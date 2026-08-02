"""
Pipeline monitoring / observability (Track H2): run health and
quality-decision drift over time.

Two data sources, combined, degrading gracefully if either is missing:

1. `data/metadata/scheduled_run_log.jsonl` (Track H1's append-only history) —
   the real time series: one entry per collection cycle, with each source's
   full stats embedded. This is what makes "drift over time" possible at
   all; without a run history there's only ever one snapshot to look at.
2. Each source's latest `data/metadata/<source>_stats.json` — a single
   current snapshot, available even before any scheduled run has ever
   happened (every collector already writes this today). Used as the
   "current health" view when there isn't run-log history yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

# A drop this large (percentage points) in accept rate vs. the immediately
# preceding run for the same source is flagged as an anomaly worth a human
# look — e.g. a source's website changed structure and extraction quietly
# broke. Not a statistically derived threshold, a practical, inspectable one.
ACCEPT_RATE_DROP_THRESHOLD = 0.30


def read_run_log(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    entries = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entries.append(json.loads(line))
    return entries


def read_latest_source_stats(stats_paths: Dict[str, Path]) -> Dict[str, Dict[str, Any]]:
    latest: Dict[str, Dict[str, Any]] = {}
    for source_id, path in stats_paths.items():
        if path.exists():
            with path.open("r", encoding="utf-8") as handle:
                latest[source_id] = json.load(handle)
    return latest


def _accept_rate(stats: Dict[str, Any]) -> Optional[float]:
    attempted = stats.get("attempted_documents")
    accepted = stats.get("accepted_documents")
    if not attempted:
        return None
    return round(accepted / attempted, 4)


def quality_decision_drift(run_log_entries: List[Dict[str, Any]], source: str) -> List[Dict[str, Any]]:
    """This source's quality_decision_distribution + accept rate, one entry
    per run it appeared in, in chronological order (run log is append-only,
    so file order already is chronological order)."""
    drift = []
    for entry in run_log_entries:
        result = entry.get("results", {}).get(source)
        if not result or result.get("status") != "ok":
            continue
        stats = result.get("stats", {})
        drift.append(
            {
                "run_at": entry.get("run_at"),
                "quality_decision_distribution": stats.get("quality_decision_distribution", {}),
                "accept_rate": _accept_rate(stats),
            }
        )
    return drift


def detect_anomalies(run_log_entries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Two anomaly kinds: a source failing outright in a run, or its accept
    rate dropping sharply vs. its own immediately preceding successful run."""
    anomalies: List[Dict[str, Any]] = []
    last_accept_rate: Dict[str, float] = {}

    for entry in run_log_entries:
        run_at = entry.get("run_at")
        for source, result in entry.get("results", {}).items():
            if result.get("status") == "error":
                anomalies.append(
                    {"run_at": run_at, "source": source, "kind": "source_failed", "detail": result.get("error")}
                )
                continue

            rate = _accept_rate(result.get("stats", {}))
            if rate is None:
                continue
            previous = last_accept_rate.get(source)
            if previous is not None and (previous - rate) >= ACCEPT_RATE_DROP_THRESHOLD:
                anomalies.append(
                    {
                        "run_at": run_at,
                        "source": source,
                        "kind": "accept_rate_drop",
                        "detail": f"accept rate dropped from {previous} to {rate}",
                    }
                )
            last_accept_rate[source] = rate

    return anomalies


def build_health_report(
    run_log_path: Path, stats_paths: Dict[str, Path]
) -> Dict[str, Any]:
    run_log_entries = read_run_log(run_log_path)
    latest_stats = read_latest_source_stats(stats_paths)

    return {
        "run_log_path": str(run_log_path),
        "runs_recorded": len(run_log_entries),
        "current_snapshot_by_source": {
            source: {
                "attempted_documents": stats.get("attempted_documents"),
                "accepted_documents": stats.get("accepted_documents"),
                "accept_rate": _accept_rate(stats),
                "quality_decision_distribution": stats.get("quality_decision_distribution", {}),
            }
            for source, stats in latest_stats.items()
        },
        "quality_decision_drift_by_source": {
            source: quality_decision_drift(run_log_entries, source) for source in stats_paths
        },
        "anomalies": detect_anomalies(run_log_entries),
    }
