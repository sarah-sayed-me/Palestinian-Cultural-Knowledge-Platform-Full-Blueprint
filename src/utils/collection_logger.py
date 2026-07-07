"""
Structured collection logging.

Every collector writes a log entry to data/metadata/collection_log.jsonl
so the team can audit exactly what was collected, when, and why documents
were rejected.

Format: one JSON object per line (JSONL), appended — never overwritten.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

DEFAULT_LOG_PATH = Path("data/metadata/collection_log.jsonl")


class CollectionLogger:
    """Writes structured run-level and document-level logs.

    Usage:
        log = CollectionLogger(source_id="wikipedia-ar")
        log.start_run()
        ...
        log.record_success(doc_id="abc", title="...")
        log.record_rejection(doc_id="xyz", reason="too_short")
        log.finish_run()
    """

    def __init__(
        self,
        source_id: str,
        log_path: Path = DEFAULT_LOG_PATH,
    ) -> None:
        self.source_id = source_id
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

        self._run_id: Optional[str] = None
        self._started_at: Optional[datetime] = None
        self._docs_attempted = 0
        self._docs_collected = 0
        self._rejection_reasons: Dict[str, int] = {}
        self._errors: list[str] = []

    # ------------------------------------------------------------------
    # Run lifecycle
    # ------------------------------------------------------------------

    def start_run(self, extra: Optional[Dict[str, Any]] = None) -> str:
        now = datetime.now(timezone.utc)
        self._run_id = f"run_{self.source_id}_{now.strftime('%Y%m%d_%H%M%S')}"
        self._started_at = now
        self._docs_attempted = 0
        self._docs_collected = 0
        self._rejection_reasons = {}
        self._errors = []

        self._append({
            "event": "run_start",
            "run_id": self._run_id,
            "source_id": self.source_id,
            "timestamp": now.isoformat(),
            **(extra or {}),
        })
        logger.info("[%s] Collection run started: %s", self.source_id, self._run_id)
        return self._run_id

    def finish_run(self) -> Dict[str, Any]:
        now = datetime.now(timezone.utc)
        duration = (now - self._started_at).total_seconds() if self._started_at else 0

        summary = {
            "event": "run_finish",
            "run_id": self._run_id,
            "source_id": self.source_id,
            "timestamp": now.isoformat(),
            "docs_attempted": self._docs_attempted,
            "docs_collected": self._docs_collected,
            "docs_rejected": self._docs_attempted - self._docs_collected,
            "rejection_reasons": self._rejection_reasons,
            "duration_seconds": round(duration, 2),
            "errors": self._errors,
        }
        self._append(summary)
        logger.info(
            "[%s] Run finished. Collected %d / %d in %.1fs",
            self.source_id,
            self._docs_collected,
            self._docs_attempted,
            duration,
        )
        return summary

    # ------------------------------------------------------------------
    # Document-level recording
    # ------------------------------------------------------------------

    def record_attempt(self) -> None:
        self._docs_attempted += 1

    def record_success(
        self,
        doc_id: str,
        title: Optional[str] = None,
        word_count: Optional[int] = None,
    ) -> None:
        self._docs_collected += 1
        # Only log individual doc events at DEBUG to avoid huge log files
        logger.debug("[%s] ✓ %s — %s (%d words)", self.source_id, doc_id, title, word_count or 0)

    def record_rejection(self, doc_id: Optional[str], reason: str) -> None:
        self._rejection_reasons[reason] = self._rejection_reasons.get(reason, 0) + 1
        logger.debug("[%s] ✗ %s — %s", self.source_id, doc_id or "unknown", reason)

    def record_error(self, message: str) -> None:
        self._errors.append(message)
        logger.warning("[%s] ERROR: %s", self.source_id, message)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _append(self, entry: Dict[str, Any]) -> None:
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
        except Exception as exc:
            logger.error("Failed to write to collection log: %s", exc)