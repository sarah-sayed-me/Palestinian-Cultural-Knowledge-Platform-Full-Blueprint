"""
Abstract base class for all data collectors.

Every collector (Wikipedia, GDELT, WAFA, etc.) must subclass BaseCollector
and implement `collect()`. This enforces a uniform interface so the
pipeline orchestrator can treat all collectors identically.

Contract:
  - `collect()` is a generator that yields DocumentMetadata objects.
  Collector is responsible for:
    - Fetching
    - Parsing
    - Normalization

    Quality validation is performed later in the pipeline.
  - Duplicates are detected externally by the pipeline using DuplicationIndex.
  - The base class provides shared helpers: logging, retry, progress bars.
"""

from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional

from src.ingestion.schemas import DocumentMetadata

logger = logging.getLogger(__name__)


class BaseCollector(ABC):
    """Abstract base for all collectors.

    Subclasses must set:
        SOURCE_ID   — machine-readable identifier, e.g. 'wikipedia-ar'
        SOURCE_NAME — human-readable name, e.g. 'Arabic Wikipedia'

    And implement:
        collect() — yields DocumentMetadata instances
    """

    SOURCE_ID: str = "base"
    SOURCE_NAME: str = "Base Collector"

    def __init__(
        self,
        output_dir: Optional[str] = None,
        max_docs: Optional[int] = None,
        request_delay: float = 1.0,
    ) -> None:
        """
        Args:
            output_dir:    Where to save raw JSON files (None = no disk save).
            max_docs:      Stop after collecting this many documents (None = unlimited).
            request_delay: Seconds to wait between HTTP requests (politeness).
        """
        self.output_dir = output_dir
        self.max_docs = max_docs
        self.request_delay = request_delay
        self.logger = logging.getLogger(f"collector.{self.SOURCE_ID}")

    @abstractmethod
    def collect(self) -> Generator[DocumentMetadata, None, None]:
        """Yield DocumentMetadata objects one at a time.

        Implementors should:
        1. Fetch raw content from the source.
        2. Parse and extract text.
        3. Normalise text using src.preprocessing.arabic_normalizer.
        4. Build and yield a DocumentMetadata object.
        5. Handle retries and errors gracefully (log and continue).
        """
        ...

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    def _sleep(self, seconds: Optional[float] = None) -> None:
        """Polite delay between requests."""
        time.sleep(seconds if seconds is not None else self.request_delay)

    def _retry(
        self,
        fn,
        *args,
        retries: int = 3,
        backoff: float = 2.0,
        **kwargs,
    ):
        """Call `fn(*args, **kwargs)` with exponential backoff retries.

        Returns the result or None if all retries fail.
        """
        for attempt in range(retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                wait = backoff ** attempt
                self.logger.warning(
                    "Attempt %d/%d failed for %s: %s. Retrying in %.1fs",
                    attempt + 1,
                    retries,
                    fn.__name__ if hasattr(fn, "__name__") else "fn",
                    exc,
                    wait,
                )
                time.sleep(wait)
        self.logger.error("All %d retries exhausted.", retries)
        return None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(source_id={self.SOURCE_ID!r})"