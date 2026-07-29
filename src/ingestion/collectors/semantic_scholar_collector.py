"""Semantic Scholar Academic Graph API (S2AG) collector.

Collects full open-access PDF text when S2AG reports one available
(`openAccessPdf`), falling back to title + abstract otherwise. Open-access
papers are explicitly published for free redistribution (typically CC-BY or
similar) — that is what "open access" means — so full-text collection here is
license_status=CLEAR, same as the abstract-only case, not a relaxation of it.
Non-open-access papers still collect abstract-only, same as before; this
collector still never fetches a PDF S2AG doesn't itself mark open access.

The public API is free and unauthenticated but tightly rate-limited (shared
across all unauthenticated callers, observed to 429 quickly) — request_delay
defaults higher than other collectors, and retries lean on BaseCollector's
exponential backoff rather than a tight retry loop.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generator, Optional

import httpx

from src.ingestion.base_collector import BaseCollector
from src.ingestion.credibility_scorer import score_source
from src.ingestion.schemas import (
    CredibilityTier,
    DocumentMetadata,
    Language,
    LicenseStatus,
    SourceType,
    make_doc_id,
)
from src.preprocessing.arabic_normalizer import full_clean

MIN_PDF_CHARS = 500  # below this, extraction likely failed/garbled — fall back to abstract


class SemanticScholarCollector(BaseCollector):
    """Collect Palestine/Arabic-heritage-related papers from S2AG — full open-access text when available, abstract otherwise."""

    SOURCE_ID = "semantic-scholar"
    SOURCE_NAME = "Semantic Scholar"

    API_URL = "https://api.semanticscholar.org/graph/v1/paper/search"
    FIELDS = "title,abstract,url,year,authors,venue,externalIds,openAccessPdf"
    PAGE_SIZE = 100

    def __init__(
        self,
        *,
        source_config: Optional[dict] = None,
        credibility_map: Optional[dict] = None,
        max_docs: Optional[int] = 100,
        request_delay: float = 3.0,
    ) -> None:
        super().__init__(output_dir=None, max_docs=max_docs, request_delay=request_delay)
        self.source_config = source_config or {}
        self.credibility_map = credibility_map or {}
        self.queries = list(self.source_config.get("queries", []))
        self.max_retries = int(self.source_config.get("max_retries", 5))
        self.timeout = int(self.source_config.get("timeout", 30))
        self._seen_paper_ids: set[str] = set()
        self._client = httpx.Client(
            headers={
                "User-Agent": (
                    "PalestinianCulturalKnowledgePlatform/0.1 "
                    "(research corpus collection; contact: local)"
                )
            },
            timeout=self.timeout,
        )

    def collect(self) -> Generator[DocumentMetadata, None, None]:
        emitted = 0
        try:
            for query in self.queries:
                if self.max_docs is not None and emitted >= self.max_docs:
                    break
                for paper in self._search(query):
                    if self.max_docs is not None and emitted >= self.max_docs:
                        break
                    document = self._paper_to_document(paper, query)
                    if document is None:
                        continue
                    emitted += 1
                    yield document
                    self._sleep()
        finally:
            self._client.close()

    def _search(self, query: str) -> Generator[dict[str, Any], None, None]:
        offset = 0
        while True:
            if self.max_docs is not None and len(self._seen_paper_ids) >= self.max_docs:
                return
            response = self._retry(self._fetch_page, query, offset, self.PAGE_SIZE, retries=self.max_retries)
            if response is None:
                self.logger.warning("Giving up on query %r after repeated failures.", query)
                return
            data = response.get("data", [])
            if not data:
                return
            for paper in data:
                paper_id = paper.get("paperId")
                if not paper_id or paper_id in self._seen_paper_ids:
                    continue
                self._seen_paper_ids.add(paper_id)
                yield paper
            if len(data) < self.PAGE_SIZE:
                return
            offset += self.PAGE_SIZE
            self._sleep()

    def _fetch_page(self, query: str, offset: int, limit: int) -> dict[str, Any]:
        response = self._client.get(
            self.API_URL,
            params={"query": query, "fields": self.FIELDS, "offset": offset, "limit": limit},
        )
        response.raise_for_status()
        return response.json()

    def _paper_to_document(self, paper: dict[str, Any], query: str) -> Optional[DocumentMetadata]:
        try:
            abstract = paper.get("abstract") or ""
            full_text, is_full_text = self._resolve_text(paper)
            if not full_text.strip():
                return None  # neither a fetchable OA PDF nor an abstract — nothing to collect

            cleaned_text = full_clean(full_text, is_wikipedia=False)
            if not cleaned_text:
                return None

            title = paper.get("title") or None
            source_url = paper.get("url") or self._doi_url(paper)
            credibility = score_source(source_url, self.credibility_map)
            language = self._detect_language(cleaned_text)
            year = paper.get("year")
            date_published = datetime(int(year), 1, 1, tzinfo=timezone.utc) if year else None
            venue = paper.get("venue") or None
            authors = [a.get("name") for a in (paper.get("authors") or []) if a.get("name")]

            return DocumentMetadata(
                doc_id=make_doc_id(source_url or title or paper.get("paperId"), cleaned_text),
                source_id=self.SOURCE_ID,
                title=title,
                text=cleaned_text,
                text_raw=full_text if is_full_text else abstract,
                summary=full_clean(abstract, is_wikipedia=False) if is_full_text and abstract else None,
                word_count=len(cleaned_text.split()),
                char_count=len(cleaned_text),
                language=language,
                source_name=venue or self.SOURCE_NAME,
                source_type=SourceType.ACADEMIC_PAPER,
                source_url=source_url,
                source_domain="semanticscholar.org",
                credibility=CredibilityTier(
                    self.source_config.get("credibility_tier", credibility.tier)
                ),
                # Open access = published for free redistribution; abstract-only relies
                # on S2AG's own metadata terms — both CLEAR. See docs/licensing_checklist.md.
                license_status=LicenseStatus.CLEAR,
                seed_category=query,
                date_published=date_published,
                tags=authors,
            )
        except Exception as exc:
            self.logger.warning("Failed to parse paper %s: %s", paper.get("paperId"), exc)
            return None

    def _resolve_text(self, paper: dict[str, Any]) -> tuple[str, bool]:
        """Returns (text, is_full_text). Tries the open-access PDF S2AG points
        at first; falls back to the abstract if there isn't one, the fetch
        fails, or extraction yields too little text to trust."""
        abstract = paper.get("abstract") or ""
        pdf_url = (paper.get("openAccessPdf") or {}).get("url")
        if not pdf_url:
            return abstract, False

        pdf_text = self._retry(self._fetch_pdf_text, pdf_url, retries=2, backoff=2.0)
        if pdf_text and len(pdf_text) >= MIN_PDF_CHARS:
            return pdf_text, True

        self.logger.info("OA PDF unusable for %s; falling back to abstract.", pdf_url)
        return abstract, False

    def _fetch_pdf_text(self, pdf_url: str) -> str:
        import pdfplumber
        from io import BytesIO

        response = self._client.get(pdf_url, follow_redirects=True)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "pdf" not in content_type.lower() and not pdf_url.lower().endswith(".pdf"):
            raise ValueError(f"Response doesn't look like a PDF (content-type={content_type!r})")

        with pdfplumber.open(BytesIO(response.content)) as pdf:
            pages = [page.extract_text() or "" for page in pdf.pages]
        return "\n".join(pages)

    @staticmethod
    def _doi_url(paper: dict[str, Any]) -> Optional[str]:
        doi = (paper.get("externalIds") or {}).get("DOI")
        return f"https://doi.org/{doi}" if doi else None

    @staticmethod
    def _detect_language(text: str) -> Language:
        try:
            import langdetect

            code = langdetect.detect(text)
        except Exception:
            return Language.ENGLISH  # S2AG is overwhelmingly English-language papers
        if code == "ar":
            return Language.ARABIC_MSA
        if code == "en":
            return Language.ENGLISH
        return Language.UNKNOWN
