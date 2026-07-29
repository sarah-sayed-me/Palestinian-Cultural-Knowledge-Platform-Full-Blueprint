"""GDELT 2.0 DOC API collector.

GDELT indexes news articles across thousands of outlets worldwide but serves
only METADATA (url, title, date, domain, language, sourcecountry) — never
full article text (see docs/licensing_checklist.md). To build a useful
document, this collector fetches the full article from its source URL, but
ONLY after checking that domain's robots.txt explicitly allows it. That check
is a fixed engineering line, not a licensing judgment call this project is
choosing to relax — it holds regardless of how the corpus is used. Absence of
a robots.txt (standard web convention) is treated as "no stated restriction";
an explicit Disallow is respected and that URL is skipped, not retried or
worked around.

Full-text extraction across many different, unknown-in-advance domains uses a
generic heuristic (concatenate all <p> tag text) rather than a hand-tuned
per-site extractor — inherently less precise than WafaCollector's verified
extraction. The existing quality checker is the backstop: pages where this
heuristic pulls too little real content are correctly rejected downstream,
not silently kept as noise.

GDELT asks unauthenticated callers to keep requests to one per 5 seconds
(stated directly in their own rate-limit response) — request_delay defaults
accordingly.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Generator, Optional
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup

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

DOC_API_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
MIN_PARAGRAPH_CHARS = 40  # filters nav/footer/caption junk out of the <p>-tag heuristic


class GdeltCollector(BaseCollector):
    """Collect full-text articles for GDELT-indexed, Palestine-related news."""

    SOURCE_ID = "gdelt"
    SOURCE_NAME = "GDELT"

    def __init__(
        self,
        *,
        source_config: Optional[dict] = None,
        credibility_map: Optional[dict] = None,
        max_docs: Optional[int] = 100,
        request_delay: float = 5.0,
    ) -> None:
        super().__init__(output_dir=None, max_docs=max_docs, request_delay=request_delay)
        self.source_config = source_config or {}
        self.credibility_map = credibility_map or {}
        self.queries = list(self.source_config.get("queries") or ["sourcecountry:PS"])
        self.max_retries = int(self.source_config.get("max_retries", 3))
        self.timeout = int(self.source_config.get("timeout", 30))
        self.max_records_per_query = int(self.source_config.get("max_records_per_query", 250))
        self._client = httpx.Client(
            headers={
                "User-Agent": (
                    "PalestinianCulturalKnowledgePlatform/0.1 "
                    "(research corpus collection; contact: local)"
                )
            },
            timeout=self.timeout,
            follow_redirects=True,
        )
        self._robots_cache: dict[str, Optional[RobotFileParser]] = {}
        self._seen_urls: set[str] = set()

    def collect(self) -> Generator[DocumentMetadata, None, None]:
        emitted = 0
        try:
            for query in self.queries:
                if self.max_docs is not None and emitted >= self.max_docs:
                    break
                articles = self._retry(self._search, query, retries=self.max_retries) or []
                for article in articles:
                    if self.max_docs is not None and emitted >= self.max_docs:
                        break
                    url = article.get("url")
                    if not url or url in self._seen_urls:
                        continue
                    self._seen_urls.add(url)
                    if not self._robots_allow(url):
                        self.logger.info("Skipping %s — robots.txt disallows.", url)
                        continue
                    document = self._retry(self._fetch_document, article, retries=self.max_retries)
                    if document is None:
                        continue
                    emitted += 1
                    yield document
                    self._sleep()
        finally:
            self._client.close()

    def _search(self, query: str) -> list[dict[str, Any]]:
        response = self._client.get(
            DOC_API_URL,
            params={
                "query": query,
                "mode": "artlist",
                "maxrecords": self.max_records_per_query,
                "format": "json",
                "sort": "hybridrel",
            },
        )
        response.raise_for_status()
        try:
            data = response.json()
        except ValueError:
            # GDELT returns a plain-text rate-limit notice (not JSON) when
            # throttling — surface it as a retryable failure, not a crash.
            raise RuntimeError(f"Non-JSON response from GDELT (likely rate-limited): {response.text[:200]}")
        return data.get("articles", []) if isinstance(data, dict) else []

    def _robots_allow(self, url: str) -> bool:
        domain = urlparse(url).netloc
        if domain not in self._robots_cache:
            self._robots_cache[domain] = self._fetch_robots_parser(domain)
        parser = self._robots_cache[domain]
        if parser is None:
            return True  # no robots.txt found — standard convention: no stated restriction
        try:
            return parser.can_fetch("*", url)
        except Exception:
            return True

    def _fetch_robots_parser(self, domain: str) -> Optional[RobotFileParser]:
        try:
            response = self._client.get(f"https://{domain}/robots.txt", timeout=10)
        except Exception:
            return None
        if response.status_code != 200:
            return None
        parser = RobotFileParser()
        parser.parse(response.text.splitlines())
        return parser

    def _fetch_document(self, article: dict[str, Any]) -> Optional[DocumentMetadata]:
        url = article["url"]
        response = self._client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        paragraphs = [p.get_text(" ", strip=True) for p in soup.find_all("p")]
        raw_text = "\n".join(p for p in paragraphs if len(p) >= MIN_PARAGRAPH_CHARS)
        cleaned_text = full_clean(raw_text, is_wikipedia=False)
        if not cleaned_text:
            return None

        title = article.get("title") or self._title_fallback(soup)
        domain = urlparse(url).netloc
        credibility = score_source(url, self.credibility_map)
        language = self._map_language(article.get("language"), cleaned_text)
        date_published = self._parse_seendate(article.get("seendate"))

        return DocumentMetadata(
            doc_id=make_doc_id(url, cleaned_text),
            source_id=self.SOURCE_ID,
            title=title,
            text=cleaned_text,
            text_raw=raw_text,
            word_count=len(cleaned_text.split()),
            char_count=len(cleaned_text),
            language=language,
            source_name=domain,
            source_type=SourceType.NEWS,
            source_url=url,
            source_domain=domain,
            # Deliberately NOT letting sources.yaml's gdelt.credibility_tier override
            # this, unlike the other collectors: GDELT aggregates from many different,
            # unknown-in-advance domains, so a single blanket tier would misrepresent
            # low-quality sources as tier_1. credibility_map's per-domain lookup (or
            # its "default" entry) is the only thing that should set this here.
            credibility=credibility.tier,
            license_status=LicenseStatus.NEEDS_REVIEW,  # see docs/licensing_checklist.md
            seed_category=article.get("sourcecountry"),
            date_published=date_published,
        )

    @staticmethod
    def _title_fallback(soup: BeautifulSoup) -> Optional[str]:
        tag = soup.find("title")
        return tag.get_text(strip=True) if tag else None

    @staticmethod
    def _parse_seendate(value: Optional[str]) -> Optional[datetime]:
        if not value:
            return None
        try:
            return datetime.strptime(value, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    @staticmethod
    def _map_language(gdelt_language: Optional[str], text: str) -> Language:
        if gdelt_language:
            lowered = gdelt_language.lower()
            if "arabic" in lowered:
                return Language.ARABIC_MSA
            if "english" in lowered:
                return Language.ENGLISH
        try:
            import langdetect

            code = langdetect.detect(text)
        except Exception:
            return Language.UNKNOWN
        if code == "ar":
            return Language.ARABIC_MSA
        if code == "en":
            return Language.ENGLISH
        return Language.UNKNOWN
