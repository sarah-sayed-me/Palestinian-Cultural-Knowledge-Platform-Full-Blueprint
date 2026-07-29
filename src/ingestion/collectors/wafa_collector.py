"""WAFA (Palestinian news agency, wafa.ps) collector.

Discovers article URLs via WAFA's own published sitemap (robots.txt explicitly
allows crawling: `Allow: /`, and lists the sitemap itself) — a discovery
mechanism the site publishes specifically for this purpose, not scraped
navigation. Walks backward day by day from today.

Article body extraction: verified empirically against real articles (see
ROADMAP.md Track D). WAFA's page template puts the breadcrumb, publish date,
title, and body all inside one `col-lg-8` column with no distinguishing class
for the body alone — the reliable anchor is the `og:title` meta tag, which
appears verbatim once at the start of the body text; everything after it is
the story (typically starting with a "<city> <date> وفا_" dateline, which is
kept as part of the text rather than stripped, matching this project's light
normalization philosophy elsewhere). The "related articles" footer widget is
stripped by src/preprocessing/arabic_normalizer.py's boilerplate patterns.

license_status is NEEDS_REVIEW — see docs/licensing_checklist.md: robots.txt
permits crawling, but that governs indexing, not redistribution, and no
confirmed reuse license was found. This project currently collects for
private research use only (see ROADMAP.md Track D), which is why this
collector is implemented despite that status.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from typing import Generator, Optional
from urllib.parse import urlparse
from xml.etree import ElementTree

import httpx
from bs4 import BeautifulSoup

from src.ingestion.base_collector import BaseCollector
from src.ingestion.schemas import (
    CredibilityTier,
    DocumentMetadata,
    Language,
    LicenseStatus,
    SourceType,
    make_doc_id,
)
from src.preprocessing.arabic_normalizer import full_clean

_SITEMAP_NS = "{http://www.sitemaps.org/schemas/sitemap/0.9}"
_DATE_RE = re.compile(r"تاريخ النشر:\s*(\d{2})/(\d{2})/(\d{4})\s+(\d{2}):(\d{2})\s*([صم])")
_CATEGORY_RE = re.compile(r"الرئيسية\s*(.+?)\s*تاريخ النشر:")


class WafaCollector(BaseCollector):
    """Collect articles from WAFA's daily sitemaps, walking backward from today."""

    SOURCE_ID = "wafa-news"
    SOURCE_NAME = "WAFA"

    BASE_URL = "https://www.wafa.ps"

    def __init__(
        self,
        *,
        source_config: Optional[dict] = None,
        credibility_map: Optional[dict] = None,
        max_docs: Optional[int] = 100,
        request_delay: float = 2.0,
    ) -> None:
        super().__init__(output_dir=None, max_docs=max_docs, request_delay=request_delay)
        self.source_config = source_config or {}
        self.credibility_map = credibility_map or {}
        self.max_retries = int(self.source_config.get("max_retries", 3))
        self.timeout = int(self.source_config.get("timeout", 30))
        self.max_days_back = int(self.source_config.get("max_days_back", 60))
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

    def collect(self) -> Generator[DocumentMetadata, None, None]:
        emitted = 0
        try:
            for url in self._iter_article_urls():
                if self.max_docs is not None and emitted >= self.max_docs:
                    return
                document = self._retry(self._fetch_document, url, retries=self.max_retries)
                if document is None:
                    continue
                emitted += 1
                yield document
                self._sleep()
        finally:
            self._client.close()

    def _iter_article_urls(self) -> Generator[str, None, None]:
        day = datetime.now(timezone.utc).date()
        for _ in range(self.max_days_back):
            urls = self._retry(self._fetch_daily_sitemap, day, retries=self.max_retries)
            if urls:
                yield from urls
            day -= timedelta(days=1)
            self._sleep()

    def _fetch_daily_sitemap(self, day) -> list[str]:
        response = self._client.get(
            f"{self.BASE_URL}/sitemap.xml", params={"yyyy": day.year, "mm": day.month, "dd": day.day}
        )
        response.raise_for_status()
        root = ElementTree.fromstring(response.content)
        return [
            loc.text.strip()
            for loc in root.iter(f"{_SITEMAP_NS}loc")
            if loc.text and "/news/" in loc.text
        ]

    def _fetch_document(self, url: str) -> Optional[DocumentMetadata]:
        response = self._client.get(url)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        og_title = soup.find("meta", property="og:title")
        title = og_title["content"].strip() if og_title and og_title.get("content") else None
        if not title:
            self.logger.warning("No og:title found for %s; skipping.", url)
            return None

        body_div = soup.find("div", class_=lambda c: bool(c) and "col-lg-8" in c)
        block_text = body_div.get_text(" ", strip=True) if body_div else ""
        if title not in block_text:
            self.logger.warning("Title not found in body block for %s; skipping.", url)
            return None
        raw_text = block_text[block_text.index(title) + len(title) :].strip()

        cleaned_text = full_clean(raw_text, is_wikipedia=False)
        if not cleaned_text:
            return None

        breadcrumb = self._extract_category(block_text)
        date_published = self._extract_date(block_text)

        return DocumentMetadata(
            doc_id=make_doc_id(url, cleaned_text),
            source_id=self.SOURCE_ID,
            title=title,
            text=cleaned_text,
            text_raw=raw_text,
            word_count=len(cleaned_text.split()),
            char_count=len(cleaned_text),
            language=Language.ARABIC_MSA,
            source_name=self.SOURCE_NAME,
            source_type=SourceType.NEWS,
            source_url=url,
            source_domain=urlparse(url).netloc,
            credibility=CredibilityTier(self.source_config.get("credibility_tier", "tier_2")),
            license_status=LicenseStatus.NEEDS_REVIEW,  # see docs/licensing_checklist.md
            seed_category=breadcrumb,
            date_published=date_published,
            tags=[breadcrumb] if breadcrumb else [],
        )

    @staticmethod
    def _extract_category(block_text: str) -> Optional[str]:
        """The category label sits inline in the same col-lg-8 block as the
        title/body, between "الرئيسية" (Home) and "تاريخ النشر:" (Published) —
        e.g. "الرئيسية محلية تاريخ النشر: ...". Verified against real pages;
        there is no separate breadcrumb element to select on this template.
        """
        match = _CATEGORY_RE.search(block_text)
        return match.group(1).strip() if match else None

    @staticmethod
    def _extract_date(block_text: str) -> Optional[datetime]:
        match = _DATE_RE.search(block_text)
        if not match:
            return None
        day, month, year, hour, minute, meridiem = match.groups()
        hour_i = int(hour) % 12
        if meridiem == "م":  # PM
            hour_i += 12
        try:
            return datetime(int(year), int(month), int(day), hour_i, int(minute), tzinfo=timezone.utc)
        except ValueError:
            return None
