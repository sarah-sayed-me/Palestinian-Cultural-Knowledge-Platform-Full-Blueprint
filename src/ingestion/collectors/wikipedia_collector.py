"""Arabic Wikipedia collector for Phase 1 corpus ingestion."""

from __future__ import annotations

from email import generator
import json
from pathlib import Path
from random import seed
from typing import Generator, Iterable, Optional
from urllib.parse import quote

from tqdm import tqdm

from src.ingestion.base_collector import BaseCollector
from src.ingestion.credibility_scorer import score_source
from src.ingestion.schemas import (
    CredibilityTier,
    DocumentMetadata,
    Language,
    SourceType,
    make_doc_id,
)
from src.preprocessing.arabic_normalizer import full_clean
from src.utils.collection_logger import CollectionLogger

MAINTENANCE_CATEGORY_PATTERNS = (
    "بوابة",
    "صيانة",
    "صفحات تستخدم",
    "صفحات مع",
    "صفحات بها",
    "قالب",
    "جميع المقالات",
    "جميع مقالات",
    "مقالات يتيمة",
    "أخطاء الاستشهاد",
    "الاستشهاد",
    "ويكي بيانات",

    # NEW
    "وصلات خارجية مكسورة",
    "بحاجة لمصادر",
    "بأخطاء في المراجع",
    "تحوي نصا",
    "ISBN",
    "بذرة",
    "بحاجة لتعديل",
    "قوالب خرائط",

    # v2 — morphological / phrase coverage (plurals & "لا تقبل ..." phrasing
    # weren't matched by the singular/short forms above)
    "قوالب",            # plural of "قالب" — catches "تستعمل قوالب معلومات"
    "بوابات",           # plural of "بوابة" — catches "ربط البوابات المعادل"
    "لا تقبل",          # catches "لا تقبل التصنيف المعادل" / "لا تقبل ربط البوابات المعادل"
    "تحتاج إلى",        # catches "تحتاج إلى صور" / "تحتاج إلى تدقيق" etc.
    "مقالات بها",       # "صفحات بها" above doesn't match "مقالات بها أقسام فارغة"
    "أقسام فارغة",
)


def is_maintenance_category(category_name: str) -> bool:
    """True if category_name is Wikipedia bookkeeping/admin noise, not topical content."""
    return any(pattern in category_name for pattern in MAINTENANCE_CATEGORY_PATTERNS)

DIASPORA_TERMINAL_PATTERNS = (
    "من أصل فلسطيني",
)


def is_diaspora_terminal_category(category_name: str) -> bool:
    """True if category_name marks a Palestinian-diaspora-origin category.

    Categories like "Jordanians of Palestinian origin" contain genuinely
    relevant articles, but their *subcategories* (ministers, parliament
    members, born-in-city grids) typically belong to the host country's
    category system, not Palestinian topics. We collect this category's
    own articles but never recurse past it — this is a structural rule
    about category *type*, not a nationality blocklist, so it doesn't
    exclude any host-country term.
    """
    return any(pattern in category_name for pattern in DIASPORA_TERMINAL_PATTERNS)

class WikipediaCollector(BaseCollector):
    """Collect Palestine-related articles from Wikipedia category seeds."""

    SOURCE_ID = "wikipedia-ar"
    SOURCE_NAME = "Arabic Wikipedia"
    CATEGORY_PREFIXES = {
        "ar": "تصنيف:",
        "en": "Category:",
    }

    def __init__(
        self,
        *,
        language: str = "ar",
        source_config: Optional[dict] = None,
        credibility_map: Optional[dict] = None,
        output_dir: Optional[str] = "data/raw/wikipedia/ar",
        max_docs: Optional[int] = 100,
        request_delay: float = 1.0,
    ) -> None:
        super().__init__(output_dir=output_dir, max_docs=max_docs, request_delay=request_delay)
        self.language = language
        self.source_config = source_config or {}
        self.credibility_map = credibility_map or {}
        self.category_depth = int(self.source_config.get("category_depth", 1))
        self.max_articles_per_category = int(
            self.source_config.get("max_articles_per_category", 100)
        )
        self.max_articles_per_language = (
            int(self.source_config["max_articles_per_language"])
            if self.source_config.get("max_articles_per_language")
            else None
        )
        if self.max_articles_per_language is not None:
            self.max_docs = (
                self.max_articles_per_language
                if self.max_docs is None
                else min(self.max_docs, self.max_articles_per_language)
            )
        self.max_retries = int(self.source_config.get("max_retries", 3))
        self.timeout = int(self.source_config.get("timeout", 30))
        self.seed_categories = list(
            (self.source_config.get("seed_categories") or {}).get(language, [])
        )
        self.output_path = Path(output_dir) if output_dir else None
        self.collection_logger = CollectionLogger(source_id=self.SOURCE_ID)
        self._seen_pages: set[str] = set()
        self._seen_categories: set[str] = set()
        self._wiki = self._build_wiki_client()

    def collect(self) -> Generator[DocumentMetadata, None, None]:
        """Yield collected Wikipedia documents one at a time."""
        self.collection_logger.start_run(
            {
                "language": self.language,
                "seed_categories": self.seed_categories,
                "max_docs": self.max_docs,
                "category_depth": self.category_depth,
            }
        )
        emitted = 0
        try:
            candidates = self._iter_candidate_pages()
            progress = tqdm(candidates, total=self.max_docs, desc=f"{self.SOURCE_NAME} articles")
            for seed, page in progress:
                if self.max_docs is not None and emitted >= self.max_docs:
                    break
                self.collection_logger.record_attempt()
                document = self._page_to_document(page, seed_category=seed)
                if document is None:
                    continue
                emitted += 1
                self.collection_logger.record_success(
                    doc_id=document.doc_id,
                    title=document.title,
                    word_count=document.word_count,
                )
                if self.output_path:
                    self._write_raw_snapshot(document)
                yield document
                self._sleep()
        finally:
            self.collection_logger.finish_run()

    def _build_wiki_client(self):
        try:
            import wikipediaapi
        except ImportError as exc:
            raise ImportError("Install wikipedia-api to run Wikipedia collection") from exc

        user_agent = (
            "PalestinianCulturalKnowledgePlatform/0.1 "
            "(research corpus collection; contact: local)"
        )
        try:
            return wikipediaapi.Wikipedia(
                user_agent=user_agent,
                language=self.language,
                extract_format=wikipediaapi.ExtractFormat.WIKI,
                timeout=self.timeout,
            )
        except TypeError:
            return wikipediaapi.Wikipedia(
                self.language,
                extract_format=wikipediaapi.ExtractFormat.WIKI,
            )

    def _iter_candidate_pages(self) -> Iterable:
        """Round-robin across seed categories so no single large seed
        (e.g. the broad top-level 'فلسطين' category) can exhaust max_docs
        before other seeds (cuisine, poetry, music, etc.) get a turn.
        """
        seed_generators = [self._iter_seed_pages(seed) for seed in self.seed_categories]
        active = list(seed_generators)
        while active:
            still_active = []
            for generator in active:
                try:
                    yield next(generator)
                    still_active.append(generator)
                except StopIteration:
                    continue
            active = still_active

    def _iter_seed_pages(self, seed: str) -> Iterable:
        category_page = self._get_page(self._category_title(seed))
        if self._page_exists(category_page):
            yield from self._walk_category(category_page, depth=0, seed=seed)
            return

        direct_page = self._get_page(seed)
        if self._page_exists(direct_page) and self._is_article(direct_page):
            key = self._page_key(direct_page)
            if key not in self._seen_pages:
                self._seen_pages.add(key)
                yield seed, direct_page

    def _walk_category(self, category_page, *, depth: int, seed: str) -> Iterable:
        category_key = self._page_key(category_page)
        if category_key in self._seen_categories:
            return
        self._seen_categories.add(category_key)

        yielded_for_category = 0
        members = self._category_members(category_page)
        for member in members:
            if self.max_docs is not None and len(self._seen_pages) >= self.max_docs:
                return

            if self._is_category(member):
                member_title = str(getattr(member, "title", ""))
                if is_maintenance_category(member_title):
                    continue
                if is_diaspora_terminal_category(member_title):
                    # Harvest this category's own articles only; stop here.
                    yield from self._walk_category(
                        member, depth=self.category_depth, seed=seed
                    )
                elif depth < self.category_depth:
                    yield from self._walk_category(member, depth=depth + 1, seed=seed)
                continue

            if not self._is_article(member):
                continue

            key = self._page_key(member)
            if key in self._seen_pages:
                continue
            self._seen_pages.add(key)
            yielded_for_category += 1
            yield seed, member

            if yielded_for_category >= self.max_articles_per_category:
                return

    def _category_members(self, category_page) -> list:
        def load_members():
            members = getattr(category_page, "categorymembers", {})
            if callable(members):
                members = members.values()
            elif isinstance(members, dict):
                members = members.values()
            return list(members)

        members = self._retry(load_members, retries=self.max_retries)
        return members or []

    def _get_page(self, title: str):
        return self._retry(self._wiki.page, title, retries=self.max_retries)

    def _page_to_document(
        self, page, *, seed_category: Optional[str] = None
    ) -> Optional[DocumentMetadata]:
        try:
            raw_text = getattr(page, "text", "") or ""
            cleaned_text = full_clean(raw_text, is_wikipedia=True)
            if not cleaned_text:
                self.collection_logger.record_rejection(
                    self._page_key(page), "empty_after_cleaning"
                )
                return None

            source_url = self._page_url(page)
            source_info = self._language_config()
            credibility = score_source(source_url, self.credibility_map)
            word_count = len(cleaned_text.split())
            categories = self._page_categories(page)

            return DocumentMetadata(
                doc_id=make_doc_id(source_url, cleaned_text),
                source_id=source_info.get("source_id", self.SOURCE_ID),
                title=getattr(page, "title", None),
                text=cleaned_text,
                text_raw=raw_text,
                word_count=word_count,
                char_count=len(cleaned_text),
                language=Language.ARABIC_MSA if self.language == "ar" else Language.ENGLISH,
                source_name=source_info.get("name", self.SOURCE_NAME),
                source_type=SourceType.ENCYCLOPEDIA,
                source_url=source_url,
                source_domain=f"{self.language}.wikipedia.org",
                credibility=CredibilityTier(source_info.get("credibility_tier", credibility.tier)),
                wikipedia_page_id=self._optional_int(page, "pageid"),
                wikipedia_revid=self._optional_int(page, "revision_id", "revid", "lastrevid"),
                wikipedia_categories=categories,
                seed_category=seed_category,
                raw_file_path=None,
            )
        except Exception as exc:
            self.collection_logger.record_error(f"Failed to parse page {self._page_key(page)}: {exc}")
            return None

    def _write_raw_snapshot(self, document: DocumentMetadata) -> None:
        assert self.output_path is not None
        self.output_path.mkdir(parents=True, exist_ok=True)
        raw_path = self.output_path / f"{document.doc_id}.json"
        payload = document.model_dump(mode="json")
        with raw_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2)

    def _language_config(self) -> dict:
        for entry in self.source_config.get("languages", []):
            if entry.get("code") == self.language:
                return entry
        return {}

    def _category_title(self, seed: str) -> str:
        prefix = self.CATEGORY_PREFIXES.get(self.language, "Category:")
        return seed if seed.startswith(prefix) else f"{prefix}{seed}"

    def _page_url(self, page) -> str:
        fullurl = getattr(page, "fullurl", None)
        if fullurl:
            return fullurl
        title = getattr(page, "title", "")
        return f"https://{self.language}.wikipedia.org/wiki/{quote(title.replace(' ', '_'))}"

    def _page_categories(self, page) -> list[str]:
        categories = getattr(page, "categories", {}) or {}
        names = categories.keys() if isinstance(categories, dict) else categories
        cleaned = (
            str(name).replace("تصنيف:", "").replace("Category:", "")
            for name in names
        )
        return sorted(name for name in cleaned if not is_maintenance_category(name))

    def _page_key(self, page) -> str:
        return str(getattr(page, "title", "") or getattr(page, "pageid", "unknown"))

    def _page_exists(self, page) -> bool:
        if page is None:
            return False
        exists = getattr(page, "exists", None)
        return bool(exists()) if callable(exists) else bool(exists)

    def _is_category(self, page) -> bool:
        title = str(getattr(page, "title", ""))
        return title.startswith(("Category:", "تصنيف:")) or getattr(page, "ns", None) == 14

    def _is_article(self, page) -> bool:
        return self._page_exists(page) and not self._is_category(page)

    def _optional_int(self, obj, *names: str) -> Optional[int]:
        for name in names:
            value = getattr(obj, name, None)
            if value is not None:
                try:
                    return int(value)
                except (TypeError, ValueError):
                    return None
        return None
