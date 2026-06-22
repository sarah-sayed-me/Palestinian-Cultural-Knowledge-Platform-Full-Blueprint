"""Audit existing Arabic Wikipedia seed categories and discover new ones.

Standalone diagnostic tool for Phase 1 corpus collection. Does NOT modify
configs/sources.yaml — it only produces JSON reports for manual review.

Two jobs:
  1. Audit: for every seed currently in configs/sources.yaml, check whether
     the category exists, is a redirect, and how many pages/subcategories
     it actually has on ar.wikipedia.org.
  2. Discovery: starting from broad Palestinian root categories, walk the
     category graph (reusing the collector's maintenance/diaspora filters)
     and score what's found by cultural relevance + content volume.

Run:
    uv run python scripts/audit_seed_categories.py
    uv run python scripts/audit_seed_categories.py --max-depth 2
    uv run python scripts/audit_seed_categories.py --skip-discovery
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import httpx
import wikipediaapi
import yaml

from src.ingestion.collectors.wikipedia_collector import (
    is_diaspora_terminal_category,
    is_maintenance_category,
)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULT_SOURCES_PATH = Path("configs/sources.yaml")
DEFAULT_OUTPUT_DIR = Path("reports/seed_audit")

USER_AGENT = (
    "PalestinianCulturalKnowledgePlatform-SeedAudit/0.1 "
    "(research corpus collection; contact: local)"
)
WIKI_API_URL = "https://ar.wikipedia.org/w/api.php"
CATEGORY_PREFIX = "تصنيف:"

ROOT_DISCOVERY_CATEGORIES = [
    "فلسطين",
    "الثقافة_الفلسطينية",
    "تاريخ_فلسطين",
    "فلسطينيون",
]

RELEVANCE_KEYWORDS = (
    "فلسطين", "فلسطيني", "فلسطينية", "فلسطينيون",
    "تراث", "ثقافة", "أدب", "شعر", "فن", "موسيقى",
    "تطريز", "عمارة", "مطبخ", "تاريخ", "نكبة", "لاجئ",
    "قرية", "قرى", "مدينة", "أثري", "مهرجان", "حرفة", "حرف",
)

MAX_DISCOVERY_DEPTH_DEFAULT = 2
MAX_SUBCATS_PER_NODE = 200       # safety cap: subcategories expanded per node
MAX_TOTAL_DISCOVERY_NODES = 1500  # safety cap: total categories visited
REQUEST_DELAY = 0.5

MIN_PAGES_FOR_RECOMMENDATION = 5
MIN_RELEVANCE_FOR_RECOMMENDATION = 1.0


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class CategoryAuditResult:
    category: str
    requested_title: str
    exists: bool
    resolved_title: Optional[str] = None
    is_redirect: bool = False
    pages: int = 0
    subcategories: int = 0
    maintenance_pages_skipped: int = 0
    suggested_replacement: Optional[str] = None
    notes: str = ""


@dataclass
class DiscoveredCategory:
    title: str
    depth: int
    pages: int
    subcategories: int
    relevance_score: float
    total_score: float
    parent: Optional[str] = None


# ---------------------------------------------------------------------------
# Wikipedia client + shared helpers
# ---------------------------------------------------------------------------


def build_wiki_client(language: str = "ar") -> wikipediaapi.Wikipedia:
    try:
        return wikipediaapi.Wikipedia(
            user_agent=USER_AGENT,
            language=language,
            extract_format=wikipediaapi.ExtractFormat.WIKI,
            timeout=30,
        )
    except TypeError:
        return wikipediaapi.Wikipedia(language, extract_format=wikipediaapi.ExtractFormat.WIKI)


def category_title(name: str) -> str:
    return name if name.startswith(CATEGORY_PREFIX) else f"{CATEGORY_PREFIX}{name}"


def classify_members(page) -> tuple[int, int, int]:
    """Return (page_count, subcategory_count, maintenance_subcats_skipped).

    Reuses is_maintenance_category from the collector so audit/discovery
    counts match what the real collector would actually keep.
    """
    members = getattr(page, "categorymembers", {})
    if isinstance(members, dict):
        members = list(members.values())

    pages = 0
    subcats = 0
    maintenance_skipped = 0
    for member in members:
        title = str(getattr(member, "title", ""))
        ns = getattr(member, "ns", None)
        is_cat = title.startswith(("Category:", CATEGORY_PREFIX)) or ns == 14
        if is_cat:
            if is_maintenance_category(title):
                maintenance_skipped += 1
                continue
            subcats += 1
        else:
            pages += 1
    return pages, subcats, maintenance_skipped


def suggest_replacement(http_client: httpx.Client, query: str) -> Optional[str]:
    """Search the Category namespace for a plausible replacement title."""
    clean_query = query.replace("_", " ").replace(CATEGORY_PREFIX, "")
    params = {
        "action": "query",
        "list": "search",
        "srsearch": clean_query,
        "srnamespace": 14,  # Category namespace
        "srlimit": 3,
        "format": "json",
    }
    try:
        response = http_client.get(WIKI_API_URL, params=params, timeout=15)
        response.raise_for_status()
        hits = response.json().get("query", {}).get("search", [])
    except Exception:
        return None
    if not hits:
        return None
    return hits[0].get("title")


# ---------------------------------------------------------------------------
# Job 1: Audit existing seeds
# ---------------------------------------------------------------------------


def load_seed_categories(sources_path: Path, language: str = "ar") -> list[str]:
    with sources_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return list((config.get("wikipedia", {}).get("seed_categories") or {}).get(language, []))


def audit_category(
    wiki: wikipediaapi.Wikipedia,
    http_client: httpx.Client,
    seed: str,
) -> CategoryAuditResult:
    requested = category_title(seed)
    page = wiki.page(requested)
    exists = bool(page.exists())

    if not exists:
        return CategoryAuditResult(
            category=seed,
            requested_title=requested,
            exists=False,
            suggested_replacement=suggest_replacement(http_client, seed),
            notes="Category page does not exist on ar.wikipedia.org",
        )

    resolved_title = str(getattr(page, "title", requested))
    is_redirect = resolved_title != requested
    pages, subcats, maint_skipped = classify_members(page)

    notes = ""
    suggestion = None
    if pages == 0 and subcats == 0:
        notes = "Category exists but is empty (no pages, no subcategories)"
        suggestion = suggest_replacement(http_client, seed)
    elif pages < 5:
        notes = f"Very few direct pages ({pages}) — likely too narrow as a standalone seed"

    return CategoryAuditResult(
        category=seed,
        requested_title=requested,
        exists=True,
        resolved_title=resolved_title,
        is_redirect=is_redirect,
        pages=pages,
        subcategories=subcats,
        maintenance_pages_skipped=maint_skipped,
        suggested_replacement=suggestion,
        notes=notes,
    )


def run_seed_audit(
    wiki: wikipediaapi.Wikipedia,
    http_client: httpx.Client,
    seeds: list[str],
) -> list[CategoryAuditResult]:
    results = []
    for seed in seeds:
        results.append(audit_category(wiki, http_client, seed))
        time.sleep(REQUEST_DELAY)
    return results


# ---------------------------------------------------------------------------
# Job 2: Discover new candidate categories
# ---------------------------------------------------------------------------


def relevance_keyword_score(title: str) -> float:
    clean = title.replace(CATEGORY_PREFIX, "")
    return float(sum(1 for kw in RELEVANCE_KEYWORDS if kw in clean))


def compute_total_score(relevance: float, pages: int, subcats: int) -> float:
    """Weighted score: cultural-relevance keywords dominate; pages and
    subcategories contribute log-scaled so one huge category can't dominate
    purely on volume."""
    page_term = math.log1p(pages)
    subcat_term = math.log1p(subcats) * 0.5
    return round(relevance * 3.0 + page_term + subcat_term, 3)


def discover_categories(
    wiki: wikipediaapi.Wikipedia,
    roots: list[str],
    *,
    max_depth: int,
) -> dict[str, DiscoveredCategory]:
    discovered: dict[str, DiscoveredCategory] = {}
    seen: set[str] = set()
    queue: deque[tuple[str, int, Optional[str]]] = deque(
        (category_title(root), 0, None) for root in roots
    )

    while queue and len(discovered) < MAX_TOTAL_DISCOVERY_NODES:
        title, depth, parent = queue.popleft()
        if title in seen:
            continue
        seen.add(title)

        page = wiki.page(title)
        if not page.exists():
            continue

        pages, subcats, _ = classify_members(page)
        score = relevance_keyword_score(title)
        discovered[title] = DiscoveredCategory(
            title=title,
            depth=depth,
            pages=pages,
            subcategories=subcats,
            relevance_score=score,
            total_score=compute_total_score(score, pages, subcats),
            parent=parent,
        )

        if depth >= max_depth:
            time.sleep(REQUEST_DELAY)
            continue

        members = getattr(page, "categorymembers", {})
        if isinstance(members, dict):
            members = list(members.values())

        added = 0
        for member in members:
            member_title = str(getattr(member, "title", ""))
            ns = getattr(member, "ns", None)
            is_cat = member_title.startswith(("Category:", CATEGORY_PREFIX)) or ns == 14
            if not is_cat or member_title in seen:
                continue
            if is_maintenance_category(member_title):
                continue
            if is_diaspora_terminal_category(member_title):
                # Record it, but never expand past it (same rule as the collector).
                queue.append((member_title, max_depth, title))
                continue
            queue.append((member_title, depth + 1, title))
            added += 1
            if added >= MAX_SUBCATS_PER_NODE:
                break
        time.sleep(REQUEST_DELAY)

    return discovered


def classify_discovered(
    discovered: dict[str, DiscoveredCategory],
) -> tuple[list[DiscoveredCategory], list[dict]]:
    recommended: list[DiscoveredCategory] = []
    rejected: list[dict] = []
    for cat in discovered.values():
        if cat.relevance_score >= MIN_RELEVANCE_FOR_RECOMMENDATION and cat.pages >= MIN_PAGES_FOR_RECOMMENDATION:
            recommended.append(cat)
        else:
            reasons = []
            if cat.relevance_score < MIN_RELEVANCE_FOR_RECOMMENDATION:
                reasons.append("low_cultural_relevance")
            if cat.pages < MIN_PAGES_FOR_RECOMMENDATION:
                reasons.append("too_few_pages")
            rejected.append({**asdict(cat), "rejection_reasons": reasons})

    recommended.sort(key=lambda c: c.total_score, reverse=True)
    rejected.sort(key=lambda r: r["total_score"], reverse=True)
    return recommended, rejected


# ---------------------------------------------------------------------------
# Output + entrypoint
# ---------------------------------------------------------------------------


def write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit existing seed categories and discover new candidates."
    )
    parser.add_argument("--sources", type=Path, default=DEFAULT_SOURCES_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-depth", type=int, default=MAX_DISCOVERY_DEPTH_DEFAULT)
    parser.add_argument(
        "--skip-discovery",
        action="store_true",
        help="Only audit existing seeds; skip new-category discovery.",
    )
    args = parser.parse_args()

    wiki = build_wiki_client()
    http_client = httpx.Client(headers={"User-Agent": USER_AGENT})

    try:
        seeds = load_seed_categories(args.sources)
        print(f"Auditing {len(seeds)} existing seed categories...")
        audit_results = run_seed_audit(wiki, http_client, seeds)
        write_json(
            args.output_dir / "category_audit_report.json",
            [asdict(r) for r in audit_results],
        )

        bad = [r for r in audit_results if not r.exists or (r.pages == 0 and r.subcategories == 0)]
        print(
            f"Audit complete. {len(bad)}/{len(seeds)} seeds are missing or empty. "
            f"See {args.output_dir / 'category_audit_report.json'}"
        )

        if not args.skip_discovery:
            print(
                f"Discovering categories from roots {ROOT_DISCOVERY_CATEGORIES} "
                f"(max depth {args.max_depth})..."
            )
            discovered = discover_categories(wiki, ROOT_DISCOVERY_CATEGORIES, max_depth=args.max_depth)
            recommended, rejected = classify_discovered(discovered)

            write_json(
                args.output_dir / "recommended_seed_categories.json",
                [asdict(c) for c in recommended],
            )
            write_json(args.output_dir / "rejected_seed_categories.json", rejected)

            print(
                f"Discovery complete. {len(discovered)} categories visited, "
                f"{len(recommended)} recommended, {len(rejected)} rejected.\n"
                f"See {args.output_dir / 'recommended_seed_categories.json'} and "
                f"{args.output_dir / 'rejected_seed_categories.json'}."
            )
    finally:
        http_client.close()


if __name__ == "__main__":
    main()