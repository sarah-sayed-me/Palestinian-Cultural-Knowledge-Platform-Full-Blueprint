"""Source credibility lookup for ingestion metadata."""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from src.ingestion.schemas import CredibilityTier


@dataclass(frozen=True)
class CredibilityScore:
    tier: CredibilityTier
    score: float


def score_source(
    source_url: str | None,
    credibility_map: dict | None = None,
    *,
    default_tier: str = "tier_3",
    default_score: float = 0.45,
) -> CredibilityScore:
    """Return the configured credibility tier and numeric score for a URL."""
    domain = _domain_from_url(source_url)
    mapping = credibility_map or {}
    entry = mapping.get(domain) or mapping.get(_strip_www(domain)) or mapping.get("default")

    if not entry:
        return CredibilityScore(CredibilityTier(default_tier), default_score)

    if isinstance(entry, dict):
        tier = entry.get("tier", default_tier)
        score = float(entry.get("score", default_score))
    else:
        tier = default_tier
        score = float(entry)
    return CredibilityScore(CredibilityTier(tier), score)


def _domain_from_url(source_url: str | None) -> str:
    if not source_url:
        return ""
    parsed = urlparse(source_url)
    return (parsed.netloc or parsed.path).lower()


def _strip_www(domain: str) -> str:
    return domain[4:] if domain.startswith("www.") else domain
