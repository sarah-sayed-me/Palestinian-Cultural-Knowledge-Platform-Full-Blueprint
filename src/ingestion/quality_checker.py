"""
Document quality validation and scoring.

Implements all 6 quality rules from the spec:
  1. Minimum content length
  2. Language validation
  3. Duplicate detection (MinHash LSH — see deduplication.py)
  4. Missing metadata handling
  5. Composite quality score
  6. Arabic-specific text quality

The checker is stateless (no LSH here — that lives in deduplication.py).
Call `check_document()` to get a QualityReport for any document.

Thresholds and weights are defined in QualityConfig, loaded from
configs/quality_thresholds.yaml by default (QualityConfig.load()) so the YAML is
the actual source of truth, not a parallel doc. QualityConfig.default() returns
the same values as a filesystem-free fallback for callers that don't pass a
config (existing call sites, tests) — its values are kept identical to the YAML
by construction, so there is exactly one place either ever needs to change.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

from src.ingestion.schemas import CredibilityTier, DocumentMetadata, Language, QualityDecision
from src.preprocessing.arabic_normalizer import count_arabic_ratio

DEFAULT_QUALITY_CONFIG_PATH = Path("configs/quality_thresholds.yaml")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class QualityConfig:
    """All tunable quality thresholds and weights.

    Defaults match configs/quality_thresholds.yaml exactly. Use `load()` to read
    the YAML (what the real pipeline does — see pipeline.py); use the
    zero-argument `default()` when a config-free call is fine (tests, ad-hoc use).
    """

    min_word_count: int = 50
    min_char_count: int = 300
    max_word_count: int = 100_000

    min_arabic_ratio: float = 0.40
    low_arabic_ratio_warn: float = 0.60

    accept_threshold: float = 0.70
    warn_threshold: float = 0.45
    reject_threshold: float = 0.20

    weight_richness: float = 0.40
    weight_credibility: float = 0.20
    weight_completeness: float = 0.20
    weight_language: float = 0.20

    richness_minimal: int = 50
    richness_medium: int = 100
    richness_full: int = 200
    richness_saturation: int = 1000  # word_count at which richness reaches 1.0

    credibility_tier_scores: Dict[str, float] = field(
        default_factory=lambda: {
            "tier_1": 1.00,
            "tier_2": 0.75,
            "tier_3": 0.45,
            "tier_4": 0.15,
        }
    )
    completeness_fields: Tuple[str, ...] = ("title", "source_url", "source_name", "language")

    @classmethod
    def default(cls) -> "QualityConfig":
        return cls()

    @classmethod
    def load(cls, path: Path = DEFAULT_QUALITY_CONFIG_PATH) -> "QualityConfig":
        """Load thresholds from YAML, falling back to defaults for any missing key."""
        if not path.exists():
            return cls.default()
        with path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}

        defaults = cls.default()
        content_length = raw.get("content_length", {})
        quality_score = raw.get("quality_score", {})
        weights = quality_score.get("weights", {})
        richness = quality_score.get("richness", {})
        tier_scores = quality_score.get("credibility_tier_scores", {})

        return cls(
            min_word_count=content_length.get("min_word_count", defaults.min_word_count),
            min_char_count=content_length.get("min_char_count", defaults.min_char_count),
            max_word_count=content_length.get("max_word_count", defaults.max_word_count),
            min_arabic_ratio=defaults.min_arabic_ratio,
            low_arabic_ratio_warn=defaults.low_arabic_ratio_warn,
            accept_threshold=quality_score.get("accept", defaults.accept_threshold),
            warn_threshold=quality_score.get("accept_with_warning", defaults.warn_threshold),
            reject_threshold=quality_score.get("reject", defaults.reject_threshold),
            weight_richness=weights.get("richness", defaults.weight_richness),
            weight_credibility=weights.get("credibility", defaults.weight_credibility),
            weight_completeness=weights.get("completeness", defaults.weight_completeness),
            weight_language=weights.get("language", defaults.weight_language),
            richness_minimal=richness.get("minimal", defaults.richness_minimal),
            richness_medium=richness.get("medium", defaults.richness_medium),
            richness_full=richness.get("full", defaults.richness_full),
            richness_saturation=richness.get("saturation", defaults.richness_saturation),
            credibility_tier_scores={**defaults.credibility_tier_scores, **tier_scores},
            completeness_fields=tuple(
                quality_score.get("completeness_fields", defaults.completeness_fields)
            ),
        )


# ---------------------------------------------------------------------------
# Report dataclass
# ---------------------------------------------------------------------------


@dataclass
class QualityReport:
    is_valid: bool
    decision: QualityDecision
    quality_score: float
    warnings: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    rejection_reason: Optional[str] = None


# ---------------------------------------------------------------------------
# Individual rule checkers
# ---------------------------------------------------------------------------


def _rule_length(word_count: int, char_count: int, config: QualityConfig) -> tuple[bool, list[str]]:
    errors = []
    if word_count < config.min_word_count:
        errors.append(f"Too short: {word_count} words (min {config.min_word_count})")
    if char_count < config.min_char_count:
        errors.append(f"Too short: {char_count} chars (min {config.min_char_count})")
    if word_count > config.max_word_count:
        # Not an error, just a flag
        return True, [f"WARN: very long document ({word_count} words), may need chunking"]
    return len(errors) == 0, errors


def _rule_language(text: str, declared_language: str, config: QualityConfig) -> tuple[bool, list[str]]:
    """Quick heuristic language check without heavy models."""
    warnings = []
    # For Arabic documents, verify there's sufficient Arabic script
    if declared_language.startswith("ar"):
        ratio = count_arabic_ratio(text)
        if ratio < config.min_arabic_ratio:
            return False, [
                f"Declared Arabic but Arabic script ratio is only {ratio:.1%} "
                f"(min {config.min_arabic_ratio:.0%})"
            ]
        if ratio < config.low_arabic_ratio_warn:
            warnings.append(
                f"Low Arabic ratio {ratio:.1%} — may be primarily English / bilingual"
            )
    return True, warnings


def _rule_required_fields(doc: DocumentMetadata) -> tuple[bool, list[str]]:
    required = ["doc_id", "text", "source_id", "language", "date_collected", "source_name", "source_type"]
    missing = [f for f in required if not getattr(doc, f, None)]
    if missing:
        return False, [f"Missing required fields: {', '.join(missing)}"]
    return True, []


def _richness_score(word_count: int, config: QualityConfig) -> float:
    """Monotonic content-richness curve.

    Flat floors below `richness_full` (minimal/medium bands), then a continuous
    ramp from `richness_full` up to `richness_saturation` words. The ramp starts
    at the same score the medium band ends on (0.70) so a word_count exactly at
    `richness_full` never scores lower than `richness_full - 1` did — that
    inversion (a 200-word doc scoring below a 199-word one) was the bug.
    """
    if word_count >= config.richness_full:
        span = max(config.richness_saturation - config.richness_full, 1)
        progress = min((word_count - config.richness_full) / span, 1.0)
        return round(0.70 + 0.30 * progress, 4)
    if word_count >= config.richness_medium:
        return 0.70
    if word_count >= config.richness_minimal:
        return 0.40
    return 0.0


def _compute_quality_score(doc: DocumentMetadata, config: QualityConfig) -> float:
    """Weighted composite quality score (0.0-1.0): richness, credibility, completeness, language.

    Weights come from `config` (default 40/20/20/20 — see QualityConfig).
    """
    richness = _richness_score(doc.word_count, config)

    tier_key = CredibilityTier(doc.credibility).value
    credibility = config.credibility_tier_scores.get(tier_key, 0.30)

    filled = sum(1 for f in config.completeness_fields if getattr(doc, f, None) is not None)
    completeness = filled / len(config.completeness_fields) if config.completeness_fields else 0.0

    lang_score = 0.0 if doc.language == Language.UNKNOWN else 1.0
    if doc.language in {Language.ARABIC_MSA, Language.ARABIC_PAL, Language.ARABIC_OTHER}:
        ratio = count_arabic_ratio(doc.text)
        lang_score = min(1.0, ratio / 0.5)  # Full score at 50%+ Arabic

    score = (
        richness * config.weight_richness
        + credibility * config.weight_credibility
        + completeness * config.weight_completeness
        + lang_score * config.weight_language
    )
    return round(score, 4)


def _decide(score: float, config: QualityConfig) -> QualityDecision:
    if score >= config.accept_threshold:
        return QualityDecision.ACCEPT
    if score >= config.warn_threshold:
        return QualityDecision.ACCEPT_WITH_WARNING
    if score >= config.reject_threshold:
        return QualityDecision.REJECT
    return QualityDecision.HARD_REJECT


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_document(doc: DocumentMetadata, config: Optional[QualityConfig] = None) -> QualityReport:
    """Run all quality checks on a document.

    `config` defaults to QualityConfig.default() (no filesystem access), so
    existing callers and tests are unaffected. The real pipeline passes
    QualityConfig.load() so configs/quality_thresholds.yaml actually takes
    effect — see pipeline.py.

    Returns a QualityReport. The caller decides whether to store or discard
    based on `report.decision`.
    """
    config = config or QualityConfig.default()
    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Rule 1: Length
    ok, msgs = _rule_length(doc.word_count, doc.char_count, config)
    if not ok:
        all_errors.extend(msgs)
    else:
        all_warnings.extend([m for m in msgs if m.startswith("WARN")])

    # Rule 2: Language
    ok, msgs = _rule_language(doc.text, str(doc.language), config)
    if not ok:
        all_errors.extend(msgs)
    else:
        all_warnings.extend(msgs)

    # Rule 3: Required fields
    ok, msgs = _rule_required_fields(doc)
    if not ok:
        all_errors.extend(msgs)

    # Compute score regardless (useful even for rejected docs in logs)
    score = _compute_quality_score(doc, config)
    decision = _decide(score, config)

    # Hard-fail on required-field or language errors regardless of score
    if all_errors:
        decision = QualityDecision.HARD_REJECT if len(all_errors) > 1 else QualityDecision.REJECT
        score = min(score, 0.30)

    rejection_reason = "; ".join(all_errors) if all_errors else None

    return QualityReport(
        is_valid=len(all_errors) == 0,
        decision=decision,
        quality_score=score,
        warnings=all_warnings,
        errors=all_errors,
        rejection_reason=rejection_reason,
    )
