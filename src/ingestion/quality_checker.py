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
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

from src.ingestion.schemas import CredibilityTier, DocumentMetadata, Language, QualityDecision
from src.preprocessing.arabic_normalizer import count_arabic_ratio


# ---------------------------------------------------------------------------
# Thresholds (keep in sync with configs/quality_thresholds.yaml)
# ---------------------------------------------------------------------------

MIN_WORD_COUNT = 50
MIN_CHAR_COUNT = 300
MAX_WORD_COUNT = 100_000
MIN_ARABIC_RATIO = 0.20   # At least 20% Arabic chars for "ar" language docs
ACCEPT_THRESHOLD = 0.70
WARN_THRESHOLD = 0.45


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


def _rule_length(word_count: int, char_count: int) -> tuple[bool, list[str]]:
    errors = []
    if word_count < MIN_WORD_COUNT:
        errors.append(f"Too short: {word_count} words (min {MIN_WORD_COUNT})")
    if char_count < MIN_CHAR_COUNT:
        errors.append(f"Too short: {char_count} chars (min {MIN_CHAR_COUNT})")
    if word_count > MAX_WORD_COUNT:
        # Not an error, just a flag
        return True, [f"WARN: very long document ({word_count} words), may need chunking"]
    return len(errors) == 0, errors


def _rule_language(text: str, declared_language: str) -> tuple[bool, list[str]]:
    """Quick heuristic language check without heavy models."""
    warnings = []
    # For Arabic documents, verify there's sufficient Arabic script
    if declared_language.startswith("ar"):
        ratio = count_arabic_ratio(text)
        if ratio < MIN_ARABIC_RATIO:
            return False, [
                f"Declared Arabic but Arabic script ratio is only {ratio:.1%} "
                f"(min {MIN_ARABIC_RATIO:.0%})"
            ]
        if ratio < 0.40:
            warnings.append(
                f"Low Arabic ratio {ratio:.1%} — may be primarily English / bilingual"
            )
    return True, warnings


def _rule_required_fields(doc: DocumentMetadata) -> tuple[bool, list[str]]:
    required = ["doc_id", "text", "source_id", "language", "date_collected"]
    missing = [f for f in required if not getattr(doc, f, None)]
    if missing:
        return False, [f"Missing required fields: {', '.join(missing)}"]
    return True, []


def _compute_quality_score(doc: DocumentMetadata) -> float:
    """Weighted composite quality score (0.0–1.0).

    Components:
      Content richness  30%
      Source credibility 35%
      Metadata completeness 20%
      Language validity  15%
    """
    # 1. Content richness (30%)
    if doc.word_count >= 200:
        richness = 1.0
    elif doc.word_count >= 100:
        richness = 0.70
    elif doc.word_count >= MIN_WORD_COUNT:
        richness = 0.40
    else:
        richness = 0.0

    # 2. Source credibility (35%)
    tier_scores = {
        CredibilityTier.TIER_1: 1.0,
        CredibilityTier.TIER_2: 0.75,
        CredibilityTier.TIER_3: 0.45,
        CredibilityTier.TIER_4: 0.15,
    }
    credibility = tier_scores.get(CredibilityTier(doc.credibility), 0.30)

    # 3. Metadata completeness (20%)
    meta_fields = ["title", "date_published", "source_url", "language"]
    filled = sum(1 for f in meta_fields if getattr(doc, f, None) is not None)
    completeness = filled / len(meta_fields)

    # 4. Language validity (15%)
    lang_score = 0.0 if doc.language == Language.UNKNOWN else 1.0
    # Bonus for correctly detected Arabic
    if str(doc.language).startswith("ar"):
        ratio = count_arabic_ratio(doc.text)
        lang_score = min(1.0, ratio / 0.5)  # Full score at 50%+ Arabic

    score = (
        richness * 0.30
        + credibility * 0.35
        + completeness * 0.20
        + lang_score * 0.15
    )
    return round(score, 4)


def _decide(score: float) -> QualityDecision:
    if score >= ACCEPT_THRESHOLD:
        return QualityDecision.ACCEPT
    if score >= WARN_THRESHOLD:
        return QualityDecision.ACCEPT_WITH_WARNING
    if score >= 0.20:
        return QualityDecision.REJECT
    return QualityDecision.HARD_REJECT


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def check_document(doc: DocumentMetadata) -> QualityReport:
    """Run all quality checks on a document.

    Returns a QualityReport. The caller decides whether to store or discard
    based on `report.decision`.
    """
    all_errors: list[str] = []
    all_warnings: list[str] = []

    # Rule 1: Length
    ok, msgs = _rule_length(doc.word_count, doc.char_count)
    if not ok:
        all_errors.extend(msgs)
    else:
        all_warnings.extend([m for m in msgs if m.startswith("WARN")])

    # Rule 2: Language
    ok, msgs = _rule_language(doc.text, str(doc.language))
    if not ok:
        all_errors.extend(msgs)
    else:
        all_warnings.extend(msgs)

    # Rule 3: Required fields
    ok, msgs = _rule_required_fields(doc)
    if not ok:
        all_errors.extend(msgs)

    # Compute score regardless (useful even for rejected docs in logs)
    score = _compute_quality_score(doc)
    decision = _decide(score)

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
