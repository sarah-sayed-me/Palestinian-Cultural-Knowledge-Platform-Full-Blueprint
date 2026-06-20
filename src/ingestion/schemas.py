"""
Unified document schema for the Palestinian Cultural Knowledge Platform.

Every document collected from ANY source must conform to DocumentMetadata
before it is accepted into the corpus. This is the single source of truth
for the data contract between the ingestion layer and all downstream stages
(NLP, KG, RAG).

Schema version: 1.1
Changes from v1.0:
  - Added wikipedia_categories field (list of category strings from AR Wikipedia)
  - Added wikipedia_page_id (stable Wikipedia page ID for cross-referencing)
  - Added wikipedia_revid (revision ID for reproducibility)
  - Replaced HttpUrl with plain str for source_url (avoids serialisation edge-cases)
  - Added rejection_reason field for bad_docs tracking
  - Fixed Config → model_config for Pydantic v2 compatibility
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator, model_validator


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class Language(str, Enum):
    ARABIC_MSA = "ar-MSA"      # Modern Standard Arabic
    ARABIC_PAL = "ar-PAL"      # Palestinian dialect
    ARABIC_OTHER = "ar-OTHER"  # Other Arabic dialect
    ENGLISH = "en"
    MULTILINGUAL = "multi"
    UNKNOWN = "unknown"


class SourceType(str, Enum):
    NEWS = "news"
    ACADEMIC_PAPER = "academic_paper"
    NGO_REPORT = "ngo_report"
    BOOK_CHAPTER = "book_chapter"
    ORAL_TESTIMONY = "oral_testimony"
    ENCYCLOPEDIA = "encyclopedia"
    MUSEUM_RECORD = "museum_record"
    GOVERNMENT_DOC = "government_doc"
    SOCIAL_MEDIA = "social_media"
    POETRY = "poetry"
    LEGAL_DOC = "legal_doc"
    BOOK = "book"
    ARCHIVE_DOCUMENT = "archive_document"
    INTERVIEW = "interview"
    IMAGE_METADATA = "image_metadata"
    OTHER = "other"


class ContentCategory(str, Enum):
    CONFLICT = "conflict"
    CULTURE = "culture"
    HISTORY = "history"
    ARTS_LITERATURE = "arts_literature"
    POLITICS = "politics"
    DAILY_LIFE = "daily_life"
    FOOD_CUISINE = "food_cuisine"
    RELIGION = "religion"
    ARCHITECTURE = "architecture"
    EDUCATION = "education"
    ECONOMY = "economy"
    MUSIC = "music"
    FOLKLORE = "folklore"
    HERITAGE = "heritage"
    BIOGRAPHY = "biography"
    GEOGRAPHY = "geography"
    UNCATEGORIZED = "uncategorized"


class CredibilityTier(str, Enum):
    TIER_1 = "tier_1"   # Established institutions (Wikipedia, UN, Reuters)
    TIER_2 = "tier_2"   # Regional credible sources (WAFA, B'Tselem)
    TIER_3 = "tier_3"   # Community / smaller sources
    TIER_4 = "tier_4"   # Unverified / needs manual review


class QualityDecision(str, Enum):
    ACCEPT = "accept"
    ACCEPT_WITH_WARNING = "accept_with_warning"
    REJECT = "reject"
    HARD_REJECT = "hard_reject"


# ---------------------------------------------------------------------------
# Main document model
# ---------------------------------------------------------------------------


class DocumentMetadata(BaseModel):
    # --- Unique identity ---
    doc_id: str = Field(
        description="SHA-256 hash of (source_url + text[:200]). "
                    "Stable across re-runs for the same content."
    )
    source_id: str = Field(
        description="Machine-readable source identifier, e.g. 'wikipedia-ar', 'wafa'"
    )

    # --- Content ---
    title: Optional[str] = None
    text: str = Field(description="Full cleaned text ready for NLP")
    text_raw: Optional[str] = Field(
        default=None,
        description="Original un-normalised text. Stored for debugging, not for NLP."
    )
    summary: Optional[str] = Field(
        default=None,
        description="Auto-generated 2–3 sentence summary (filled later)"
    )
    word_count: int
    char_count: int

    # --- Language ---
    language: Language
    dialect_region: Optional[str] = Field(
        default=None,
        description="e.g. 'Palestinian', 'Egyptian'. Only for Arabic dialect docs."
    )
    is_bilingual: bool = Field(
        default=False,
        description="True if the document contains significant AR and EN content"
    )

    # --- Source & provenance ---
    source_name: str
    source_type: SourceType
    source_url: Optional[str] = None
    source_domain: Optional[str] = None
    credibility: CredibilityTier

    # --- Wikipedia-specific (None for non-Wikipedia sources) ---
    wikipedia_page_id: Optional[int] = Field(
        default=None,
        description="Stable Wikipedia page ID for cross-referencing with Wikidata"
    )
    wikipedia_revid: Optional[int] = Field(
        default=None,
        description="Revision ID captured at collection time for reproducibility"
    )
    wikipedia_categories: List[str] = Field(
        default_factory=list,
        description="Raw category strings from Wikipedia, e.g. ['فلسطين', 'تاريخ فلسطين']"
    )

    # --- Temporal ---
    date_published: Optional[datetime] = None
    date_collected: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    date_processed: Optional[datetime] = None
    decade: Optional[int] = Field(
        default=None,
        description="Decade bucket for temporal analysis, e.g. 1940, 1950, 2020"
    )

    # --- Classification (filled by downstream NLP pipeline) ---
    category: ContentCategory = ContentCategory.UNCATEGORIZED
    category_confidence: Optional[float] = Field(
        default=None, ge=0.0, le=1.0
    )
    tags: List[str] = Field(default_factory=list)
    geographic_scope: Optional[str] = Field(
        default=None,
        description="e.g. 'Gaza', 'West Bank', 'Pre-1948', 'Diaspora', 'General'"
    )

    # --- Quality signals ---
    quality_score: Optional[float] = Field(default=None, ge=0.0, le=1.0)
    quality_decision: Optional[QualityDecision] = None
    has_duplicate: bool = False
    minhash_signature: Optional[List[int]] = Field(
        default=None,
        description="MinHash signature bands for LSH deduplication"
    )
    is_duplicate_of: Optional[str] = Field(
        default=None,
        description="doc_id of the canonical document if this is a duplicate"
    )
    rejection_reason: Optional[str] = Field(
        default=None,
        description="Human-readable reason if the document was rejected"
    )

    # --- NLP artifacts (filled by downstream pipeline) ---
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Filled by NER pipeline in Phase 2"
    )
    topic_id: Optional[int] = Field(
        default=None,
        description="BERTopic cluster ID, filled in Phase 3"
    )
    topic_label: Optional[str] = None
    embedding_id: Optional[str] = Field(
        default=None,
        description="ChromaDB document ID, filled in Phase 9"
    )

    # --- File references ---
    raw_file_path: Optional[str] = None
    processed_file_path: Optional[str] = None

    model_config = {"use_enum_values": True}

    # --- Validators ---

    @field_validator("decade", mode="before")
    @classmethod
    def compute_decade(cls, v: Optional[int]) -> Optional[int]:
        if v is None:
            return None
        # Round down to nearest decade
        return (v // 10) * 10

    @model_validator(mode="after")
    def set_decade_from_date(self) -> "DocumentMetadata":
        if self.decade is None and self.date_published is not None:
            self.decade = (self.date_published.year // 10) * 10
        return self

    # --- Helpers ---

    def to_hf_dict(self) -> Dict[str, Any]:
        """Return a flat dictionary suitable for HuggingFace datasets.

        Drops large/internal fields (text_raw, minhash_signature, raw_file_path,
        processed_file_path) and converts datetimes to ISO strings.
        """
        d = self.model_dump(
            exclude={
                "text_raw",
                "minhash_signature",
                "raw_file_path",
                "processed_file_path",
                "entities",        # populated later
                "embedding_id",    # populated later
            }
        )
        # Serialise datetimes
        for key in ("date_published", "date_collected", "date_processed"):
            if d.get(key) is not None:
                d[key] = d[key].isoformat() if hasattr(d[key], "isoformat") else str(d[key])
        return d


# ---------------------------------------------------------------------------
# Factory helper
# ---------------------------------------------------------------------------


def make_doc_id(source_url: Optional[str], text: str) -> str:
    """Generate a stable SHA-256 document ID.

    Uses the first 200 characters of the text as a fingerprint alongside
    the URL so that the same article re-collected gets the same ID.
    """
    fingerprint = (source_url or "") + text[:200]
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()