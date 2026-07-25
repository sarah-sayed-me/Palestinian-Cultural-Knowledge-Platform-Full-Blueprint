"""
Knowledge graph schemas: KGEntity and KGRelation.

KGEntity is a corpus-scope canonicalization of the per-document entity mentions
already produced by src/ingestion/entity_extractor.py — it aggregates the same
{text, normalized, type, mention_count} shape across documents rather than
introducing a new extraction pipeline. KGRelation is the output of relation
extraction (Track E). Both are sibling models to DocumentMetadata, referenced by
doc_id / entity_id — see ROADMAP.md Section 2.
"""

from __future__ import annotations

import hashlib
from typing import List, Optional

from pydantic import BaseModel, Field


class KGEntity(BaseModel):
    entity_id: str = Field(
        description="SHA-256 of (normalized_name + type) until entity-linked, at which "
        "point wikidata_qid becomes the preferred external identifier."
    )
    canonical_name: str
    type: str = Field(
        description="Reuses entity_extractor.py's existing types, e.g. PERSON, LOCATION, "
        "ORGANIZATION, HERITAGE_FOOD, HERITAGE_CRAFT, ..."
    )
    wikidata_qid: Optional[str] = Field(
        default=None, description="Filled by entity linking (Track E); None until linked."
    )
    mention_count: int = Field(default=0, ge=0)
    source_doc_ids: List[str] = Field(default_factory=list)


class KGRelation(BaseModel):
    relation_id: str
    subject_entity_id: str
    predicate: str
    object_entity_id: str
    confidence: float = Field(ge=0.0, le=1.0)
    source_doc_id: str
    evidence_sentence: Optional[str] = None


def make_entity_id(normalized_name: str, entity_type: str) -> str:
    fingerprint = f"{normalized_name}:{entity_type}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()


def make_relation_id(
    subject_entity_id: str, predicate: str, object_entity_id: str, source_doc_id: str
) -> str:
    fingerprint = f"{subject_entity_id}:{predicate}:{object_entity_id}:{source_doc_id}"
    return hashlib.sha256(fingerprint.encode("utf-8")).hexdigest()
