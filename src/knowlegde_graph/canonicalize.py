"""
Entity canonicalization (Track E1).

Aggregates the per-document `entities` field already produced by
src/ingestion/entity_extractor.py (via scripts/run_ner.py) into corpus-scope
KGEntity records: one row per distinct (normalized text, type) pair, with
mention counts and source_doc_ids merged across every document it appears
in. This is deliberately NOT a new extraction pipeline — entity_extractor.py
already does the hard part (NER + heritage-dictionary matching, de-duplicated
per document via the same (normalized, type) key); canonicalization just
lifts that key from document scope to corpus scope. See ROADMAP.md Section 2
and Track E1.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Tuple

from src.knowlegde_graph.schemas import KGEntity, make_entity_id


def iter_ner_documents(paths: Iterable[Path]) -> Iterator[Dict[str, Any]]:
    """Yield NER-enriched documents from one or more JSONL files.

    Documents without an `entities` field (NER wasn't run on them) are
    skipped rather than erroring — callers can freely point this at a mix of
    *.ner.jsonl and not-yet-processed files.
    """
    for path in paths:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                doc = json.loads(line)
                if doc.get("entities"):
                    yield doc


def canonicalize_entities(documents: Iterable[Dict[str, Any]]) -> List[KGEntity]:
    """Aggregate per-document entity mentions into corpus-scope KGEntity records.

    Grouping key is (normalized, type) — the exact key entity_extractor.py's
    own _aggregate() already uses within one document — so corpus-scope
    canonicalization is a pure lift, not a new matching heuristic.
    """
    groups: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for doc in documents:
        doc_id = doc.get("doc_id")
        for mention in doc.get("entities", []):
            normalized = mention.get("normalized")
            entity_type = mention.get("type")
            if not normalized or not entity_type:
                continue
            key = (normalized, entity_type)
            group = groups.setdefault(
                key,
                {
                    "surface_forms": Counter(),
                    "mention_count": 0,
                    "source_doc_ids": [],
                    "canonical_hint": None,
                },
            )
            group["surface_forms"][mention.get("text", normalized)] += mention.get("mention_count", 1)
            group["mention_count"] += mention.get("mention_count", 1)
            if doc_id and doc_id not in group["source_doc_ids"]:
                group["source_doc_ids"].append(doc_id)
            # Heritage-dictionary mentions carry an authoritative canonical
            # form (entity_extractor.py's HeritageEntry.canonical) — prefer
            # it over whichever surface form happens to be most frequent.
            if mention.get("canonical") and not group["canonical_hint"]:
                group["canonical_hint"] = mention["canonical"]

    entities: List[KGEntity] = []
    for (normalized, entity_type), group in groups.items():
        canonical_name = group["canonical_hint"] or group["surface_forms"].most_common(1)[0][0]
        entities.append(
            KGEntity(
                entity_id=make_entity_id(normalized, entity_type),
                canonical_name=canonical_name,
                type=entity_type,
                mention_count=group["mention_count"],
                source_doc_ids=group["source_doc_ids"],
            )
        )

    entities.sort(key=lambda e: -e.mention_count)
    return entities


def write_kg_entities(entities: List[KGEntity], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for entity in entities:
            handle.write(json.dumps(entity.model_dump(), ensure_ascii=False) + "\n")


def read_kg_entities(path: Path) -> List[KGEntity]:
    entities: List[KGEntity] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                entities.append(KGEntity.model_validate_json(line))
    return entities
