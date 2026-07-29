"""
Wikidata alias-table entity linker (Track E2).

Cheap-first, per ROADMAP.md's own note: exact/fuzzy match against a
pre-pulled alias dump (scripts/fetch_wikidata_aliases.py), rather than
reaching for mGENRE's full neural entity-linking model. Escalate to mGENRE
only if this linker's precision (see eval/kg_eval.py) proves insufficient —
"validate cheap first."

Two-tier matching:
  1. Exact match on the normalized name (using the same normalize_arabic()
     the rest of the ingestion pipeline already uses, so linking is
     consistent with how documents/entities were cleaned).
  2. Fuzzy fallback (difflib.SequenceMatcher ratio) above a conservative
     threshold — catches near-misses (definite article, minor spelling
     variance) without over-matching unrelated entities.

Before either tier runs, Wikidata items matching _EXCLUDED_INSTANCE_OF are
dropped from the index entirely (disambiguation pages, newspapers, films,
...) — added after real linking output showed this corpus's top three
entities by mention count each collided with a same-labeled item of the
wrong kind. See that constant's own comment for the specific real examples.

Candidate generation for the fuzzy tier uses a character-bigram inverted
index (see _bigrams/_candidate_indices below), not a full scan. This was
NOT the first design: an earlier version pruned candidates purely by string
length (any alias outside a length window provably cannot reach
FUZZY_THRESHOLD, since difflib's ratio() is bounded by 2*min(len)/(len_sum)).
That length prune is real and lossless, but at this corpus's actual scale —
11,565 entities and ~7,700 alias names, both clustering heavily in the 5-13
character range (confirmed by inspecting the real length histograms, not
assumed) — the length-windowed candidate sets were still hundreds of
thousands of comparisons and didn't finish in several minutes. The bigram
index is a genuine approximation (unlike the length prune): two strings
that are 90%+ similar could, in principle, share very few bigrams if their
differing characters are spread out. That's an accepted trade-off for a
linker this project's own roadmap already describes as "a cheap heuristic,
not a disambiguation engine" — see eval/kg_eval.py's real precision numbers
for whether it holds up in practice.
"""

from __future__ import annotations

import difflib
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from src.knowlegde_graph.schemas import KGEntity
from src.preprocessing.arabic_normalizer import normalize_arabic

FUZZY_THRESHOLD = 0.90
# An alias must share at least this many bigrams with the query to be
# considered a fuzzy candidate at all — cheap enough to run over every
# alias, and aggressive enough to keep the expensive SequenceMatcher.ratio()
# step down to a small candidate set per query.
MIN_SHARED_BIGRAMS = 2

# Wikidata items whose instance_of (P31) is one of these are excluded from
# the index entirely, even when their label exactly matches an entity we're
# trying to link. Found by actually checking real linked QIDs against
# Wikidata, not assumed: "فلسطين" (Palestine — this corpus's single most-
# mentioned entity) initially linked to a newspaper named "Felesteen";
# "حيفا" (Haifa) linked to a 1996 film titled "Haifa"; "غزة" (Gaza) linked to
# a Wikimedia disambiguation page. All three coincidentally share a label
# with the real place/topic without being it. A general-purpose entity
# linker would need a much larger exclusion list (or proper type-aware
# disambiguation); this one is deliberately narrow — just the classes that
# were observed causing real, high-impact mismatches on this corpus.
_EXCLUDED_INSTANCE_OF = {
    "Q4167410",  # Wikimedia disambiguation page
    "Q11032",  # newspaper
    "Q1002697",  # periodical
    "Q41298",  # magazine
    "Q11424",  # film
    "Q506240",  # television film
    "Q5398426",  # television series
    "Q7889",  # video game
    "Q571",  # book
    "Q13442814",  # scholarly article
}


def _bigrams(s: str) -> Set[str]:
    if len(s) < 2:
        return {s} if s else set()
    return {s[i : i + 2] for i in range(len(s) - 1)}


@dataclass
class LinkResult:
    qid: Optional[str]
    matched_alias: Optional[str] = None
    method: str = "none"  # "exact" | "fuzzy" | "none"
    score: float = 0.0


class WikidataAliasLinker:
    """Indexes every (normalized alias -> qid) pair from an alias dump for
    O(1) exact lookups, plus a bigram-indexed fuzzy fallback."""

    def __init__(self, alias_entries: List[dict]):
        self._exact_index: Dict[str, str] = {}
        self._alias_by_qid: Dict[str, List[str]] = {}
        # Flat list of (normalized, qid, original_alias); the bigram index
        # below stores positions into this list rather than duplicating the
        # tuples themselves.
        self._aliases: List[Tuple[str, str, str]] = []
        self._bigram_index: Dict[str, List[int]] = {}

        for entry in alias_entries:
            if _EXCLUDED_INSTANCE_OF.intersection(entry.get("instance_of") or []):
                continue
            qid = entry["qid"]
            names = [entry.get("label")] + list(entry.get("aliases") or [])
            names = [n for n in names if n]
            self._alias_by_qid[qid] = names
            for name in names:
                key = normalize_arabic(name)
                # First one wins on exact-index collisions — not an error:
                # some aliases are genuinely ambiguous across entities (e.g.
                # common surnames). The alias table is a cheap heuristic, not
                # a disambiguation engine — see the eval/kg_eval.py notes on
                # what this linker gets wrong.
                self._exact_index.setdefault(key, qid)
                index = len(self._aliases)
                self._aliases.append((key, qid, name))
                for bigram in _bigrams(key):
                    self._bigram_index.setdefault(bigram, []).append(index)

    def _candidate_indices(self, normalized: str) -> List[int]:
        """Aliases sharing at least MIN_SHARED_BIGRAMS bigrams with the
        query, further pruned to only lengths that could reach
        FUZZY_THRESHOLD (see module docstring for both filters)."""
        bigrams = _bigrams(normalized)
        if not bigrams:
            return []
        shared_counts: Counter[int] = Counter()
        for bigram in bigrams:
            for index in self._bigram_index.get(bigram, []):
                shared_counts[index] += 1

        threshold = min(MIN_SHARED_BIGRAMS, len(bigrams))
        min_len, max_len = self._window(len(normalized))
        return [
            index
            for index, count in shared_counts.items()
            if count >= threshold and min_len <= len(self._aliases[index][0]) <= max_len
        ]

    @classmethod
    def from_dump(cls, path: Path) -> "WikidataAliasLinker":
        from src.knowlegde_graph.wikidata_aliases import read_alias_dump

        return cls(read_alias_dump(path))

    def link(self, canonical_name: str) -> LinkResult:
        normalized = normalize_arabic(canonical_name)

        qid = self._exact_index.get(normalized)
        if qid:
            return LinkResult(qid=qid, matched_alias=canonical_name, method="exact", score=1.0)

        best_qid, best_alias, best_score = None, None, 0.0
        for index in self._candidate_indices(normalized):
            alias_normalized, alias_qid, original_alias = self._aliases[index]
            score = difflib.SequenceMatcher(None, normalized, alias_normalized).ratio()
            if score > best_score:
                best_qid, best_alias, best_score = alias_qid, original_alias, score

        if best_qid is not None and best_score >= FUZZY_THRESHOLD:
            return LinkResult(qid=best_qid, matched_alias=best_alias, method="fuzzy", score=best_score)

        return LinkResult(qid=None, method="none", score=best_score)

    def link_entities(self, entities: List[KGEntity]) -> List[Tuple[KGEntity, LinkResult]]:
        """Return (linked_entity, LinkResult) pairs — the caller gets both
        the updated KGEntity (wikidata_qid filled where matched) and the
        match metadata (method/score) in one pass, without re-linking.

        Same bigram-indexed candidate generation as link(), just batched —
        the candidate set per entity is small enough at this point that a
        straightforward per-entity loop (not a further alias-outer
        restructure) is fine; the bigram index is what made this tractable
        at this corpus's real scale, not the loop order.
        """
        results = []
        for entity in entities:
            result = self.link(entity.canonical_name)
            linked_entity = entity.model_copy(update={"wikidata_qid": result.qid})
            results.append((linked_entity, result))
        return results

    @staticmethod
    def _window(length: int) -> Tuple[int, int]:
        """Alias-length bounds within which FUZZY_THRESHOLD is reachable at
        all, from difflib's own ratio() bound: ratio() = 2*M/(la+lb) with
        M <= min(la, lb), so ratio()'s maximum possible value is fixed
        purely by the two lengths. This half of the prune is lossless —
        unlike the bigram filter above, it can never cause a false negative.
        """
        if length == 0:
            return (1, 0)  # empty range — nothing can match a zero-length name
        return (
            math.ceil(length * FUZZY_THRESHOLD / (2 - FUZZY_THRESHOLD)),
            math.floor(length * (2 - FUZZY_THRESHOLD) / FUZZY_THRESHOLD),
        )
