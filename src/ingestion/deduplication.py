"""Near-duplicate detection using MinHash and Locality-Sensitive Hashing."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False

logger = logging.getLogger(__name__)

NUM_PERM = 128
SHINGLE_SIZE = 3


@dataclass
class DupResult:
    is_duplicate: bool
    canonical_id: Optional[str] = None


class DuplicationIndex:
    """Maintains an in-memory LSH index for one pipeline run."""

    def __init__(self, threshold: float = 0.70, num_perm: int = NUM_PERM):
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if _HAS_DATASKETCH else None
        self._signatures: dict[str, MinHash] = {}
        self._fallback_shingles: dict[str, set[bytes]] = {}
        self._count = 0
        self._duplicate_count = 0

    def check_and_register(self, doc_id: str, text: str, source_id: Optional[str] = None) -> DupResult:
        """Check whether text is a near-duplicate, registering unique docs.

        source_id is accepted-but-ignored here so callers (pipeline.py) can
        treat DuplicationIndex and PersistentDuplicationIndex identically —
        only the persistent subclass actually uses it (to tag which source
        first collected a document).
        """
        if doc_id in self._signatures or doc_id in self._fallback_shingles:
            # The same doc_id was already indexed (e.g. the same URL
            # re-collected while live-page content like ads/related-links
            # shifted slightly). doc_id equality is a stronger, deterministic
            # duplicate signal than the LSH/Jaccard similarity check below,
            # which is probabilistic and can miss this case — and re-inserting
            # an existing key into the LSH raises, rather than just missing a
            # duplicate, so this must be checked first.
            self._count += 1
            self._duplicate_count += 1
            return DupResult(is_duplicate=True, canonical_id=doc_id)

        if not _HAS_DATASKETCH:
            return self._check_and_register_fallback(doc_id, text)

        minhash = self._compute_minhash(text)
        assert self._lsh is not None
        candidates = self._lsh.query(minhash)
        self._count += 1

        if candidates:
            self._duplicate_count += 1
            return DupResult(is_duplicate=True, canonical_id=sorted(candidates)[0])

        self._lsh.insert(doc_id, minhash)
        self._signatures[doc_id] = minhash
        return DupResult(is_duplicate=False)

    def _compute_minhash(self, text: str) -> "MinHash":
        minhash = MinHash(num_perm=self.num_perm)
        if len(text) < SHINGLE_SIZE:
            shingles = {text.encode("utf-8")}
        else:
            shingles = {
                text[i : i + SHINGLE_SIZE].encode("utf-8")
                for i in range(len(text) - SHINGLE_SIZE + 1)
            }
        for shingle in shingles:
            minhash.update(shingle)
        return minhash

    def _check_and_register_fallback(self, doc_id: str, text: str) -> DupResult:
        shingles = self._compute_shingles(text)
        self._count += 1
        for candidate_id, candidate_shingles in self._fallback_shingles.items():
            similarity = _jaccard(shingles, candidate_shingles)
            if similarity >= self.threshold:
                self._duplicate_count += 1
                return DupResult(is_duplicate=True, canonical_id=candidate_id)
        self._fallback_shingles[doc_id] = shingles
        return DupResult(is_duplicate=False)

    def _compute_shingles(self, text: str) -> set[bytes]:
        if len(text) < SHINGLE_SIZE:
            return {text.encode("utf-8")}
        return {
            text[i : i + SHINGLE_SIZE].encode("utf-8")
            for i in range(len(text) - SHINGLE_SIZE + 1)
        }

    def stats(self) -> dict:
        return {
            "total_checked": self._count,
            "duplicates_found": self._duplicate_count,
            "unique_documents": self._count - self._duplicate_count,
            "threshold": self.threshold,
            "backend": "datasketch" if _HAS_DATASKETCH else "python_jaccard",
        }


def _jaccard(left: set[bytes], right: set[bytes]) -> float:
    if not left and not right:
        return 1.0
    union = left | right
    if not union:
        return 0.0
    return len(left & right) / len(union)


class PersistentDuplicationIndex(DuplicationIndex):
    """A DuplicationIndex that loads existing signatures from Postgres on
    construction and persists new ones back, so near-duplicate detection works
    ACROSS pipeline runs and sources — not just within one run, which was the
    gap ROADMAP.md Track D6 flagged: the plain DuplicationIndex is rebuilt from
    scratch every run, so re-running (or adding a second source) couldn't catch
    a document already collected.

    Degrades to in-memory-only (a logged warning, not silent data loss) if
    Postgres isn't reachable, so ingestion doesn't hard-depend on the RAG
    vector store being up — cross-run dedup is degraded, not broken.
    """

    TABLE = "ingestion_dedup_index"

    def __init__(self, threshold: float = 0.70, num_perm: int = NUM_PERM, conn=None):
        super().__init__(threshold=threshold, num_perm=num_perm)
        self._conn = conn
        self._persistent = conn is not None and _HAS_DATASKETCH
        if conn is not None and not _HAS_DATASKETCH:
            logger.warning(
                "datasketch is not installed; persistent dedup requires it (the "
                "shingle fallback has no serializable signature). Falling back "
                "to in-memory-only, same as a plain DuplicationIndex."
            )
        if self._persistent:
            self._create_schema()
            self._load_existing()

    def _create_schema(self) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {self.TABLE} (
                    doc_id TEXT PRIMARY KEY,
                    source_id TEXT,
                    minhash_seed BIGINT NOT NULL,
                    minhash_values BYTEA NOT NULL,
                    first_seen_at TIMESTAMPTZ DEFAULT now()
                );
                """
            )
        self._conn.commit()

    def _load_existing(self) -> None:
        import numpy as np

        with self._conn.cursor() as cur:
            cur.execute(f"SELECT doc_id, minhash_seed, minhash_values FROM {self.TABLE};")
            rows = cur.fetchall()
        for doc_id, seed, values_bytes in rows:
            values = np.frombuffer(bytes(values_bytes), dtype=np.uint64)
            minhash = MinHash(num_perm=self.num_perm, seed=seed, hashvalues=values)
            self._lsh.insert(doc_id, minhash)
            self._signatures[doc_id] = minhash
        if rows:
            logger.info("Loaded %d existing dedup signatures from %s.", len(rows), self.TABLE)

    def check_and_register(self, doc_id: str, text: str, source_id: Optional[str] = None) -> DupResult:
        result = super().check_and_register(doc_id, text)
        if self._persistent and not result.is_duplicate:
            self._persist(doc_id, source_id)
        return result

    def _persist(self, doc_id: str, source_id: Optional[str]) -> None:
        minhash = self._signatures[doc_id]
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                INSERT INTO {self.TABLE} (doc_id, source_id, minhash_seed, minhash_values)
                VALUES (%s, %s, %s, %s)
                ON CONFLICT (doc_id) DO NOTHING;
                """,
                (doc_id, source_id, int(minhash.seed), minhash.hashvalues.tobytes()),
            )
        self._conn.commit()

    def stats(self) -> dict:
        base = super().stats()
        base["persistent"] = self._persistent
        return base
