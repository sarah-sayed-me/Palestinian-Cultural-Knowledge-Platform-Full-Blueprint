"""Near-duplicate detection using MinHash and Locality-Sensitive Hashing."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

try:
    from datasketch import MinHash, MinHashLSH

    _HAS_DATASKETCH = True
except ImportError:
    _HAS_DATASKETCH = False

NUM_PERM = 128
SHINGLE_SIZE = 3


@dataclass
class DupResult:
    is_duplicate: bool
    canonical_id: Optional[str] = None


class DuplicationIndex:
    """Maintains an in-memory LSH index for one pipeline run."""

    def __init__(self, threshold: float = 0.80, num_perm: int = NUM_PERM):
        self.threshold = threshold
        self.num_perm = num_perm
        self._lsh = MinHashLSH(threshold=threshold, num_perm=num_perm) if _HAS_DATASKETCH else None
        self._signatures: dict[str, MinHash] = {}
        self._fallback_shingles: dict[str, set[bytes]] = {}
        self._count = 0
        self._duplicate_count = 0

    def check_and_register(self, doc_id: str, text: str) -> DupResult:
        """Check whether text is a near-duplicate, registering unique docs."""
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
