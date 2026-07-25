"""
Named Entity Recognition for the Palestinian Cultural Knowledge Platform.

Two sources are combined per document:

  1. CAMeL Tools NER      -> general entities: PERSON, LOCATION, ORGANIZATION, MISC
  2. Heritage dictionary   -> Palestinian cultural entities (food, clothing,
                              craft, heritage practice, place, plant), defined
                              in configs/heritage_entities.yaml

Design goals (see project roadmap, Phase 7 — NER + KG):
  - multi-word entities with correct character offsets
  - sentence-level indexing
  - confidence scores (1.0 for exact dictionary matches; CAMeL score if the
    installed CAMeL Tools version exposes one, otherwise a documented
    placeholder — CAMeL's public NER API does not return per-token
    probabilities)
  - de-duplicated entities per document, with mention counts + positions
    instead of repeated identical entries
  - a single lazily-loaded CAMeL model shared across the whole run (not
    reloaded per document), batched inference across many sentences at once
  - best-effort GPU placement, silent CPU fallback

This module intentionally does NOT do relation extraction, entity linking,
or model fine-tuning — those are later pipeline phases.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import yaml

from src.preprocessing.arabic_normalizer import normalize_arabic

logger = logging.getLogger(__name__)

DEFAULT_HERITAGE_CONFIG = Path("configs/heritage_entities.yaml")

# CAMeL Tools B-I-O label prefixes we normalise into our own entity types.
_CAMEL_TYPE_MAP = {
    "PERS": "PERSON",
    "PER": "PERSON",
    "LOC": "LOCATION",
    "ORG": "ORGANIZATION",
    "MISC": "MISC",
}

# Suffixes stripped/added when generating dictionary variants and when
# normalising candidate tokens at match time. This is a light heuristic —
# NOT full morphological analysis. Broken plurals (e.g. ثوب -> أثواب) are
# NOT covered; that requires CAMeL morphology (deferred to a later phase,
# see project roadmap).
_PLURAL_SUFFIXES = ("ات", "ية", "ين", "ون", "ة")

# Common single/double-letter proclitics (conjunction "و", preposition
# "ف/ب/ل/ك", and combinations with the definite article) that can attach to
# the *first* word of a phrase in running text, e.g. "ودبس التمر" (and the
# date molasses), "بالكوفية", "كالمقلوبة". Stripped at match time only
# (candidate side), longest first so "وال"/"فال" aren't mistaken for "و"/"ف".
_PROCLITIC_PREFIXES = ("وال", "فال", "بال", "كال", "لل", "و", "ف", "ب", "ل", "ك")


# ---------------------------------------------------------------------------
# Shared data structures
# ---------------------------------------------------------------------------


@dataclass
class Sentence:
    index: int
    text: str
    start: int  # char offset into the full document text
    end: int


@dataclass
class Token:
    text: str
    start: int  # char offset into the full document text
    end: int


@dataclass
class RawMention:
    text: str
    normalized: str
    entity_type: str
    source: str  # "camel_ner" | "heritage_dictionary"
    start_char: int
    end_char: int
    sentence_index: int
    confidence: float
    canonical: Optional[str] = None


# ---------------------------------------------------------------------------
# Sentence / token segmentation (shared by both extractors so offsets and
# CAMeL input tokens always agree with each other).
# ---------------------------------------------------------------------------

_SENTENCE_RE = re.compile(r"[^.!؟؛\n]+(?:[.!؟؛]+|\n+|$)", re.UNICODE)
_TOKEN_RE = re.compile(r"[\u0600-\u06FFA-Za-z0-9]+|[^\s\u0600-\u06FFA-Za-z0-9]", re.UNICODE)


def split_sentences(text: str) -> List[Sentence]:
    """Lightweight sentence segmentation with character offsets.

    Deliberately dependency-free (no punkt/pysbd model download) — good
    enough for sentence *indexing*, which is all the entities field needs.
    """
    sentences: List[Sentence] = []
    for match in _SENTENCE_RE.finditer(text):
        raw = match.group()
        stripped = raw.strip()
        if not stripped:
            continue
        leading_ws = len(raw) - len(raw.lstrip())
        start = match.start() + leading_ws
        end = start + len(stripped)
        sentences.append(Sentence(index=len(sentences), text=stripped, start=start, end=end))
    return sentences


def tokenize_with_offsets(sentence: Sentence) -> List[Token]:
    """Tokenize a sentence, returning tokens with absolute document offsets.

    We use our own tokenizer (rather than CAMeL's `simple_word_tokenize`)
    specifically so every token carries an exact character span — CAMeL's
    tokenizer does not return offsets, and re-aligning its output against
    the original string is error-prone. Any consistent word/punctuation
    tokenization is valid input for CAMeL's sequence tagger.
    """
    tokens = []
    for match in _TOKEN_RE.finditer(sentence.text):
        tokens.append(
            Token(
                text=match.group(),
                start=sentence.start + match.start(),
                end=sentence.start + match.end(),
            )
        )
    return tokens


# ---------------------------------------------------------------------------
# Heritage dictionary matcher
# ---------------------------------------------------------------------------


@dataclass
class HeritageEntry:
    canonical: str
    category: str


class HeritageMatcher:
    """Multi-word dictionary matcher for Palestinian cultural entities.

    Handles common Arabic surface variation automatically:
      - definite article "ال" attached to the first word, the last word, or
        both (covers both adjective-noun phrases like "الصابون النابلسي"
        and idafa/construct phrases like "ماء الورد", "دبس التمر")
      - a small set of plural/nisba suffixes on the last word
      - the existing `normalize_arabic()` normalization (alef/teh-marbuta
        unification, diacritic removal) already used by the ingestion
        pipeline, so matching is consistent with how documents are cleaned
    """

    def __init__(self, entries: Sequence[HeritageEntry]):
        self._index: Dict[Tuple[str, ...], HeritageEntry] = {}
        self._max_len = 1
        for entry in entries:
            for variant in self._generate_variants(entry.canonical):
                self._index[variant] = entry
                self._max_len = max(self._max_len, len(variant))

    @classmethod
    def from_yaml(cls, path: Path = DEFAULT_HERITAGE_CONFIG) -> "HeritageMatcher":
        entries = load_heritage_entries(path)
        return cls(entries)

    # -- variant generation (dictionary side) --------------------------------

    @staticmethod
    def _with_article(word: str) -> str:
        return word if word.startswith("ال") else "ال" + word

    @staticmethod
    def _strip_suffix(word: str) -> Optional[str]:
        for suf in _PLURAL_SUFFIXES:
            if word.endswith(suf) and len(word) > len(suf) + 1:
                return word[: -len(suf)]
        return None

    def _generate_variants(self, canonical: str) -> set[Tuple[str, ...]]:
        base_words = tuple(normalize_arabic(w) for w in canonical.split())
        variants: set[Tuple[str, ...]] = {base_words}

        if len(base_words) == 1:
            variants.add((self._with_article(base_words[0]),))
        else:
            first, last = base_words[0], base_words[-1]
            middle = base_words[1:-1]
            for f, l in (
                (self._with_article(first), last),
                (first, self._with_article(last)),
                (self._with_article(first), self._with_article(last)),
            ):
                variants.add((f,) + middle + (l,))

        # Suffix variants on the last word of every variant generated so far.
        for variant in list(variants):
            last_word = variant[-1]
            for suf in _PLURAL_SUFFIXES:
                if not last_word.endswith(suf):
                    variants.add(variant[:-1] + (last_word + suf,))
        return variants

    # -- matching (document side) --------------------------------------------

    def _normalize_token(self, token_text: str) -> str:
        return normalize_arabic(token_text)

    @staticmethod
    def _strip_proclitic(word: str) -> Optional[str]:
        """Strip a leading conjunction/preposition proclitic (و، ف، ب، ل، ك
        and combinations with "ال"), e.g. "ودبس" -> "دبس", "بالكوفية" ->
        "كوفية". Conservative: only strips if a reasonable stem remains.
        """
        for prefix in _PROCLITIC_PREFIXES:
            if word.startswith(prefix) and len(word) - len(prefix) >= 2:
                return word[len(prefix):]
        return None

    def _candidate_lookup(self, candidate: Tuple[str, ...]) -> Optional[HeritageEntry]:
        """Try `candidate` as-is, then with a proclitic stripped off the
        first word, then with a plural-suffix stripped off the last word
        (and both combined). These are cheap dict-lookup probes of the
        candidate the document actually contains — not new dictionary
        entries.
        """
        entry = self._index.get(candidate)
        if entry is not None:
            return entry

        variants_to_try: List[Tuple[str, ...]] = [candidate]
        stripped_first = self._strip_proclitic(candidate[0]) if candidate else None
        if stripped_first is not None:
            variants_to_try.append((stripped_first,) + candidate[1:])

        tried = {candidate}
        for variant in list(variants_to_try):
            if len(variant) > 1:
                stripped_last = self._strip_suffix(variant[-1])
                if stripped_last is not None:
                    variants_to_try.append(variant[:-1] + (stripped_last,))
            elif len(variant) == 1:
                stripped_last = self._strip_suffix(variant[0])
                if stripped_last is not None:
                    variants_to_try.append((stripped_last,))

        for variant in variants_to_try:
            if variant in tried:
                continue
            tried.add(variant)
            entry = self._index.get(variant)
            if entry is not None:
                return entry
        return None

    def find_mentions(
        self, tokens: List[Token], sentence_index: int
    ) -> Iterable[RawMention]:
        """Greedy longest-match-first scan over one sentence's tokens."""
        normalized = [self._normalize_token(t.text) for t in tokens]
        i = 0
        n = len(tokens)
        while i < n:
            matched = False
            max_span = min(self._max_len, n - i)
            for span in range(max_span, 0, -1):
                candidate = tuple(normalized[i : i + span])
                entry = self._candidate_lookup(candidate)

                if entry is not None:
                    start_tok, end_tok = tokens[i], tokens[i + span - 1]
                    yield RawMention(
                        text=" ".join(t.text for t in tokens[i : i + span]),
                        normalized=" ".join(candidate),
                        entity_type=f"HERITAGE_{entry.category.upper()}",
                        source="heritage_dictionary",
                        start_char=start_tok.start,
                        end_char=end_tok.end,
                        sentence_index=sentence_index,
                        confidence=1.0,
                        canonical=entry.canonical,
                    )
                    i += span
                    matched = True
                    break
            if not matched:
                i += 1


def load_heritage_entries(path: Path = DEFAULT_HERITAGE_CONFIG) -> List[HeritageEntry]:
    with Path(path).open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    entries: List[HeritageEntry] = []
    for category, items in raw.items():
        for item in items or []:
            entries.append(HeritageEntry(canonical=item["canonical"], category=category))
            for variant in item.get("variants", []) or []:
                entries.append(HeritageEntry(canonical=variant, category=category))
    return entries


# ---------------------------------------------------------------------------
# CAMeL Tools NER backend — lazy singleton, batched, best-effort GPU
# ---------------------------------------------------------------------------


class CamelNERBackend:
    """Thin wrapper around camel_tools.ner.NERecognizer.

    Loaded once per process (the model load is the expensive part — several
    seconds to tens of seconds depending on the backend). Call `predict_batch`
    with many tokenized sentences at once; do not construct a new instance
    per document.

    GPU note: CAMeL Tools does not expose a stable, version-independent
    public API for device placement. Recent camel-tools releases (>=1.5)
    back NERecognizer with a transformers `AutoModelForTokenClassification`,
    which *can* run on GPU — but older releases used an sklearn-crfsuite /
    BiLSTM-CRF backend that is CPU-only regardless of what we do here.
    We therefore attempt GPU placement defensively via attribute
    introspection and silently stay on CPU if that fails or if no CUDA
    device is available. Always check `backend.device` after construction
    to see what was actually used.
    """

    _singleton: Optional["CamelNERBackend"] = None

    def __init__(self, prefer_gpu: bool = True):
        self.available = False
        self.device = "cpu"
        self._recognizer = None
        self._load_error: Optional[Exception] = None

        try:
            from camel_tools.ner import NERecognizer  # noqa: WPS433 (lazy import)
        except ImportError as exc:
            self._load_error = exc
            logger.warning("camel-tools is not installed; NER will only use the heritage dictionary.")
            return

        try:
            self._recognizer = NERecognizer.pretrained()
        except Exception as exc:  # pretrained model not downloaded, etc.
            self._load_error = exc
            logger.warning("Failed to load CAMeL Tools NER model: %s", exc)
            return

        self.available = True
        if prefer_gpu:
            self._try_move_to_gpu()

    def _try_move_to_gpu(self) -> None:
        try:
            import torch
        except ImportError:
            return
        if not torch.cuda.is_available():
            return
        for attr in ("model", "_model", "tagger", "_tagger"):
            candidate = getattr(self._recognizer, attr, None)
            if candidate is not None and hasattr(candidate, "to"):
                try:
                    candidate.to("cuda")
                    self.device = "cuda"
                    logger.info("CAMeL NER model moved to CUDA via attribute '%s'.", attr)
                except Exception as exc:  # pragma: no cover - hardware dependent
                    logger.warning("Could not move CAMeL NER model to CUDA: %s", exc)
                return

    @classmethod
    def get(cls, prefer_gpu: bool = True) -> "CamelNERBackend":
        if cls._singleton is None:
            cls._singleton = cls(prefer_gpu=prefer_gpu)
        return cls._singleton

    def predict_batch(self, tokenized_sentences: List[List[str]]) -> List[List[str]]:
        """Tag many pre-tokenized sentences in one call.

        Falls back to per-sentence calls only if the installed camel-tools
        version doesn't support batch prediction (older API).
        """
        if not self.available or not tokenized_sentences:
            return [["O"] * len(sent) for sent in tokenized_sentences]
        try:
            return self._recognizer.predict(tokenized_sentences)
        except TypeError:
            logger.debug("CAMeL NERecognizer.predict does not support batching in this version; "
                         "falling back to per-sentence calls.")
            return [self._recognizer.predict_sentence(sent) for sent in tokenized_sentences]


def _spans_from_bio(tokens: List[Token], tags: List[str], sentence_index: int) -> Iterable[RawMention]:
    """Group consecutive B-/I- tags of the same type into entity spans."""
    i = 0
    n = len(tags)
    while i < n:
        tag = tags[i]
        if tag == "O" or "-" not in tag:
            i += 1
            continue
        prefix, raw_type = tag.split("-", 1)
        if prefix != "B":
            # Stray I- tag with no preceding B- (malformed sequence) — skip.
            i += 1
            continue
        j = i + 1
        while j < n and tags[j] == f"I-{raw_type}":
            j += 1
        entity_type = _CAMEL_TYPE_MAP.get(raw_type.upper(), raw_type.upper())
        start_tok, end_tok = tokens[i], tokens[j - 1]
        surface = " ".join(t.text for t in tokens[i:j])
        yield RawMention(
            text=surface,
            normalized=normalize_arabic(surface),
            entity_type=entity_type,
            source="camel_ner",
            start_char=start_tok.start,
            end_char=end_tok.end,
            sentence_index=sentence_index,
            # CAMeL Tools' public NER API returns labels only, not
            # probabilities, in any released version at time of writing.
            # 0.75 is a documented placeholder, not a measured score.
            confidence=0.75,
        )
        i = j


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class EntityExtractor:
    """Runs both entity sources over one or many documents and aggregates results."""

    def __init__(
        self,
        heritage_matcher: Optional[HeritageMatcher] = None,
        camel_backend: Optional[CamelNERBackend] = None,
        use_camel: bool = True,
    ):
        self.heritage_matcher = heritage_matcher or HeritageMatcher.from_yaml()
        self.use_camel = use_camel
        self.camel_backend = camel_backend or (CamelNERBackend.get() if use_camel else None)

    # -- single document -----------------------------------------------------

    def extract(self, text: str) -> List[Dict[str, Any]]:
        """Extract de-duplicated, aggregated entities for one document's text."""
        return self._aggregate(self._extract_raw(text))

    # -- batched, many documents at once -------------------------------------

    def extract_many(self, texts: Sequence[str]) -> List[List[Dict[str, Any]]]:
        """Extract entities for many documents, batching CAMeL inference
        across ALL sentences of ALL documents in one call. This is the
        primary throughput win over per-document / per-sentence inference:
        one padding/forward-pass batch instead of N small ones.
        """
        per_doc_sentences: List[List[Sentence]] = [split_sentences(t) for t in texts]
        per_doc_tokens: List[List[List[Token]]] = [
            [tokenize_with_offsets(sent) for sent in sentences] for sentences in per_doc_sentences
        ]

        flat_token_texts: List[List[str]] = []
        owner: List[Tuple[int, int]] = []  # (doc_index, sentence_position_within_doc)
        for doc_idx, sentences_tokens in enumerate(per_doc_tokens):
            for sent_pos, tokens in enumerate(sentences_tokens):
                flat_token_texts.append([t.text for t in tokens])
                owner.append((doc_idx, sent_pos))

        camel_tags = (
            self.camel_backend.predict_batch(flat_token_texts)
            if self.use_camel
            else [["O"] * len(t) for t in flat_token_texts]
        )

        raw_by_doc: List[List[RawMention]] = [[] for _ in texts]
        for flat_idx, (doc_idx, sent_pos) in enumerate(owner):
            sentence = per_doc_sentences[doc_idx][sent_pos]
            tokens = per_doc_tokens[doc_idx][sent_pos]
            raw_by_doc[doc_idx].extend(_spans_from_bio(tokens, camel_tags[flat_idx], sentence.index))
            raw_by_doc[doc_idx].extend(self.heritage_matcher.find_mentions(tokens, sentence.index))

        return [self._aggregate(mentions) for mentions in raw_by_doc]

    # -- internal --------------------------------------------------------------

    def _extract_raw(self, text: str) -> List[RawMention]:
        sentences = split_sentences(text)
        sentence_tokens = [tokenize_with_offsets(s) for s in sentences]

        camel_tags = (
            self.camel_backend.predict_batch([[t.text for t in toks] for toks in sentence_tokens])
            if self.use_camel
            else [["O"] * len(toks) for toks in sentence_tokens]
        )

        mentions: List[RawMention] = []
        for sentence, tokens, tags in zip(sentences, sentence_tokens, camel_tags):
            mentions.extend(_spans_from_bio(tokens, tags, sentence.index))
            mentions.extend(self.heritage_matcher.find_mentions(tokens, sentence.index))
        return mentions

    @staticmethod
    def _aggregate(mentions: Iterable[RawMention]) -> List[Dict[str, Any]]:
        """De-duplicate mentions of the same (normalized text, type) within
        a document into a single entity record with mention count + all
        mention positions, instead of one row per occurrence.
        """
        grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
        for m in mentions:
            key = (m.normalized, m.entity_type)
            if key not in grouped:
                grouped[key] = {
                    "text": m.text,
                    "normalized": m.normalized,
                    "type": m.entity_type,
                    "source": m.source,
                    "canonical": m.canonical,
                    "confidence": m.confidence,
                    "mention_count": 0,
                    "positions": [],
                }
            entry = grouped[key]
            entry["mention_count"] += 1
            entry["confidence"] = max(entry["confidence"], m.confidence)
            entry["positions"].append(
                {
                    "start_char": m.start_char,
                    "end_char": m.end_char,
                    "sentence_index": m.sentence_index,
                }
            )

        # Stable order: highest-confidence / most-frequent entities first.
        results = list(grouped.values())
        results.sort(key=lambda e: (-e["confidence"], -e["mention_count"], e["text"]))
        return results