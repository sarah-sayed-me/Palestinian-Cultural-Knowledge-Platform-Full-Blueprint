"""
LLM-prompted relation extraction (Track E3).

Reuses the same Ollama model/config as the RAG generator (src/rag/generator.py,
configs/rag.yaml's `generation` block) per ROADMAP.md's instruction to use
"the Track B/C generator" — same model, same host resolution logic — rather
than standing up a second local-LLM dependency. The prompt/parsing shape is
different from RAG's cited-answer generation (structured JSON extraction over
one sentence + an entity pair, not a Q&A over retrieved chunks), so this is
its own thin client rather than reusing OllamaGenerator.generate() directly.

Entity pairs are drawn from sentence-level co-occurrence: two entities
mentioned in the same sentence are candidates for a relation. Sentence
boundaries reuse entity_extractor.split_sentences() — the same segmentation
NER already used to produce the `positions` field on each entity mention
(see src/ingestion/entity_extractor.py), so sentence_index here always
agrees with what's already stored on disk; no re-derivation drift.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple

from src.ingestion.entity_extractor import Sentence, split_sentences
from src.knowlegde_graph.schemas import KGRelation, make_entity_id, make_relation_id
from src.rag.config import GenerationConfig

logger = logging.getLogger(__name__)

# Bounds LLM calls per sentence: C(4, 2) = 6 pairs at most. A sentence
# mentioning more than 4 distinct entities is rare and, when it happens, is
# usually a list/enumeration rather than a sentence expressing real pairwise
# relations — not worth the extra calls.
MAX_ENTITIES_PER_SENTENCE = 4
MIN_CONFIDENCE = 0.5

SYSTEM_PROMPT = """You are a relation-extraction assistant. Given one sentence and two \
entities mentioned in it, decide whether the sentence states a clear relationship between \
them. Respond with ONLY a JSON object, no other text:
{"predicate": "<short_snake_case_relation_or_null>", "confidence": <0.0-1.0>}

Rules:
- predicate must be a short, general, reusable snake_case relation label (e.g. located_in, \
part_of, known_for, born_in, produced_in, member_of, capital_of, occurred_in), not a full \
sentence.
- If the sentence does not clearly state a relation between exactly these two entities, \
respond {"predicate": null, "confidence": 0.0}.
- Never invent information not stated in the sentence."""

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_NULL_PREDICATES = {"none", "null", "no_relation", "n/a"}


@dataclass
class EntityMention:
    entity_id: str
    canonical_name: str
    type: str
    surface: str
    start_char: int


def _sentence_entity_mentions(doc: Dict[str, Any]) -> Dict[int, List[EntityMention]]:
    """Group this document's entity mentions by sentence_index, one
    EntityMention per DISTINCT entity per sentence (first occurrence wins if
    the same entity is mentioned more than once in one sentence)."""
    by_sentence: Dict[int, Dict[str, EntityMention]] = {}
    for entity in doc.get("entities", []):
        normalized = entity.get("normalized")
        entity_type = entity.get("type")
        if not normalized or not entity_type:
            continue
        entity_id = make_entity_id(normalized, entity_type)
        for position in entity.get("positions", []):
            sentence_index = position.get("sentence_index")
            if sentence_index is None:
                continue
            bucket = by_sentence.setdefault(sentence_index, {})
            if entity_id not in bucket:
                bucket[entity_id] = EntityMention(
                    entity_id=entity_id,
                    canonical_name=entity.get("canonical") or entity.get("text", normalized),
                    type=entity_type,
                    surface=entity.get("text", normalized),
                    start_char=position.get("start_char", 0),
                )
    return {idx: sorted(mentions.values(), key=lambda m: m.start_char) for idx, mentions in by_sentence.items()}


def candidate_pairs(
    doc: Dict[str, Any], sentences: List[Sentence]
) -> Iterable[Tuple[Sentence, EntityMention, EntityMention]]:
    """Yield (sentence, subject, object) candidates for every sentence with
    2+ distinct co-occurring entities. Text order decides subject/object:
    the earlier-appearing entity is the subject."""
    sentence_by_index = {s.index: s for s in sentences}
    for sentence_index, mentions in _sentence_entity_mentions(doc).items():
        if len(mentions) < 2:
            continue
        sentence = sentence_by_index.get(sentence_index)
        if sentence is None:
            continue
        capped = mentions[:MAX_ENTITIES_PER_SENTENCE]
        for i in range(len(capped)):
            for j in range(i + 1, len(capped)):
                yield sentence, capped[i], capped[j]


def parse_llm_response(raw: str) -> Optional[Tuple[str, float]]:
    """Extract (predicate, confidence) from a raw LLM response, tolerating
    stray text/markdown fences around the JSON object. Returns None if no
    usable JSON was found."""
    match = _JSON_RE.search(raw)
    if not match:
        return None
    try:
        payload = json.loads(match.group())
    except json.JSONDecodeError:
        return None
    predicate = payload.get("predicate")
    if not predicate or not isinstance(predicate, str):
        return None
    try:
        confidence = float(payload.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    return predicate.strip().lower(), confidence


class OllamaRelationExtractor:
    def __init__(self, config: Optional[GenerationConfig] = None, host: Optional[str] = None):
        import os

        import ollama

        self.config = config or GenerationConfig()
        resolved_host = host or os.environ.get(self.config.host_env_var, "http://localhost:11434")
        self.client = ollama.Client(host=resolved_host)

    def _call_llm(self, sentence_text: str, subject: EntityMention, object_: EntityMention) -> str:
        user_message = (
            f'Sentence: "{sentence_text}"\n'
            f'Entity 1 (subject): "{subject.surface}" (type: {subject.type})\n'
            f'Entity 2 (object): "{object_.surface}" (type: {object_.type})'
        )
        response = self.client.chat(
            model=self.config.model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            # qwen3 (this project's default model, configs/rag.yaml) is a
            # hybrid "thinking" model that emits a <think>...</think> block
            # before its answer unless told not to — verified directly: with
            # think left at its default, a 100-token budget was consumed
            # entirely by the thinking preamble and returned empty content.
            # think=False (supported by Ollama for hybrid-thinking models)
            # skips that block; confirmed this drops per-call latency from
            # 7s-with-empty-output to ~1.4s-with-a-real-answer.
            think=False,
            options={"temperature": 0.0, "num_predict": 150},
        )
        return response.message.content or ""

    def extract_relation(
        self, doc_id: str, sentence: Sentence, subject: EntityMention, object_: EntityMention
    ) -> Optional[KGRelation]:
        raw = self._call_llm(sentence.text, subject, object_)
        parsed = parse_llm_response(raw)
        if parsed is None:
            return None
        predicate, confidence = parsed
        if predicate in _NULL_PREDICATES or confidence < MIN_CONFIDENCE:
            return None
        return KGRelation(
            relation_id=make_relation_id(subject.entity_id, predicate, object_.entity_id, doc_id),
            subject_entity_id=subject.entity_id,
            predicate=predicate,
            object_entity_id=object_.entity_id,
            confidence=confidence,
            source_doc_id=doc_id,
            evidence_sentence=sentence.text,
        )

    def extract_document(self, doc: Dict[str, Any], *, max_pairs: Optional[int] = None) -> List[KGRelation]:
        text = doc.get("text", "")
        doc_id = doc.get("doc_id")
        if not text or not doc_id:
            return []
        sentences = split_sentences(text)
        relations: List[KGRelation] = []
        for count, (sentence, subject, obj) in enumerate(candidate_pairs(doc, sentences)):
            if max_pairs is not None and count >= max_pairs:
                break
            relation = self.extract_relation(doc_id, sentence, subject, obj)
            if relation is not None:
                relations.append(relation)
        return relations
