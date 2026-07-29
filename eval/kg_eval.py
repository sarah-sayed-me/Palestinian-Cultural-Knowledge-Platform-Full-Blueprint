"""
Knowledge graph evaluation (Track E5).

Two independent metrics, each writing its own EvalReport (same convention as
eval/ner_eval.py, eval/retrieval_eval.py, eval/rag_eval.py, so results are
comparable/dashboard-ready — see eval/schemas.py):

  - Relation-extraction precision: a hand-checked sample of real KGRelation
    records (was the predicate actually correct given its evidence_sentence?).
    Precision only, not recall — recall would require exhaustively
    gold-labeling every possible relation in the sampled documents, which
    isn't practical by hand at this stage. Same "be honest about what's
    actually measured" approach as C3/C5's citation-precision discussion.
  - Entity-linking accuracy: a hand-checked sample of KGEntity ->
    wikidata_qid links (does the QID the alias-table linker chose actually
    match the entity's real-world referent? Or, for entities with no correct
    QID in the alias dump, did the linker correctly abstain rather than
    guess wrong?).

Both gold sets are hand-built from real pipeline output — see
eval/gold/kg_relations_gold.json and eval/gold/kg_entity_linking_gold.json,
and ROADMAP.md Track E for how they were built and what real numbers they
produced.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from eval.schemas import EvalReport

DEFAULT_RELATIONS_GOLD = Path("eval/gold/kg_relations_gold.json")
DEFAULT_LINKING_GOLD = Path("eval/gold/kg_entity_linking_gold.json")
DEFAULT_OUTPUT_DIR = Path("eval_reports")


def _load_json(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def evaluate_relation_extraction(gold_path: Path = DEFAULT_RELATIONS_GOLD) -> EvalReport:
    records = _load_json(gold_path)
    if not records:
        raise ValueError(f"No gold records in {gold_path}")

    correct = sum(1 for r in records if r["is_correct"])
    by_predicate: Dict[str, Dict[str, int]] = {}
    for r in records:
        bucket = by_predicate.setdefault(r["predicate"], {"correct": 0, "total": 0})
        bucket["total"] += 1
        if r["is_correct"]:
            bucket["correct"] += 1

    return EvalReport(
        eval_name="kg_relations_v1",
        dataset_size=len(records),
        metrics={
            "precision": round(correct / len(records), 4),
            "correct": correct,
            "total": len(records),
        },
        notes=(
            "Precision only (see module docstring for why recall isn't reported). "
            f"Per-predicate breakdown: {json.dumps(by_predicate, ensure_ascii=False)}"
        ),
    )


def evaluate_entity_linking(gold_path: Path = DEFAULT_LINKING_GOLD) -> EvalReport:
    records = _load_json(gold_path)
    if not records:
        raise ValueError(f"No gold records in {gold_path}")

    # "linkable" = a correct QID genuinely exists in the alias dump for this
    # entity; accuracy is judged only over these. Entities with no correct
    # QID available are scored separately, as an abstention-rate check —
    # comparing "did it guess the right QID" against "did it correctly say
    # no QID exists" would conflate two different failure modes.
    linkable = [r for r in records if r.get("expected_qid") is not None]
    correct = sum(1 for r in linkable if r["linked_qid"] == r["expected_qid"])

    unlinkable = [r for r in records if r.get("expected_qid") is None]
    correct_abstentions = sum(1 for r in unlinkable if r["linked_qid"] is None)

    return EvalReport(
        eval_name="kg_entity_linking_v1",
        dataset_size=len(records),
        metrics={
            "accuracy_on_linkable_entities": round(correct / len(linkable), 4) if linkable else 0.0,
            "linkable_sample_size": len(linkable),
            "correct_abstention_rate": round(correct_abstentions / len(unlinkable), 4) if unlinkable else 0.0,
            "no_qid_expected_sample_size": len(unlinkable),
        },
        notes=(
            "linked_qid is what WikidataAliasLinker actually chose; expected_qid is the "
            "hand-verified correct Wikidata QID, or null if no correct entity exists in the "
            "alias dump (i.e. the linker should abstain, not guess)."
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Track E (knowledge graph) evaluation.")
    parser.add_argument("--relations-gold", type=Path, default=DEFAULT_RELATIONS_GOLD)
    parser.add_argument("--linking-gold", type=Path, default=DEFAULT_LINKING_GOLD)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    relations_report = evaluate_relation_extraction(args.relations_gold)
    linking_report = evaluate_entity_linking(args.linking_gold)

    (args.output_dir / "kg_relations_v1.json").write_text(
        json.dumps(relations_report.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (args.output_dir / "kg_entity_linking_v1.json").write_text(
        json.dumps(linking_report.to_json_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(
        json.dumps(
            {"relations": relations_report.to_json_dict(), "entity_linking": linking_report.to_json_dict()},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
