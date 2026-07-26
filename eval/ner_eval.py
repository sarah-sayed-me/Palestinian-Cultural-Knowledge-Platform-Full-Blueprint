"""
eval/ner_eval.py — compare a manually-annotated gold set against the live
EntityExtractor (CAMeL NER + heritage dictionary), and report precision /
recall / F1, both overall and per entity type.

Design choice: this script calls EntityExtractor.extract() directly on each
gold paragraph's `text`, instead of reading scripts/run_ner.py's output file.
run_ner.py works at whole-document granularity; the gold set here is at
paragraph granularity, so re-running extraction on the exact gold text is
the only way to compare apples to apples.

Usage:
    python -m eval.ner_eval \
        --gold ner_gold_clean.json \
        --output eval_reports/ner_v1.json \
        --mismatches eval_reports/ner_v1_mismatches.md \
        --no-gpu     # optional, forces CPU
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from eval.schemas import EvalReport  # noqa: E402
from src.ingestion.entity_extractor import EntityExtractor  # noqa: E402


def normalize_text(s: str) -> str:
    """Loose normalization for the 'normalized_match' metric variant only —
    NOT used for exact_match, which compares raw strings as annotated."""
    s = s.strip()
    s = re.sub(r"[\u064B-\u0652\u0670\u0640]", "", s)  # diacritics + tatweel
    s = s.replace("ة", "ه").replace("ى", "ي").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    return s


def to_pairs(entities: list[dict[str, Any]], normalized: bool) -> set[tuple[str, str]]:
    out = set()
    for e in entities:
        text = normalize_text(e["text"]) if normalized else e["text"].strip()
        out.add((text, e["type"]))
    return out


def prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    return precision, recall, f1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("eval_reports/ner_v1.json"))
    parser.add_argument("--mismatches", type=Path, default=Path("eval_reports/ner_v1_mismatches.md"))
    parser.add_argument("--no-camel", action="store_true", help="dictionary-only, skip CAMeL model (fast dry run)")
    parser.add_argument("--no-gpu", action="store_true")
    args = parser.parse_args()

    with args.gold.open(encoding="utf-8") as f:
        gold_items = json.load(f)

    extractor = EntityExtractor(use_camel=not args.no_camel)

    # overall + per-type counters, for both matching strategies
    overall = {"exact": [0, 0, 0], "normalized": [0, 0, 0]}  # [tp, fp, fn]
    per_type: dict[str, dict[str, list[int]]] = defaultdict(
        lambda: {"exact": [0, 0, 0], "normalized": [0, 0, 0]}
    )
    mismatch_lines: list[str] = []

    for item in gold_items:
        text_id = item["text_id"]
        text = item["text"]
        gold_entities = item.get("gold_entities", [])

        system_entities = extractor.extract(text)

        for mode in ("exact", "normalized"):
            gold_pairs = to_pairs(gold_entities, normalized=(mode == "normalized"))
            sys_pairs = to_pairs(system_entities, normalized=(mode == "normalized"))

            tp_pairs = gold_pairs & sys_pairs
            fp_pairs = sys_pairs - gold_pairs
            fn_pairs = gold_pairs - sys_pairs

            overall[mode][0] += len(tp_pairs)
            overall[mode][1] += len(fp_pairs)
            overall[mode][2] += len(fn_pairs)

            for _, etype in tp_pairs:
                per_type[etype][mode][0] += 1
            for _, etype in fp_pairs:
                per_type[etype][mode][1] += 1
            for _, etype in fn_pairs:
                per_type[etype][mode][2] += 1

            if mode == "exact" and (fp_pairs or fn_pairs):
                if fp_pairs:
                    mismatch_lines.append(
                        f"- [{text_id}] النظام طلّع دول غلط (False Positive): "
                        + ", ".join(f'"{t}"({ty})' for t, ty in sorted(fp_pairs))
                    )
                if fn_pairs:
                    mismatch_lines.append(
                        f"- [{text_id}] النظام فوّت دول (False Negative): "
                        + ", ".join(f'"{t}"({ty})' for t, ty in sorted(fn_pairs))
                    )

    metrics: dict[str, float] = {}
    for mode in ("exact", "normalized"):
        tp, fp, fn = overall[mode]
        p, r, f1 = prf(tp, fp, fn)
        metrics[f"precision_{mode}"] = round(p, 4)
        metrics[f"recall_{mode}"] = round(r, 4)
        metrics[f"f1_{mode}"] = round(f1, 4)
        metrics[f"tp_{mode}"] = tp
        metrics[f"fp_{mode}"] = fp
        metrics[f"fn_{mode}"] = fn

    per_type_summary = {}
    for etype, modes in per_type.items():
        tp, fp, fn = modes["exact"]
        p, r, f1 = prf(tp, fp, fn)
        per_type_summary[etype] = {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4), "support": tp + fn}

    report = EvalReport(
        eval_name="ner_v1",
        dataset_size=len(gold_items),
        metrics=metrics,
        notes=(
            "exact = raw string match (as annotated). normalized = diacritics/alef/teh-marbuta-insensitive match. "
            "Per-type breakdown (exact match) written alongside this report."
        ),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        json.dump({"report": report.to_json_dict(), "per_type": per_type_summary}, f, ensure_ascii=False, indent=2)

    args.mismatches.parent.mkdir(parents=True, exist_ok=True)
    with args.mismatches.open("w", encoding="utf-8") as f:
        f.write("# تقرير الفروق بين الـ Gold Set ومخرجات النظام (exact match)\n\n")
        f.write(f"عدد الفقرات: {len(gold_items)}\n\n")
        f.write("---\n\n")
        f.write("\n".join(mismatch_lines))

    print(json.dumps(report.to_json_dict(), ensure_ascii=False, indent=2))
    print("\nPer-type breakdown:")
    for etype, m in sorted(per_type_summary.items()):
        print(f"  {etype}: precision={m['precision']} recall={m['recall']} f1={m['f1']} (support={m['support']})")
    print(f"\nSaved report: {args.output}")
    print(f"Saved mismatches: {args.mismatches}")


if __name__ == "__main__":
    main()
