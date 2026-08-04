"""Export processed ingestion JSONL to Hugging Face Datasets-ready files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, List, Union

import pandas as pd

DEFAULT_INPUT = Path("data/processed/wikipedia_ar_documents.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/hf/wikipedia_ar")


def _is_clear(record: dict[str, Any]) -> bool:
    """True if license_status == 'clear', used only when --clear-only is passed.

    This project currently exports for private research use (not public
    redistribution) — see docs/licensing_checklist.md and ROADMAP.md Track D
    for the reasoning. license_status is still recorded on every document as
    provenance, but does not gate the default export. Pass --clear-only
    if/when preparing a subset for public release.
    """
    return record.get("license_status") == "clear"


def export_to_hf_dataset(
    *,
    input_path: Union[Path, List[Path]] = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_jsonl_copy: bool = False,
    clear_only: bool = False,
) -> dict[str, Any]:
    """Convert accepted processed documents into Parquet dataset artifacts.

    Includes everything by default (private research corpus). Pass
    clear_only=True to restrict to license_status == "clear" — e.g. when
    preparing a subset intended for public release.

    input_path accepts either one file or a list — passing every source's
    file combines them into a single dataset (each record already carries
    its own source_id, so downstream users can still filter by source).
    """
    input_paths = [input_path] if isinstance(input_path, Path) else list(input_path)
    missing = [p for p in input_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Processed input(s) not found: {missing}")

    all_records: list[dict[str, Any]] = []
    for path in input_paths:
        all_records.extend(_read_jsonl(path))
    if not all_records:
        raise ValueError(f"No records found in {input_paths}")

    records = [r for r in all_records if _is_clear(r)] if clear_only else all_records
    excluded_count = len(all_records) - len(records)
    if not records:
        raise ValueError(
            f"No 'clear' records in {input_paths} ({excluded_count} excluded by --clear-only). "
            f"Drop --clear-only to export everything."
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(records)
    train_path = output_dir / "train.parquet"
    dataframe.to_parquet(train_path, index=False)

    jsonl_copy = None
    if write_jsonl_copy:
        jsonl_copy = output_dir / "train.jsonl"
        with jsonl_copy.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    by_source = dataframe["source_id"].value_counts().to_dict() if "source_id" in dataframe.columns else {}
    by_language = dataframe["language"].value_counts().to_dict() if "language" in dataframe.columns else {}
    summary = {
        "input": [str(p) for p in input_paths],
        "output_dir": str(output_dir),
        "train_parquet": str(train_path),
        "jsonl_copy": str(jsonl_copy) if jsonl_copy else None,
        "num_rows": len(records),
        "excluded_as_not_clear": excluded_count,
        "columns": list(dataframe.columns),
        "total_words": int(dataframe.get("word_count", pd.Series(dtype=int)).sum()),
        "by_source": by_source,
        "by_language": by_language,
    }
    if summary["num_rows"]:
        summary["average_words"] = round(summary["total_words"] / summary["num_rows"], 2)

    summary_path = output_dir / "dataset_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
    summary["summary_json"] = str(summary_path)
    return summary


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_number} in {path}: {exc}") from exc
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Export processed corpus to HF dataset files.")
    parser.add_argument("--input", type=Path, nargs="+", default=[DEFAULT_INPUT])
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-jsonl-copy", action="store_true")
    parser.add_argument(
        "--clear-only",
        action="store_true",
        help="Restrict to license_status == 'clear' — for preparing a public-release subset.",
    )
    args = parser.parse_args()
    summary = export_to_hf_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        write_jsonl_copy=args.write_jsonl_copy,
        clear_only=args.clear_only,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
