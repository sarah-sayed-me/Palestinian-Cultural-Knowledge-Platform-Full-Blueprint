"""Export processed ingestion JSONL to Hugging Face Datasets-ready files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_INPUT = Path("data/processed/wikipedia_ar_documents.jsonl")
DEFAULT_OUTPUT_DIR = Path("data/processed/hf/wikipedia_ar")


def export_to_hf_dataset(
    *,
    input_path: Path = DEFAULT_INPUT,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    write_jsonl_copy: bool = False,
) -> dict[str, Any]:
    """Convert accepted processed documents into Parquet dataset artifacts."""
    if not input_path.exists():
        raise FileNotFoundError(f"Processed input not found: {input_path}")

    records = _read_jsonl(input_path)
    if not records:
        raise ValueError(f"No records found in {input_path}")

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

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "train_parquet": str(train_path),
        "jsonl_copy": str(jsonl_copy) if jsonl_copy else None,
        "num_rows": len(records),
        "columns": list(dataframe.columns),
        "total_words": int(dataframe.get("word_count", pd.Series(dtype=int)).sum()),
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
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--write-jsonl-copy", action="store_true")
    args = parser.parse_args()
    summary = export_to_hf_dataset(
        input_path=args.input,
        output_dir=args.output_dir,
        write_jsonl_copy=args.write_jsonl_copy,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
