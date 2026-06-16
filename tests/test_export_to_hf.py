import json

from scripts.export_to_hf import export_to_hf_dataset


def test_export_to_hf_dataset_writes_parquet(tmp_path):
    input_path = tmp_path / "docs.jsonl"
    output_dir = tmp_path / "hf"
    record = {
        "doc_id": "1",
        "title": "فلسطين",
        "text": "فلسطين ثقافة وتاريخ",
        "word_count": 3,
        "language": "ar-MSA",
    }
    input_path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = export_to_hf_dataset(input_path=input_path, output_dir=output_dir)

    assert summary["num_rows"] == 1
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "dataset_summary.json").exists()
