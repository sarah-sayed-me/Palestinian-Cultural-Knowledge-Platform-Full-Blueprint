import json

import pytest

from scripts.export_to_hf import _is_clear, export_to_hf_dataset


def _record(**overrides):
    base = {
        "doc_id": "1",
        "title": "فلسطين",
        "text": "فلسطين ثقافة وتاريخ",
        "word_count": 3,
        "language": "ar-MSA",
        "source_id": "wikipedia-ar",
        "license_status": "clear",
    }
    base.update(overrides)
    return base


def test_export_to_hf_dataset_writes_parquet(tmp_path):
    input_path = tmp_path / "docs.jsonl"
    output_dir = tmp_path / "hf"
    input_path.write_text(json.dumps(_record(), ensure_ascii=False) + "\n", encoding="utf-8")

    summary = export_to_hf_dataset(input_path=input_path, output_dir=output_dir)

    assert summary["num_rows"] == 1
    assert summary["excluded_as_not_clear"] == 0
    assert (output_dir / "train.parquet").exists()
    assert (output_dir / "dataset_summary.json").exists()


def test_is_clear_uses_explicit_license_status():
    assert _is_clear(_record(license_status="clear")) is True
    assert _is_clear(_record(license_status="needs_review")) is False
    assert _is_clear(_record(license_status="blocked")) is False


def test_export_includes_everything_by_default_regardless_of_license_status(tmp_path):
    input_path = tmp_path / "docs.jsonl"
    output_dir = tmp_path / "hf"
    records = [
        _record(doc_id="1", license_status="clear"),
        _record(doc_id="2", license_status="needs_review", source_id="wafa-news"),
        _record(doc_id="3", license_status="blocked", source_id="gdelt"),
    ]
    input_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )

    summary = export_to_hf_dataset(input_path=input_path, output_dir=output_dir)

    assert summary["num_rows"] == 3
    assert summary["excluded_as_not_clear"] == 0


def test_export_clear_only_restricts_to_clear_documents(tmp_path):
    input_path = tmp_path / "docs.jsonl"
    output_dir = tmp_path / "hf"
    records = [
        _record(doc_id="1", license_status="clear"),
        _record(doc_id="2", license_status="needs_review", source_id="gdelt"),
    ]
    input_path.write_text(
        "\n".join(json.dumps(r, ensure_ascii=False) for r in records), encoding="utf-8"
    )

    summary = export_to_hf_dataset(input_path=input_path, output_dir=output_dir, clear_only=True)

    assert summary["num_rows"] == 1
    assert summary["excluded_as_not_clear"] == 1


def test_export_combines_multiple_input_files(tmp_path):
    wiki_path = tmp_path / "wikipedia_ar.jsonl"
    wafa_path = tmp_path / "wafa.jsonl"
    output_dir = tmp_path / "hf"
    wiki_path.write_text(json.dumps(_record(doc_id="1"), ensure_ascii=False) + "\n", encoding="utf-8")
    wafa_path.write_text(
        json.dumps(_record(doc_id="2", source_id="wafa-news", license_status="needs_review"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary = export_to_hf_dataset(input_path=[wiki_path, wafa_path], output_dir=output_dir)

    assert summary["num_rows"] == 2
    assert summary["by_source"] == {"wikipedia-ar": 1, "wafa-news": 1}
    assert summary["input"] == [str(wiki_path), str(wafa_path)]


def test_export_clear_only_raises_when_nothing_is_clear(tmp_path):
    input_path = tmp_path / "docs.jsonl"
    output_dir = tmp_path / "hf"
    input_path.write_text(
        json.dumps(_record(license_status="needs_review", source_id="gdelt"), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="No 'clear' records"):
        export_to_hf_dataset(input_path=input_path, output_dir=output_dir, clear_only=True)
