import json

from eval.kg_eval import evaluate_entity_linking, evaluate_relation_extraction


def _write_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def test_evaluate_relation_extraction_computes_precision(tmp_path):
    gold_path = tmp_path / "relations_gold.json"
    _write_json(
        gold_path,
        [
            {"predicate": "located_in", "is_correct": True},
            {"predicate": "located_in", "is_correct": True},
            {"predicate": "part_of", "is_correct": False},
            {"predicate": "known_for", "is_correct": True},
        ],
    )

    report = evaluate_relation_extraction(gold_path)

    assert report.eval_name == "kg_relations_v1"
    assert report.dataset_size == 4
    assert report.metrics["precision"] == 0.75
    assert report.metrics["correct"] == 3
    assert report.metrics["total"] == 4


def test_evaluate_relation_extraction_raises_on_empty_gold(tmp_path):
    gold_path = tmp_path / "empty.json"
    _write_json(gold_path, [])

    try:
        evaluate_relation_extraction(gold_path)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_evaluate_entity_linking_separates_accuracy_and_abstention(tmp_path):
    gold_path = tmp_path / "linking_gold.json"
    _write_json(
        gold_path,
        [
            # linkable, correctly linked
            {"canonical_name": "يافا", "linked_qid": "Q1234", "expected_qid": "Q1234"},
            # linkable, but linked to the wrong QID
            {"canonical_name": "حيفا", "linked_qid": "Q9999", "expected_qid": "Q5678"},
            # not linkable (no correct QID exists) — correctly abstained
            {"canonical_name": "كيان محلي غير موثق", "linked_qid": None, "expected_qid": None},
            # not linkable, but the linker guessed wrong (should have abstained)
            {"canonical_name": "شيء آخر غير موجود", "linked_qid": "Q4321", "expected_qid": None},
        ],
    )

    report = evaluate_entity_linking(gold_path)

    assert report.eval_name == "kg_entity_linking_v1"
    assert report.dataset_size == 4
    assert report.metrics["linkable_sample_size"] == 2
    assert report.metrics["accuracy_on_linkable_entities"] == 0.5
    assert report.metrics["no_qid_expected_sample_size"] == 2
    assert report.metrics["correct_abstention_rate"] == 0.5


def test_evaluate_entity_linking_raises_on_empty_gold(tmp_path):
    gold_path = tmp_path / "empty.json"
    _write_json(gold_path, [])

    try:
        evaluate_entity_linking(gold_path)
        assert False, "expected ValueError"
    except ValueError:
        pass
