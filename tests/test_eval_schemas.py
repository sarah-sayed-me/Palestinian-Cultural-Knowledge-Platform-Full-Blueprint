from eval.schemas import EvalReport


def test_eval_report_serializes_run_at_to_iso_string():
    report = EvalReport(eval_name="ner_v1", dataset_size=50, metrics={"precision": 0.81, "recall": 0.74})

    payload = report.to_json_dict()

    assert payload["eval_name"] == "ner_v1"
    assert isinstance(payload["run_at"], str)
    assert payload["metrics"]["precision"] == 0.81
