import json

from scripts.run_scheduled_collection import run_cycle


def test_run_cycle_logs_success_for_every_source(tmp_path, monkeypatch):
    import scripts.run_scheduled_collection as module

    monkeypatch.setitem(module._RUNNERS, "wafa", lambda max_docs: {"accepted_documents": 5})
    monkeypatch.setitem(module._RUNNERS, "gdelt", lambda max_docs: {"accepted_documents": 2})
    log_path = tmp_path / "log.jsonl"

    summary = run_cycle(sources=["wafa", "gdelt"], max_docs=10, log_path=log_path)

    assert summary["sources_succeeded"] == ["wafa", "gdelt"]
    assert summary["sources_failed"] == []
    assert summary["results"]["wafa"]["stats"]["accepted_documents"] == 5

    logged = json.loads(log_path.read_text(encoding="utf-8").strip())
    assert logged["sources_succeeded"] == ["wafa", "gdelt"]


def test_run_cycle_isolates_one_source_failure_from_the_rest(tmp_path, monkeypatch):
    import scripts.run_scheduled_collection as module

    def failing(max_docs):
        raise RuntimeError("rate limited")

    monkeypatch.setitem(module._RUNNERS, "wafa", failing)
    monkeypatch.setitem(module._RUNNERS, "gdelt", lambda max_docs: {"accepted_documents": 3})
    log_path = tmp_path / "log.jsonl"

    summary = run_cycle(sources=["wafa", "gdelt"], max_docs=10, log_path=log_path)

    assert summary["sources_failed"] == ["wafa"]
    assert summary["sources_succeeded"] == ["gdelt"]
    assert "rate limited" in summary["results"]["wafa"]["error"]


def test_run_cycle_appends_to_existing_log(tmp_path, monkeypatch):
    import scripts.run_scheduled_collection as module

    monkeypatch.setitem(module._RUNNERS, "wafa", lambda max_docs: {"accepted_documents": 1})
    log_path = tmp_path / "log.jsonl"

    run_cycle(sources=["wafa"], max_docs=10, log_path=log_path)
    run_cycle(sources=["wafa"], max_docs=10, log_path=log_path)

    lines = log_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
