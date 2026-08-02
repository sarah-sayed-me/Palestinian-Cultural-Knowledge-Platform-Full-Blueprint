import json

from src.monitoring.health_report import (
    build_health_report,
    detect_anomalies,
    quality_decision_drift,
    read_latest_source_stats,
    read_run_log,
)


def _write_jsonl(path, entries):
    with path.open("w", encoding="utf-8") as handle:
        for entry in entries:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def test_read_run_log_returns_empty_list_when_file_missing(tmp_path):
    assert read_run_log(tmp_path / "does_not_exist.jsonl") == []


def test_read_run_log_parses_jsonl(tmp_path):
    path = tmp_path / "log.jsonl"
    _write_jsonl(path, [{"run_at": "t1"}, {"run_at": "t2"}])

    entries = read_run_log(path)

    assert len(entries) == 2
    assert entries[0]["run_at"] == "t1"


def test_read_latest_source_stats_skips_missing_files(tmp_path):
    existing = tmp_path / "wafa_stats.json"
    existing.write_text(json.dumps({"accepted_documents": 5}), encoding="utf-8")

    result = read_latest_source_stats({"wafa": existing, "gdelt": tmp_path / "missing.json"})

    assert result == {"wafa": {"accepted_documents": 5}}


def test_quality_decision_drift_tracks_accept_rate_over_runs():
    run_log = [
        {
            "run_at": "t1",
            "results": {
                "wafa": {
                    "status": "ok",
                    "stats": {
                        "attempted_documents": 100,
                        "accepted_documents": 90,
                        "quality_decision_distribution": {"accept": 90, "reject": 10},
                    },
                }
            },
        },
        {
            "run_at": "t2",
            "results": {
                "wafa": {
                    "status": "ok",
                    "stats": {
                        "attempted_documents": 100,
                        "accepted_documents": 40,
                        "quality_decision_distribution": {"accept": 40, "reject": 60},
                    },
                }
            },
        },
    ]

    drift = quality_decision_drift(run_log, "wafa")

    assert len(drift) == 2
    assert drift[0]["accept_rate"] == 0.9
    assert drift[1]["accept_rate"] == 0.4


def test_quality_decision_drift_skips_failed_runs():
    run_log = [{"run_at": "t1", "results": {"wafa": {"status": "error", "error": "boom"}}}]

    assert quality_decision_drift(run_log, "wafa") == []


def test_detect_anomalies_flags_source_failure():
    run_log = [{"run_at": "t1", "results": {"gdelt": {"status": "error", "error": "rate limited"}}}]

    anomalies = detect_anomalies(run_log)

    assert len(anomalies) == 1
    assert anomalies[0]["kind"] == "source_failed"
    assert anomalies[0]["source"] == "gdelt"


def test_detect_anomalies_flags_sharp_accept_rate_drop():
    run_log = [
        {
            "run_at": "t1",
            "results": {"wafa": {"status": "ok", "stats": {"attempted_documents": 100, "accepted_documents": 90}}},
        },
        {
            "run_at": "t2",
            "results": {"wafa": {"status": "ok", "stats": {"attempted_documents": 100, "accepted_documents": 40}}},
        },
    ]

    anomalies = detect_anomalies(run_log)

    assert len(anomalies) == 1
    assert anomalies[0]["kind"] == "accept_rate_drop"


def test_detect_anomalies_does_not_flag_small_fluctuations():
    run_log = [
        {
            "run_at": "t1",
            "results": {"wafa": {"status": "ok", "stats": {"attempted_documents": 100, "accepted_documents": 90}}},
        },
        {
            "run_at": "t2",
            "results": {"wafa": {"status": "ok", "stats": {"attempted_documents": 100, "accepted_documents": 80}}},
        },
    ]

    assert detect_anomalies(run_log) == []


def test_build_health_report_combines_snapshot_and_drift(tmp_path):
    run_log_path = tmp_path / "log.jsonl"
    _write_jsonl(
        run_log_path,
        [
            {
                "run_at": "t1",
                "results": {
                    "wafa": {
                        "status": "ok",
                        "stats": {
                            "attempted_documents": 10,
                            "accepted_documents": 8,
                            "quality_decision_distribution": {"accept": 8},
                        },
                    }
                },
            }
        ],
    )
    stats_path = tmp_path / "wafa_stats.json"
    stats_path.write_text(
        json.dumps({"attempted_documents": 10, "accepted_documents": 8, "quality_decision_distribution": {"accept": 8}}),
        encoding="utf-8",
    )

    report = build_health_report(run_log_path, {"wafa": stats_path})

    assert report["runs_recorded"] == 1
    assert report["current_snapshot_by_source"]["wafa"]["accept_rate"] == 0.8
    assert len(report["quality_decision_drift_by_source"]["wafa"]) == 1
    assert report["anomalies"] == []
