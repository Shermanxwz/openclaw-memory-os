from __future__ import annotations

import os
from pathlib import Path

from openclaw_memory_os.evaluation_reports import (
    list_evaluation_reports,
    load_latest_evaluation_report,
    save_evaluation_report,
    unavailable_envelope,
)


def test_no_report_is_explicitly_unavailable(tmp_path):
    assert load_latest_evaluation_report(report_dir=tmp_path) is None
    assert unavailable_envelope()["status"] == "unavailable"


def test_report_round_trip_history_and_permissions(tmp_path):
    report = {
        "report_id": "report-1",
        "status": "ok",
        "corpus_snapshot_id": "snapshot-1",
        "metrics": {"useful_at_1": 0.75},
        "active_metrics": {"useful_at_1": 0.5},
        "candidate_metrics": {"useful_at_1": 0.75},
        "policy": {"active_version": 1, "candidate_version": 2},
        "decision": {"status": "shadow"},
    }
    path = save_evaluation_report(report, report_dir=tmp_path)
    loaded = load_latest_evaluation_report(report_dir=tmp_path)
    assert loaded["report_id"] == "report-1"
    assert loaded["metrics"]["useful_at_1"] == 0.75
    assert list_evaluation_reports(report_dir=tmp_path)[0]["report_id"] == "report-1"
    if os.name != "nt":
        assert (tmp_path.stat().st_mode & 0o777) == 0o700
        assert (path.stat().st_mode & 0o777) == 0o600
        assert ((tmp_path / "latest.json").stat().st_mode & 0o777) == 0o600


def test_dashboard_and_cli_have_no_noop_ranker():
    app_source = Path("openclaw_memory_os/app.py").read_text(encoding="utf-8")
    cli_source = Path("scripts/evaluate_retrieval.py").read_text(encoding="utf-8")
    assert "_no_op_rank_fn" not in app_source
    assert "_no_op_rank_fn" not in cli_source
    assert "load_latest_evaluation_report" in app_source
    assert "load_latest_evaluation_report" in cli_source


def test_runner_does_not_invent_fixed_candidate():
    source = Path("scripts/run_evolution_cycle.py").read_text(encoding="utf-8")
    assert "candidate_clamped" not in source
    assert 'candidate_clamped["importance_weight"]' not in source
