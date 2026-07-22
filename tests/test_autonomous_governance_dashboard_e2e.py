"""End-to-end dashboard ingestion test for the autonomous governance status.

Walks the full chain that operators will exercise every Tuesday at 04:01:

    1. The weekly runner writes the redacted status JSON via the
       ``write_autonomous_governance_status`` helper.
    2. The dashboard reader picks it up through the ``MEMORY_OS_GOVERNANCE_STATUS``
       env override.
    3. ``/api/health`` propagates ``last_run`` / ``last_result`` /
       ``last_summary`` so the operator UI can render the compact
       ``Memory 自主治理`` card.

This test deliberately exercises the *real* writer (not a mock), so a
regression in either direction surfaces here.
"""

from __future__ import annotations

import json

import pytest
from fastapi.testclient import TestClient

from openclaw_memory_os.analytics import write_autonomous_governance_status
from openclaw_memory_os.app import create_app
from openclaw_memory_os.config import reset_settings_cache


@pytest.fixture
def client():
    return TestClient(create_app())


def _iso_now(offset_seconds: int = 0) -> str:
    from datetime import datetime, timedelta

    from openclaw_memory_os.models import _OPERATOR_TZ

    dt = datetime.now(_OPERATOR_TZ) + timedelta(seconds=offset_seconds)
    return dt.strftime("%Y-%m-%d %H:%M")


def test_writer_to_dashboard_e2e(monkeypatch, tmp_path):
    """Full chain: writer → status file → /api/health → dashboard fields."""
    status_file = tmp_path / "governance-e2e.json"

    fixed_run = _iso_now()
    fixed_summary = "2 supersede links; 1 promotion; no deletions."
    write_autonomous_governance_status(
        status_file_path=status_file,
        result_token="ok",
        summary=fixed_summary,
        finished_at=fixed_run,
    )

    assert status_file.exists()
    payload = json.loads(status_file.read_text(encoding="utf-8"))
    assert payload["last_run"] == fixed_run
    assert payload["last_result"] == "ok"
    assert payload["last_summary"] == fixed_summary

    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    reset_settings_cache()
    try:
        with TestClient(create_app()) as c:
            r = c.get("/api/health")
        assert r.status_code == 200
        body = r.json()
        gov = body["autonomous_governance"]
        assert gov["last_run"] == fixed_run
        assert gov["last_result"] == "ok"
        assert gov["last_summary"] == fixed_summary
        # next_run is computed from the fixed schedule; the contract is Tue 04:01 Asia/Shanghai.
        assert gov["next_run"]
        assert "+08:00" in gov["next_run"]
    finally:
        reset_settings_cache()


def test_writer_to_dashboard_e2e_failed_state(monkeypatch, tmp_path):
    status_file = tmp_path / "governance-failed.json"
    write_autonomous_governance_status(
        status_file_path=status_file,
        result_token="failed",
        summary="deep audit failed; exit=17",
        finished_at="2026-07-14 04:01",
    )
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    reset_settings_cache()
    try:
        with TestClient(create_app()) as c:
            r = c.get("/api/health")
        assert r.status_code == 200
        gov = r.json()["autonomous_governance"]
        assert gov["last_result"] == "failed"
        assert "17" in gov["last_summary"]
    finally:
        reset_settings_cache()


def test_dashboard_overview_renders_status_after_e2e_write(monkeypatch, tmp_path):
    """The overview page must surface the writer's output verbatim."""
    status_file = tmp_path / "gov.json"
    write_autonomous_governance_status(
        status_file_path=status_file,
        result_token="ok",
        summary="e2e dashboard render",
        finished_at="2026-07-14 04:01",
    )
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    reset_settings_cache()
    try:
        with TestClient(create_app()) as c:
            r = c.get("/dashboard/overview")
        assert r.status_code == 200
        html = r.text
        assert "Memory 自主治理" in html
        # The rendered summary text appears in the data-governance attribute.
        assert "e2e dashboard render" in html
        # Result token reaches the UI.
        assert 'data-governance="result"' in html
    finally:
        reset_settings_cache()


def test_dashboard_handles_unknown_when_status_missing(monkeypatch, tmp_path):
    """No status file → dashboard must report unknown, not crash."""
    monkeypatch.delenv("MEMORY_OS_GOVERNANCE_STATUS", raising=False)
    # Make the XDG fallback directory unreadable to avoid picking up a real file.
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    reset_settings_cache()
    try:
        with TestClient(create_app()) as c:
            r = c.get("/api/health")
        assert r.status_code == 200
        gov = r.json()["autonomous_governance"]
        # last_run / last_result / last_summary all None → unknown in the UI.
        assert gov["last_run"] is None
        assert gov["last_result"] is None
        assert gov["last_summary"] is None
    finally:
        reset_settings_cache()


def test_status_file_round_trip_preserves_three_keys(monkeypatch, tmp_path):
    """Write → read → /api/health must keep the schema discipline end-to-end."""
    status_file = tmp_path / "round-trip.json"
    write_autonomous_governance_status(
        status_file_path=status_file,
        result_token="running",
        summary="active deep audit",
        finished_at="2026-07-14 04:01",
    )
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    reset_settings_cache()
    try:
        with TestClient(create_app()) as c:
            r = c.get("/api/health")
        gov = r.json()["autonomous_governance"]
        assert set(gov.keys()) >= {"last_run", "last_result", "last_summary"}
        assert gov["last_result"] == "running"
    finally:
        reset_settings_cache()


def test_writer_persists_across_process_restarts(monkeypatch, tmp_path):
    """The on-disk status file is the source of truth across restarts."""
    status_file = tmp_path / "persistent.json"
    write_autonomous_governance_status(
        status_file_path=status_file,
        result_token="ok",
        summary="persisted",
        finished_at="2026-07-14 04:01",
    )

    # New client, new app instance — proves the file, not the process,
    # is what the dashboard reads.
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    reset_settings_cache()
    try:
        with TestClient(create_app()) as c:
            r = c.get("/api/health")
        gov = r.json()["autonomous_governance"]
        assert gov["last_summary"] == "persisted"
        assert gov["last_result"] == "ok"
    finally:
        reset_settings_cache()


def test_dashboard_does_not_leak_writer_artifacts(monkeypatch, tmp_path):
    """Dashboard JSON must not echo tmp paths, env vars, or writer internals."""
    status_file = tmp_path / "leak-check.json"
    write_autonomous_governance_status(
        status_file_path=status_file,
        result_token="ok",
        summary="leak check ok",
        finished_at="2026-07-14 04:01",
    )
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    reset_settings_cache()
    try:
        with TestClient(create_app()) as c:
            api = c.get("/api/health").json()
            html = c.get("/dashboard/overview").text
    finally:
        reset_settings_cache()

    # API JSON: only the contract keys are exposed.
    gov = api["autonomous_governance"]
    forbidden_api = {
        "collections",
        "collection",
        "file",
        "paths",
        "path",
        "tokens",
        "ip",
        "host",
        "log",
    }
    for key in forbidden_api:
        assert key not in gov, f"leaked key {key!r} in /api/health"

    # HTML must not contain the tmp path or the writer's tempfile prefix.
    for needle in (
        str(status_file),
        ".autonomous-governance-",
        "MEMORY_OS_GOVERNANCE_STATUS",
        "tmp_path",
    ):
        assert needle not in html, f"leaked {needle!r} in dashboard HTML"