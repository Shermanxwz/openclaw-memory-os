"""Tests for the compact autonomous Memory governance dashboard card.

The weekly ``weekly-memory-autonomous-content-governance`` job runs Tue 04:01
Asia/Shanghai with ``FORCE_CONTENT_SUPERSEDE=1``. The dashboard should show
operational status (last run, next run, result), not verbose policy prose.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from openclaw_memory_os.app import create_app
from openclaw_memory_os.config import reset_settings_cache
from openclaw_memory_os.models import AutonomousGovernanceJob


def _client():
    app = create_app()
    return TestClient(app)


def test_autonomous_governance_defaults_match_contract():
    job = AutonomousGovernanceJob()
    assert job.name == "weekly-memory-autonomous-content-governance"
    assert job.schedule == "Tue 04:01 Asia/Shanghai"
    assert job.mode == "FORCE_CONTENT_SUPERSEDE=1"
    assert job.scope == "memory-content"
    assert job.last_run is None
    assert job.last_result is None
    assert job.last_summary is None
    assert set(job.allowed_actions) == {"supersede", "expire", "archive", "dedupe", "promote"}
    boundary = job.safety_boundary.lower()
    for forbidden in ("never physically delete", "repo", "config", "secrets"):
        assert forbidden in boundary


def test_autonomous_governance_for_dashboard_computes_next_run():
    job = AutonomousGovernanceJob.for_dashboard()
    payload = job.model_dump(mode="json")
    assert payload["name"] == "weekly-memory-autonomous-content-governance"
    assert payload["mode"] == "FORCE_CONTENT_SUPERSEDE=1"
    assert payload["next_run"]
    assert "+08:00" in payload["next_run"]


def test_api_health_includes_governance_status_from_status_file(monkeypatch, tmp_path):
    status_file = tmp_path / "autonomous-governance.json"
    status_file.write_text(
        json.dumps(
            {
                "last_run": "2026-07-14 04:01",
                "last_result": "ok",
                "last_summary": "2 supersede links; 1 promotion; no deletions.",
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    reset_settings_cache()
    try:
        with _client() as c:
            r = c.get("/api/health")
        assert r.status_code == 200
        gov = r.json()["autonomous_governance"]
        assert gov["last_run"] == "2026-07-14 04:01"
        assert gov["last_result"] == "ok"
        assert gov["last_summary"] == "2 supersede links; 1 promotion; no deletions."
        assert gov["next_run"]
    finally:
        reset_settings_cache()


def test_dashboard_overview_renders_single_compact_status_card():
    with _client() as c:
        r = c.get("/dashboard/overview")
    assert r.status_code == 200
    html = r.text

    assert 'id="autonomous-governance-cards"' in html
    assert html.count('data-governance-card="status"') == 1
    assert 'data-governance-card="job"' not in html
    assert 'data-governance-card="safety"' not in html

    for needle in (
        "Memory 自主治理",
        "上次启动",
        "下次计划",
        "运行结果",
        "Tue 04:01 Asia/Shanghai",
        "FORCE_CONTENT_SUPERSEDE=1",
        "memory-content",
    ):
        assert needle in html

    # No verbose policy wall on the dashboard.
    assert "Allowed actions" not in html
    assert "Never physically delete" not in html
    assert "Job name · weekly cron" not in html


def test_dashboard_overview_uses_data_attributes_for_status_values():
    with _client() as c:
        r = c.get("/dashboard/overview")
    html = r.text
    for attr in (
        'data-governance="last-run"',
        'data-governance="next-run"',
        'data-governance="result"',
        'data-governance="summary"',
        'data-governance="meta"',
    ):
        assert attr in html


def test_dashboard_overview_renders_governance_block_with_auth_enabled(monkeypatch):
    monkeypatch.setenv("MEMORY_OS_TOKEN", "test-token")
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    reset_settings_cache()
    try:
        with _client() as c:
            r = c.get(
                "/dashboard/overview",
                headers={"Authorization": "Bearer test-token"},
            )
        assert r.status_code == 200
        assert "Memory 自主治理" in r.text
        assert "Internal Server Error" not in r.text
        assert "autonomous_governance is undefined" not in r.text
    finally:
        reset_settings_cache()


def test_dashboard_governance_card_does_not_leak_paths_or_secrets():
    with _client() as c:
        r = c.get("/dashboard/overview")
    html = r.text
    start = html.find("autonomous-governance-cards")
    assert start != -1
    end = html.find("</div>", html.find("Memory 自主治理", start))
    region = html[start : end + 6] if end != -1 else html[start : start + 2500]

    for needle in (
        "/var/log/",
        "/tmp/",
        "/root/",
        "api_key",
        "apikey",
        "token=",
        "0.0.0.0",
        "127.0.0.1",
        "localhost",
    ):
        assert needle not in region


def test_dashboard_other_sections_do_not_render_governance_card():
    with _client() as c:
        for section in ("tiers", "duplicates", "recall", "governance", "strategy", "health", "security"):
            r = c.get(f"/dashboard/{section}")
            assert r.status_code == 200
            assert "autonomous-governance-cards" not in r.text
