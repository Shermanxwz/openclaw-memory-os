from __future__ import annotations

import sqlite3

import pytest
from fastapi import HTTPException, Response


def test_session_store_failure_does_not_issue_cookie(monkeypatch, tmp_path):
    import openclaw_memory_os.auth as auth

    class BrokenStore:
        def create(self, *args, **kwargs):
            raise sqlite3.OperationalError("disk full")

    monkeypatch.setattr(auth, "get_session_store", lambda: BrokenStore())
    response = Response()
    with pytest.raises(HTTPException) as exc:
        auth.set_session_cookie(response, "secret-session")
    assert exc.value.status_code == 503
    assert "set-cookie" not in response.headers


def test_recall_db_permissions_and_retention_preserves_feedback(monkeypatch, tmp_path):
    import openclaw_memory_os.recall_feedback as rf

    state = tmp_path / "state"
    monkeypatch.setattr(rf, "_RECALL_DB_DIR", state)
    monkeypatch.setattr(rf, "_RECALL_DB", state / "recall_feedback.db")
    conn = rf._get_db(run_legacy_migration=False)
    try:
        old = "2000-01-01 00:00:00"
        conn.execute("INSERT INTO recall_runs(query_id, query_text, created_at) VALUES ('q-old','x',?)", (old,))
        conn.execute("INSERT INTO recall_results(query_id,candidate_key) VALUES ('q-old','sample:1')")
        conn.execute("INSERT INTO feedback_events(query_id,candidate_key,useful) VALUES ('q-old','sample:1',1)")
        conn.commit()
    finally:
        conn.close()
    assert oct((state / "recall_feedback.db").stat().st_mode & 0o777) == "0o600"
    assert rf._retention_cleanup() == 1
    conn = rf._get_db(run_legacy_migration=False)
    try:
        assert conn.execute("SELECT COUNT(*) FROM recall_runs").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM recall_results").fetchone()[0] == 0
        assert conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0] == 1
    finally:
        conn.close()


def test_policy_store_defaults_to_persistent_active_and_restores_slots(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("MEMORY_OS_POLICY_DIR", raising=False)
    monkeypatch.delenv("MEMORY_OS_POLICY_PATH", raising=False)
    from openclaw_memory_os.policy_store import Policy, PolicyStatus, PolicyStore, baseline_policy

    store = PolicyStore()
    assert store.path == tmp_path / "openclaw-memory-os" / "policies" / "active.json"
    active = Policy(**baseline_policy, version=10, status=PolicyStatus.ACTIVE)
    candidate_data = dict(baseline_policy)
    candidate_data.update(version=11, parent_version=10, status=PolicyStatus.SHADOW)
    candidate = Policy(**candidate_data)
    store.set(active)
    store.save(active)
    store.set_shadow(candidate)
    reloaded = PolicyStore()
    assert reloaded.get().version == 10
    assert reloaded.get_shadow() is not None
    assert reloaded.get_shadow().version == 11
