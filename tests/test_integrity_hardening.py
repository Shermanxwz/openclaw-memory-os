from __future__ import annotations

import importlib
import os
import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openclaw_memory_os import auth, sessions
from openclaw_memory_os.policy_store import Policy, PolicyStatus, PolicyStore, baseline_policy
from openclaw_memory_os.sessions import SessionStore


def _empty_env(monkeypatch, tmp_path: Path, *, token: str = "") -> None:
    env_file = tmp_path / "empty.env"
    env_file.write_text("", encoding="utf-8")
    monkeypatch.setenv("MEMORY_OS_ENV_FILE", str(env_file))
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("MEMORY_OS_POLICY_DIR", str(tmp_path / "policies"))
    monkeypatch.setenv("MEMORY_OS_TOKEN", token)
    monkeypatch.setenv("MEMORY_OS_PASSWORD", "")
    monkeypatch.setenv("MEMORY_OS_TOTP_SECRET", "")
    monkeypatch.setenv("PASSWORD_TOTP_AUTH", "off")
    monkeypatch.setenv("RETRIEVAL_ENGINE_V2", "on")
    from openclaw_memory_os.config import reset_settings_cache
    reset_settings_cache()


def _v1_db(path: Path, token: str = "legacy-cookie") -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE sessions (token TEXT PRIMARY KEY, issued_at TEXT NOT NULL, "
            "max_age INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, last_seen_at TEXT)"
        )
        conn.execute(
            "INSERT INTO sessions VALUES (?, ?, ?, ?, ?)",
            (token, "2026-07-16T00:00:00+00:00", 999999, 0, None),
        )
        conn.commit()
    finally:
        conn.close()


def test_v1_session_migration_preserves_row_and_removes_raw_token(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"
    _v1_db(db)
    store = SessionStore(db)
    try:
        assert store.contains("legacy-cookie")
        columns = {row[1] for row in store._conn.execute("PRAGMA table_info(sessions)")}
        assert "token_hash" in columns and "user_id" in columns
        assert "token" not in columns
        dump = "\n".join(store._conn.iterdump())
        assert "legacy-cookie" not in dump
    finally:
        store.close()


def test_v1_session_migration_rolls_back_before_destructive_ddl(
    tmp_path: Path, monkeypatch
) -> None:
    db = tmp_path / "sessions.db"
    _v1_db(db)
    monkeypatch.setattr(sessions, "_hash_token", lambda token: (_ for _ in ()).throw(RuntimeError("boom")))
    with pytest.raises(RuntimeError, match="boom"):
        SessionStore(db)
    conn = sqlite3.connect(db)
    try:
        columns = {row[1] for row in conn.execute("PRAGMA table_info(sessions)")}
        assert "token" in columns and "token_hash" not in columns
        assert conn.execute("SELECT token FROM sessions").fetchone()[0] == "legacy-cookie"
    finally:
        conn.close()


def test_v2_session_migration_adds_user_id_atomically(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute(
            "CREATE TABLE sessions (token_hash TEXT PRIMARY KEY, issued_at TEXT NOT NULL, "
            "max_age INTEGER NOT NULL, revoked INTEGER NOT NULL DEFAULT 0, last_seen_at TEXT)"
        )
        conn.commit()
    finally:
        conn.close()
    store = SessionStore(db)
    try:
        columns = {row[1] for row in store._conn.execute("PRAGMA table_info(sessions)")}
        assert "user_id" in columns
    finally:
        store.close()


def test_unknown_session_schema_is_rejected_without_overwrite(tmp_path: Path) -> None:
    db = tmp_path / "sessions.db"
    conn = sqlite3.connect(db)
    try:
        conn.execute("CREATE TABLE sessions (unexpected TEXT)")
        conn.execute("INSERT INTO sessions VALUES ('keep-me')")
        conn.commit()
    finally:
        conn.close()
    with pytest.raises(sqlite3.DatabaseError, match="unsupported sessions schema"):
        SessionStore(db)
    conn = sqlite3.connect(db)
    try:
        assert conn.execute("SELECT unexpected FROM sessions").fetchone()[0] == "keep-me"
    finally:
        conn.close()


def test_test_store_swap_returns_usable_previous_store(tmp_path: Path) -> None:
    first = SessionStore(tmp_path / "first.db")
    second = SessionStore(tmp_path / "second.db")
    original = auth._set_session_store_for_tests(first)
    try:
        previous = auth._set_session_store_for_tests(second)
        assert previous is first
        auth._set_session_store_for_tests(previous)
        first.create("still-open", 3600)
        assert first.is_valid("still-open")
    finally:
        auth._set_session_store_for_tests(original)
        first.close()
        second.close()


class _FailingRevokeStore:
    def revoke(self, token: str) -> bool:
        raise sqlite3.OperationalError("disk unavailable")

    def revoke_all(self) -> int:
        raise sqlite3.OperationalError("disk unavailable")


def test_logout_revocation_failure_returns_503_without_cookie_clear(
    monkeypatch, tmp_path: Path
) -> None:
    _empty_env(monkeypatch, tmp_path, token="static-secret")
    app_module = importlib.import_module("openclaw_memory_os.app")
    app = app_module.create_app()
    original = auth._set_session_store_for_tests(_FailingRevokeStore())  # type: ignore[arg-type]
    try:
        with TestClient(app) as client:
            response = client.post(
                "/logout",
                data={"csrf_token": "csrf"},
                cookies={"csrf_token": "csrf", "memory_os_session": "stolen-cookie"},
                follow_redirects=False,
            )
        assert response.status_code == 503
        assert "memory_os_session=""" not in response.headers.get("set-cookie", "")
    finally:
        auth._set_session_store_for_tests(original)


def test_static_bearer_logout_does_not_revoke_static_credential(
    monkeypatch, tmp_path: Path
) -> None:
    _empty_env(monkeypatch, tmp_path, token="static-secret")
    store = SessionStore(tmp_path / "auth-sessions.db")
    original = auth._set_session_store_for_tests(store)
    auth._revoked_sessions.discard("static-secret")
    try:
        app_module = importlib.import_module("openclaw_memory_os.app")
        app = app_module.create_app()
        with TestClient(app) as client:
            logout = client.post(
                "/logout",
                data={"csrf_token": "csrf"},
                cookies={"csrf_token": "csrf"},
                headers={"Authorization": "Bearer static-secret"},
                follow_redirects=False,
            )
            assert logout.status_code == 303
            health = client.get(
                "/api/health", headers={"Authorization": "Bearer static-secret"}
            )
            assert health.status_code == 200
        assert "static-secret" not in auth._revoked_sessions
    finally:
        auth._set_session_store_for_tests(original)
        store.close()


def test_revoke_all_failure_is_not_reported_as_success(monkeypatch, tmp_path: Path) -> None:
    _empty_env(monkeypatch, tmp_path)
    original = auth._set_session_store_for_tests(_FailingRevokeStore())  # type: ignore[arg-type]
    try:
        app_module = importlib.import_module("openclaw_memory_os.app")
        with TestClient(app_module.create_app()) as client:
            response = client.post("/api/security/sessions/revoke-all")
        assert response.status_code == 503
    finally:
        auth._set_session_store_for_tests(original)


def test_manual_policy_rollback_restores_previous_not_baseline(
    monkeypatch, tmp_path: Path
) -> None:
    _empty_env(monkeypatch, tmp_path)
    app_module = importlib.import_module("openclaw_memory_os.app")
    app = app_module.create_app()
    with TestClient(app) as client:
        store: PolicyStore = app.state.policy_store
        p5 = Policy(**baseline_policy, version=5, status=PolicyStatus.ACTIVE)
        p6_kwargs = dict(baseline_policy)
        p6_kwargs.update(version=6, parent_version=5, status=PolicyStatus.ACTIVE)
        p6 = Policy(**p6_kwargs)
        store.set(p5)
        store.save()
        store.set(p6)
        store.save()
        response = client.post("/api/evolution/rollback")
        assert response.status_code == 200, response.text
        assert response.json()["rollback_target"] == "previous"
        assert response.json()["policy_version"] == "v5"
        assert store.get().version == 5


def test_candidate_reject_removes_persisted_shadow(monkeypatch, tmp_path: Path) -> None:
    _empty_env(monkeypatch, tmp_path)
    app_module = importlib.import_module("openclaw_memory_os.app")
    app = app_module.create_app()
    with TestClient(app) as client:
        store: PolicyStore = app.state.policy_store
        candidate_kwargs = dict(baseline_policy)
        candidate_kwargs.update(
            version=2,
            parent_version=store.get().version,
            status=PolicyStatus.SHADOW,
        )
        candidate = Policy(**candidate_kwargs)
        store.set_shadow(candidate)
        candidate_path = Path(os.environ["MEMORY_OS_POLICY_DIR"]) / "candidate.json"
        assert candidate_path.exists()
        response = client.post("/api/evolution/candidate/reject")
        assert response.status_code == 200, response.text
        assert response.json()["rejected_version"] == 2
        assert not candidate_path.exists()
        assert store.get_shadow() is None
    restarted = PolicyStore(policy_dir=Path(os.environ["MEMORY_OS_POLICY_DIR"]))
    assert restarted.get_shadow() is None


def test_failed_candidate_cleanup_cannot_resurrect_promoted_shadow(
    tmp_path: Path, monkeypatch
) -> None:
    policy_dir = tmp_path / "policies"
    store = PolicyStore(policy_dir=policy_dir)
    store.save()
    candidate_kwargs = dict(baseline_policy)
    candidate_kwargs.update(
        version=2,
        parent_version=store.get().version,
        status=PolicyStatus.SHADOW,
    )
    candidate = Policy(**candidate_kwargs)
    store.set_shadow(candidate)
    candidate_path = policy_dir / "candidate.json"
    real_unlink = Path.unlink

    def deny_candidate_unlink(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path == candidate_path:
            raise OSError("simulated unlink failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_candidate_unlink)
    store.promote()
    assert candidate_path.exists()
    restarted = PolicyStore(policy_dir=policy_dir)
    assert restarted.get().version == 2
    assert restarted.get_shadow() is None


def test_canonical_retrieval_failure_never_uses_legacy_fallback(
    monkeypatch, tmp_path: Path
) -> None:
    _empty_env(monkeypatch, tmp_path)
    app_module = importlib.import_module("openclaw_memory_os.app")
    legacy_called = False

    def fail_retrieve(self, *args, **kwargs):  # type: ignore[no-untyped-def]
        raise RuntimeError("engine failure")

    def forbidden_legacy(*args, **kwargs):  # type: ignore[no-untyped-def]
        nonlocal legacy_called
        legacy_called = True
        raise AssertionError("legacy fallback must not run")

    monkeypatch.setattr(app_module.RetrievalEngine, "retrieve", fail_retrieve)
    monkeypatch.setattr(app_module, "build_recall_response", forbidden_legacy)
    with TestClient(app_module.create_app()) as client:
        response = client.post(
            "/api/recall-test", json={"query": "contract", "mode": "hybrid", "limit": 5}
        )
    assert response.status_code == 503
    assert response.json()["detail"] == "Retrieval engine unavailable."
    assert legacy_called is False


def test_logout_prefers_session_cookie_over_authorization_header(
    monkeypatch, tmp_path: Path
) -> None:
    _empty_env(monkeypatch, tmp_path, token="static-secret")
    store = SessionStore(tmp_path / "logout-priority.db")
    store.create("browser-cookie", 3600)
    original = auth._set_session_store_for_tests(store)
    try:
        app_module = importlib.import_module("openclaw_memory_os.app")
        with TestClient(app_module.create_app()) as client:
            response = client.post(
                "/logout",
                data={"csrf_token": "csrf"},
                cookies={
                    "csrf_token": "csrf",
                    "memory_os_session": "browser-cookie",
                },
                headers={"Authorization": "Bearer static-secret"},
                follow_redirects=False,
            )
            assert response.status_code == 303
            assert store.is_valid("browser-cookie") is False
            health = client.get(
                "/api/health", headers={"Authorization": "Bearer static-secret"}
            )
            assert health.status_code == 200
    finally:
        auth._set_session_store_for_tests(original)
        store.close()


def test_candidate_delete_failure_emits_no_false_rejected_history(
    tmp_path: Path, monkeypatch
) -> None:
    policy_dir = tmp_path / "policies"
    store = PolicyStore(policy_dir=policy_dir)
    store.save()
    candidate_kwargs = dict(baseline_policy)
    candidate_kwargs.update(
        version=2,
        parent_version=store.get().version,
        status=PolicyStatus.SHADOW,
    )
    store.set_shadow(Policy(**candidate_kwargs))
    candidate_path = policy_dir / "candidate.json"
    real_unlink = Path.unlink

    def deny_candidate_unlink(path: Path, *args, **kwargs):  # type: ignore[no-untyped-def]
        if path == candidate_path:
            raise OSError("simulated candidate delete failure")
        return real_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", deny_candidate_unlink)
    with pytest.raises(OSError, match="simulated candidate delete failure"):
        store.reject_shadow()
    assert store.get_shadow() is not None
    assert candidate_path.exists()
    history = list((policy_dir / "history").glob("*-rejected-*.json"))
    assert history == []

