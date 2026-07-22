"""Tests for the v0.3.0.x feature flags.

Each flag has a safe default that preserves the previous behaviour.
This module verifies both the on-path (default) and the off-path
(operator-overridden) for:

* ``RETRIEVAL_ENGINE_V2``  — engine vs legacy ``build_recall_response``
* ``STRUCTURED_FEEDBACK``  — recall_runs persistence + /api/feedback path
* ``EVOLUTION_ENABLED``    — evolution endpoints are a safe no-op
* ``PASSWORD_TOTP_AUTH``   — legacy bearer-token path is preserved
* ``SHADOW_ENABLED``       — surfaced in /api/strategy state

The tests use ``monkeypatch.setenv`` + ``reset_settings_cache`` so the
new flags are picked up before ``create_app()`` is called. The shared
``_clean_env`` autouse fixture (``tests/conftest.py``) clears the
flag-related vars between tests so the on-path tests see the
production default.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openclaw_memory_os.app import create_app
from openclaw_memory_os.auth import attempt_login
from openclaw_memory_os.config import get_settings, reset_settings_cache


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _client_for_env(monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Build a TestClient after the test has set any feature-flag env vars.

    The shared ``_clean_env`` autouse fixture clears the flag vars first,
    so this helper just needs to make sure the cached Settings are
    rebuilt against the current process environment.
    """
    reset_settings_cache()
    app = create_app()
    return TestClient(app)


def _count_recall_runs(db_path: Path) -> int:
    """Best-effort rowcount helper that does not require the helper module."""
    if not db_path.exists():
        return 0
    with sqlite3.connect(str(db_path)) as conn:
        cur = conn.execute("SELECT COUNT(*) FROM recall_runs")
        row = cur.fetchone()
        return int(row[0]) if row else 0


# ---------------------------------------------------------------------------
# RETRIEVAL_ENGINE_V2
# ---------------------------------------------------------------------------


class TestRetrievalEngineV2:
    def test_default_on_uses_v030_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (flag unset / on) hits the v0.3.0 retrieval engine.

        The v0.3.0 engine stamps ``diagnostics`` with at least one of
        ``dense_available`` / ``lexical_available`` (the engine
        envelope), and we also confirm ``policy_version`` is set.
        """
        monkeypatch.delenv("RETRIEVAL_ENGINE_V2", raising=False)
        with _client_for_env(monkeypatch) as c:
            r = c.post(
                "/api/recall-test",
                json={"query": "recall", "mode": "hybrid", "limit": 3},
            )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["policy_version"]
        # v0.3.0 engine envelope: at least one channel availability field
        # should be populated. The legacy path leaves diagnostics empty.
        diag = data.get("diagnostics") or {}
        assert "dense_available" in diag or "lexical_available" in diag

    def test_off_uses_legacy_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``RETRIEVAL_ENGINE_V2=off`` uses ``ranking.build_recall_response``."""
        monkeypatch.setenv("RETRIEVAL_ENGINE_V2", "off")
        with _client_for_env(monkeypatch) as c:
            r = c.post(
                "/api/recall-test",
                json={"query": "recall", "mode": "hybrid", "limit": 3},
            )
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["query"] == "recall"
        assert data["policy_version"]
        # Legacy path: the v0.3.0 engine envelope is NOT populated.
        diag = data.get("diagnostics") or {}
        assert "dense_available" not in diag
        assert "lexical_available" not in diag
        assert "candidate_count" not in diag


# ---------------------------------------------------------------------------
# STRUCTURED_FEEDBACK
# ---------------------------------------------------------------------------


class TestStructuredFeedback:
    @pytest.fixture()
    def _isolated_recall_db(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
        """Redirect the recall_feedback module at a per-test DB path.

        ``_RECALL_DB_DIR`` is computed once at module import time so
        changing ``MEMORY_OS_RECALL_STATE_DIR`` later doesn't affect
        the on-disk location. We patch the module-level constants
        directly so each test gets a fresh, isolated DB.
        """
        from openclaw_memory_os import recall_feedback

        db_dir = tmp_path / "openclaw-memory-os"
        db_path = db_dir / "recall_feedback.db"
        monkeypatch.setattr(recall_feedback, "_RECALL_DB_DIR", db_dir)
        monkeypatch.setattr(recall_feedback, "_RECALL_DB", db_path)
        # Belt-and-braces: also set the env var so subsequent reads
        # via getenv stay in sync if any code path consults it again.
        monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path))
        return db_path

    def test_default_on_persists_recall_runs(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _isolated_recall_db: Path,
    ) -> None:
        """Default (flag unset / on) writes the recall_runs row."""
        monkeypatch.delenv("STRUCTURED_FEEDBACK", raising=False)
        with _client_for_env(monkeypatch) as c:
            r = c.post(
                "/api/recall-test",
                json={"query": "audit-yes", "mode": "hybrid", "limit": 3},
            )
        assert r.status_code == 200, r.text
        assert _count_recall_runs(_isolated_recall_db) >= 1

    def test_off_skips_recall_runs_persistence(
        self,
        monkeypatch: pytest.MonkeyPatch,
        _isolated_recall_db: Path,
    ) -> None:
        """``STRUCTURED_FEEDBACK=off`` skips recall_runs persistence on the API."""
        monkeypatch.setenv("STRUCTURED_FEEDBACK", "off")
        with _client_for_env(monkeypatch) as c:
            r = c.post(
                "/api/recall-test",
                json={"query": "audit-no", "mode": "hybrid", "limit": 3},
            )
        assert r.status_code == 200, r.text
        assert _count_recall_runs(_isolated_recall_db) == 0

    def test_off_feedback_routes_to_legacy(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """``STRUCTURED_FEEDBACK=off`` sends /api/feedback to the legacy path
        even when ``query_id`` / ``candidate_key`` are present."""
        monkeypatch.setenv("STRUCTURED_FEEDBACK", "off")
        # Isolate the audit log so we can introspect the legacy write.
        from openclaw_memory_os.audit import get_audit_store

        audit_path = tmp_path / "audit_off.sqlite"
        store = get_audit_store(db_path=audit_path)

        with _client_for_env(monkeypatch) as c:
            r = c.post(
                "/api/feedback",
                json={
                    "memory_id": "mem-legacy",
                    "query": "legacy-route",
                    "useful": True,
                    "query_id": "fake-qid-should-be-ignored",
                    "candidate_key": "fake:key",
                },
            )
        assert r.status_code == 200, r.text
        entries = store.list_recent(limit=5, action="feedback")
        assert entries, "legacy feedback path should write to the audit log"
        # Legacy feedback carries memory_id / query; we routed the call
        # with memory_id="mem-legacy" so the row must surface it.
        assert any(e.memory_id == "mem-legacy" for e in entries)


# ---------------------------------------------------------------------------
# EVOLUTION_ENABLED
# ---------------------------------------------------------------------------


class TestEvolutionEnabled:
    def _post(self, monkeypatch: pytest.MonkeyPatch, endpoint: str):
        """Authenticate via ``Authorization: Bearer`` and POST to ``endpoint``.

        The bearer-token path bypasses the CSRF check, which keeps the
        test setup simple. We still need to set ``MEMORY_OS_TOKEN`` so
        the app gate requires auth; the autouse ``_clean_env`` fixture
        clears it between tests.
        """
        monkeypatch.setenv("MEMORY_OS_TOKEN", "secret")
        reset_settings_cache()
        app = create_app()
        with TestClient(app) as c:
            return c.post(endpoint, headers={"Authorization": "Bearer secret"})

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/evolution/pause",
            "/api/evolution/resume",
            "/api/evolution/candidate/reject",
        ],
    )
    def test_default_on_mutates_state(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        """On-path evolution endpoints mutate state and return status=ok."""
        monkeypatch.delenv("EVOLUTION_ENABLED", raising=False)
        r = self._post(monkeypatch, endpoint)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("status") != "disabled"
        assert "state" in data

    @pytest.mark.parametrize(
        "endpoint",
        [
            "/api/evolution/pause",
            "/api/evolution/resume",
            "/api/evolution/candidate/reject",
            "/api/evolution/rollback",
        ],
    )
    def test_off_returns_disabled_noop(
        self, monkeypatch: pytest.MonkeyPatch, endpoint: str
    ) -> None:
        """``EVOLUTION_ENABLED=off`` makes every evolution endpoint a no-op."""
        monkeypatch.setenv("EVOLUTION_ENABLED", "off")
        r = self._post(monkeypatch, endpoint)
        assert r.status_code == 200, r.text
        data = r.json()
        assert data.get("status") == "disabled"
        assert data.get("reason") == "evolution_enabled=off"

    def test_off_rollback_preserves_policy(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``EVOLUTION_ENABLED=off`` leaves the active policy untouched."""
        monkeypatch.setenv("EVOLUTION_ENABLED", "off")
        r = self._post(monkeypatch, "/api/evolution/rollback")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["status"] == "disabled"
        # Disabled rollback never returns a real policy_version.
        assert data.get("policy_version") in (None, "")


# ---------------------------------------------------------------------------
# PASSWORD_TOTP_AUTH
# ---------------------------------------------------------------------------


class TestPasswordTotpAuth:
    def test_default_on_accepts_valid_password(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On-path: password+TOTP path validates the password alone when
        no TOTP secret is configured."""
        monkeypatch.delenv("PASSWORD_TOTP_AUTH", raising=False)
        monkeypatch.delenv("MEMORY_OS_PASSWORD", raising=False)
        monkeypatch.setenv("MEMORY_OS_PASSWORD", "hunter2")
        s = get_settings()
        # Default-on: a valid password unlocks login (no TOTP required).
        assert attempt_login(password="hunter2", totp_code="", token="", settings=s) is True
        # Wrong password is rejected even on the default-on path.
        assert attempt_login(password="wrong", totp_code="", token="", settings=s) is False

    def test_on_with_totp_requires_totp_code(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """On-path with TOTP: a correct password without a TOTP code is rejected."""
        monkeypatch.delenv("PASSWORD_TOTP_AUTH", raising=False)
        monkeypatch.delenv("MEMORY_OS_PASSWORD", raising=False)
        monkeypatch.delenv("MEMORY_OS_TOTP_SECRET", raising=False)
        monkeypatch.setenv("MEMORY_OS_PASSWORD", "hunter2")
        monkeypatch.setenv("MEMORY_OS_TOTP_SECRET", "JBSWY3DPEHPK3PXP")
        s = get_settings()
        # Wrong / missing TOTP → fail
        assert attempt_login(password="hunter2", totp_code="000000", token="", settings=s) is False
        assert attempt_login(password="hunter2", totp_code="", token="", settings=s) is False

    def test_off_ignores_password_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``PASSWORD_TOTP_AUTH=off`` makes the password+TOTP path inert.

        Even with ``MEMORY_OS_PASSWORD`` configured, a valid password
        alone is not enough — the legacy bearer-token path is the only
        way in.
        """
        monkeypatch.delenv("MEMORY_OS_TOKEN", raising=False)
        monkeypatch.delenv("MEMORY_OS_PASSWORD", raising=False)
        monkeypatch.setenv("PASSWORD_TOTP_AUTH", "off")
        monkeypatch.setenv("MEMORY_OS_TOKEN", "legacy-shared")
        monkeypatch.setenv("MEMORY_OS_PASSWORD", "hunter2")
        s = get_settings()
        assert s.password_totp_auth is False
        # Password alone: rejected (legacy path, token required).
        assert attempt_login(password="hunter2", totp_code="", token="", settings=s) is False
        # Bearer token: accepted.
        assert attempt_login(password="", totp_code="", token="legacy-shared", settings=s) is True
        # Wrong bearer token: rejected.
        assert attempt_login(password="hunter2", totp_code="", token="nope", settings=s) is False

    def test_off_auth_enabled_reflects_token_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """``auth_enabled`` is token-only when ``password_totp_auth=off`` and no token is set."""
        monkeypatch.delenv("MEMORY_OS_TOKEN", raising=False)
        monkeypatch.delenv("MEMORY_OS_PASSWORD", raising=False)
        monkeypatch.setenv("PASSWORD_TOTP_AUTH", "off")
        monkeypatch.setenv("MEMORY_OS_PASSWORD", "hunter2")
        # Deliberately no MEMORY_OS_TOKEN.
        s = get_settings()
        assert s.password_totp_auth is False
        # No token + flag-off => auth_enabled must be False even though
        # MEMORY_OS_PASSWORD is configured.
        assert s.auth_enabled is False


# ---------------------------------------------------------------------------
# SHADOW_ENABLED
# ---------------------------------------------------------------------------


class TestShadowEnabled:
    def test_default_on_surfaces_in_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Default (on) surfaces ``shadow_enabled=True`` in /api/strategy state."""
        monkeypatch.delenv("SHADOW_ENABLED", raising=False)
        with _client_for_env(monkeypatch) as c:
            r = c.get("/api/strategy")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["state"].get("shadow_enabled") is True

    def test_off_surfaces_in_strategy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """``SHADOW_ENABLED=off`` surfaces ``shadow_enabled=False`` in /api/strategy state."""
        monkeypatch.setenv("SHADOW_ENABLED", "off")
        with _client_for_env(monkeypatch) as c:
            r = c.get("/api/strategy")
        assert r.status_code == 200, r.text
        data = r.json()
        assert data["state"].get("shadow_enabled") is False


# ---------------------------------------------------------------------------
# Settings defaults sanity-check (defensive: catch silent regressions)
# ---------------------------------------------------------------------------


class TestSettingsDefaults:
    def test_all_flags_default_to_true(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """All flag defaults stay on so a missing env var preserves current behaviour."""
        for name in (
            "RETRIEVAL_ENGINE_V2",
            "STRUCTURED_FEEDBACK",
            "EVOLUTION_ENABLED",
            "SHADOW_ENABLED",
            "PASSWORD_TOTP_AUTH",
        ):
            monkeypatch.delenv(name, raising=False)
        reset_settings_cache()
        s = get_settings()
        assert s.retrieval_engine_v2 is True
        assert s.structured_feedback is True
        assert s.evolution_enabled is True
        assert s.shadow_enabled is True
        assert s.password_totp_auth is True
