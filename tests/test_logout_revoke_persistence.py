"""Test that logout revocations persist across service restarts.

This is the core regression test for the bug: logout wrote revoked=1
to the sessions DB, but the WAL was not flushed before the process
exited, so a restart would lose the revocation. The fix adds
close_session_store() to the lifespan shutdown handler, which flushes
the WAL cleanly.

These tests verify:
1. Logout writes revoked=1 to the persistent store.
2. After simulating a "restart" (close + reopen store), revoked=1 persists.
3. A revoked cookie is rejected after the simulated restart.
4. An active (non-revoked) session survives the simulated restart.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from openclaw_memory_os.auth import (
    close_session_store,
    revoke_session,
    verify_token,
    _set_session_store_for_tests,
    _revoked_sessions,
)
from openclaw_memory_os.sessions import SessionStore


@pytest.fixture(autouse=True)
def _reset_store():
    """Clean up the module-level store between tests."""
    prev = _set_session_store_for_tests(None)
    _revoked_sessions.clear()
    yield
    _set_session_store_for_tests(None)
    _revoked_sessions.clear()
    if prev is not None:
        try:
            prev.close()
        except Exception:
            pass


def _make_store(db_path: Path) -> SessionStore:
    """Create a SessionStore and set it as the module-level singleton."""
    store = SessionStore(db_path)
    _set_session_store_for_tests(store)
    return store


class TestLogoutRevokePersistence:
    """Revocations must survive a simulated restart (close + reopen)."""

    def test_revoke_persists_after_close_reopen(self, tmp_path: Path) -> None:
        """A revoked session should remain revoked after closing and
        reopening the store (simulating a process restart)."""
        db_path = tmp_path / "sessions.db"
        store = _make_store(db_path)

        # Create an active session.
        token = "test-session-token-abc"
        store.create(token, 43200)

        # Verify it's valid.
        assert store.is_valid(token)

        # Revoke it.
        revoked = revoke_session(token)
        assert revoked

        # Close the store (simulates process shutdown with WAL flush).
        close_session_store()

        # Reopen the store (simulates process restart).
        store2 = _make_store(db_path)

        # The revocation should persist.
        assert not store2.is_valid(token)

        # Clean up.
        close_session_store()

    def test_active_session_survives_close_reopen(self, tmp_path: Path) -> None:
        """An active (non-revoked) session should remain valid after
        closing and reopening the store."""
        db_path = tmp_path / "sessions.db"
        store = _make_store(db_path)

        # Create an active session.
        token = "active-session-token-xyz"
        store.create(token, 43200)

        # Verify it's valid.
        assert store.is_valid(token)

        # Close the store (simulates process shutdown with WAL flush).
        close_session_store()

        # Reopen the store (simulates process restart).
        store2 = _make_store(db_path)

        # The session should still be valid.
        assert store2.is_valid(token)

        # Clean up.
        close_session_store()

    def test_revoked_cookie_rejected_after_restart(self, tmp_path: Path) -> None:
        """A revoked cookie should be rejected by verify_token after a
        simulated restart, even though the in-memory revocation cache
        is empty."""
        db_path = tmp_path / "sessions.db"
        store = _make_store(db_path)

        # Create and revoke a session.
        token = "revoked-token-456"
        store.create(token, 43200)
        revoke_session(token)

        # Close and reopen (simulates restart, clears in-memory cache).
        close_session_store()
        _revoked_sessions.clear()  # Explicitly clear the in-memory cache.
        _make_store(db_path)  # Reopen store (simulates restart)

        # verify_token should reject the revoked token by checking
        # the persistent store, not just the in-memory cache.
        # Note: verify_token also checks MEMORY_OS_TOKEN, so we need
        # to ensure the token doesn't match that.
        result = verify_token(token, _make_no_token_settings())
        assert not result, "Revoked token should be rejected after restart"

        # Clean up.
        close_session_store()

    def test_multiple_revocations_persist(self, tmp_path: Path) -> None:
        """Multiple revocations should all persist across a restart."""
        db_path = tmp_path / "sessions.db"
        store = _make_store(db_path)

        tokens = [f"token-{i}" for i in range(5)]
        for token in tokens:
            store.create(token, 43200)

        # Revoke the first 3.
        for token in tokens[:3]:
            revoke_session(token)

        # Close and reopen.
        close_session_store()
        _revoked_sessions.clear()
        store2 = _make_store(db_path)

        # First 3 should be revoked.
        for token in tokens[:3]:
            assert not store2.is_valid(token), f"{token} should be revoked"

        # Last 2 should still be valid.
        for token in tokens[3:]:
            assert store2.is_valid(token), f"{token} should be valid"

        # Clean up.
        close_session_store()


class TestWALFlushOnShutdown:
    """Verify that close_session_store actually flushes the WAL."""

    def test_wal_data_in_main_db_after_close(self, tmp_path: Path) -> None:
        """After close_session_store, data written to the WAL should
        be visible in the main DB file."""
        db_path = tmp_path / "sessions.db"
        store = _make_store(db_path)

        # Create a session (this goes to the WAL).
        token = "wal-test-token"
        store.create(token, 43200)

        # Close the store (should flush WAL).
        close_session_store()

        # Open a fresh connection (not through SessionStore) to verify
        # the data is in the main DB file.
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        conn.close()
        assert count == 1, "Session data should be in the main DB after close"

    def test_revocation_in_main_db_after_close(self, tmp_path: Path) -> None:
        """After close_session_store, a revocation should be in the
        main DB file."""
        db_path = tmp_path / "sessions.db"
        store = _make_store(db_path)

        token = "revoke-wal-token"
        store.create(token, 43200)
        revoke_session(token)

        # Close the store (should flush WAL with the revocation).
        close_session_store()

        # Verify the revocation is in the main DB.
        conn = sqlite3.connect(str(db_path))
        token_hash = conn.execute(
            "SELECT token_hash FROM sessions WHERE revoked=1"
        ).fetchone()
        conn.close()
        assert token_hash is not None, "Revocation should be in the main DB"


def _make_no_token_settings():
    """Create a Settings-like object with auth enabled but no static token."""
    from unittest.mock import MagicMock
    settings = MagicMock()
    settings.auth_enabled = True
    settings.memory_os_token = ""
    settings.memory_os_password = ""
    settings.memory_os_totp_secret = ""
    return settings
