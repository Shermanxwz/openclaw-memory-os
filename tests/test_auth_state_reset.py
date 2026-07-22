"""Tests for process-wide auth state reset between test cases.

Covers the _reset_auth_state_for_tests() path used by the conftest
autouse fixture, ensuring that module-level singletons and caches are
cleaned up properly so that subsequent tests start from a known state.
"""

from __future__ import annotations

import os

from openclaw_memory_os import auth as auth
from openclaw_memory_os.auth import (
    SessionStore,
    _reset_auth_state_for_tests,
    _revoked_sessions,
    get_session_store,
    revoke_session,
    verify_token,
)
from openclaw_memory_os.config import reset_settings_cache


def test_state_reset_full_sequence(tmp_path, monkeypatch):
    """9-step sequence: create store, revoke, reset, verify clean slate, re-auth."""

    # Redirect sessions DB to tmp_path for isolation
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(tmp_path / "sessions.db"))

    # 1. Create a SessionStore (forces module singleton init)
    store = get_session_store()
    assert isinstance(store, SessionStore)

    # Create a real session row so revoke_session can actually revoke it
    test_token = "test-session-token-for-reset"
    store.create(test_token, max_age=3600)

    # 2. Revoke the session token
    original_token = os.environ.get("MEMORY_OS_TOKEN")
    os.environ["MEMORY_OS_TOKEN"] = "secret"
    try:
        reset_settings_cache()

        result = revoke_session(test_token)
        assert result is True, "revoke_session should return True for valid session"

        # 3. Confirm _revoked_sessions contains the token
        assert test_token in _revoked_sessions, (
            f"Expected {test_token!r} in _revoked_sessions, got {_revoked_sessions}"
        )

        # 4. Call _reset_auth_state_for_tests()
        _reset_auth_state_for_tests()

        # 5. Confirm _revoked_sessions is empty
        assert len(_revoked_sessions) == 0, (
            f"Expected empty _revoked_sessions after reset, got {_revoked_sessions}"
        )

        # 6. Confirm global _session_store is None
        assert auth._session_store is None, (
            f"Expected _session_store is None after reset, got {auth._session_store}"
        )

        # 7. Set MEMORY_OS_TOKEN=secret (already set above)
        # 8. Clear Settings cache
        reset_settings_cache()

        # 9. Static bearer "secret" passes again after full reset
        verify_token("secret")
    finally:
        if original_token is None:
            os.environ.pop("MEMORY_OS_TOKEN", None)
        else:
            os.environ["MEMORY_OS_TOKEN"] = original_token
        reset_settings_cache()


def test_reset_auth_state_idempotent():
    """Calling _reset_auth_state_for_tests() multiple times is safe."""
    # First reset to ensure clean state
    _reset_auth_state_for_tests()

    # Second reset - should not raise
    _reset_auth_state_for_tests()

    # Third reset - still safe
    _reset_auth_state_for_tests()

    assert auth._session_store is None
    assert len(_revoked_sessions) == 0


def test_reset_closes_existing_store(tmp_path, monkeypatch):
    """_reset_auth_state_for_tests() closes any existing SessionStore."""
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(tmp_path / "sessions.db"))

    get_session_store()
    assert auth._session_store is not None

    _reset_auth_state_for_tests()

    assert auth._session_store is None
