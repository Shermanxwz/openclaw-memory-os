"""Tests for the auth helpers and token gate."""

from __future__ import annotations

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from openclaw_memory_os.auth import extract_token, require_auth, verify_token
from openclaw_memory_os.config import Settings


def _make_request(headers: dict | None = None, query: str = "", cookies: dict | None = None):
    """Build a minimal FastAPI ``Request``-like object for unit tests."""

    class _Req:
        def __init__(self):
            self.headers = headers or {}
            self.query_params = {}
            self.cookies = cookies or {}
            for kv in query.split("&"):
                if not kv:
                    continue
                k, _, v = kv.partition("=")
                self.query_params[k] = v

    return _Req()


def test_extract_token_from_header():
    req = _make_request({"Authorization": "Bearer secret123"})
    assert extract_token(req) == "secret123"


def test_extract_token_from_cookie():
    req = _make_request(cookies={"memory_os_session": "cookie-secret"})
    assert extract_token(req) == "cookie-secret"


def test_extract_token_query_is_rejected():
    """Query-string tokens are no longer accepted (privacy fix)."""
    req = _make_request(query="token=qs-secret")
    assert extract_token(req) is None


def test_extract_token_header_wins_over_cookie():
    req = _make_request(
        headers={"Authorization": "Bearer header-wins"},
        cookies={"memory_os_session": "cookie-secret"},
    )
    assert extract_token(req) == "header-wins"


def test_extract_token_none():
    req = _make_request()
    assert extract_token(req) is None


def test_verify_token_disabled_when_no_token():
    s = Settings(memory_os_token=None)
    assert s.auth_enabled is False
    assert verify_token("anything", s) is True
    assert verify_token(None, s) is True


def test_verify_token_rejects_wrong():
    s = Settings(memory_os_token="correct")
    assert verify_token("correct", s) is True
    assert verify_token("wrong", s) is False
    assert verify_token(None, s) is False
    assert verify_token("", s) is False


def test_require_auth_skips_when_disabled():
    # No token configured => require_auth is a no-op even without a request token.
    require_auth(_make_request())


def test_require_auth_blocks_when_enabled(monkeypatch):
    monkeypatch.setenv("MEMORY_OS_TOKEN", "secret")
    from openclaw_memory_os.config import get_settings
    get_settings.cache_clear()
    from openclaw_memory_os.auth import require_auth as _require
    with pytest.raises(HTTPException) as excinfo:
        _require(_make_request())
    assert excinfo.value.status_code == 401


def test_require_auth_passes_with_correct_token(monkeypatch):
    monkeypatch.setenv("MEMORY_OS_TOKEN", "secret")
    from openclaw_memory_os.config import get_settings
    get_settings.cache_clear()
    from openclaw_memory_os.auth import require_auth as _require
    _require(_make_request(headers={"Authorization": "Bearer secret"}))


def test_health_endpoint_is_public():
    """The /health endpoint is documented as auth-free. Verify via TestClient."""
    # Lazy import so unit tests above don't pull in FastAPI plumbing.
    from openclaw_memory_os.app import create_app

    app = create_app()
    with TestClient(app) as client:
        r = client.get("/health")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        # Privacy: /health must NOT leak backend / service / auth info.
        assert "service" not in body
        assert "backend" not in body
        assert "auth_enabled" not in body


# ---------------------------------------------------------------------------
# P0 regression tests for the password-mode authentication bypass.
#
# Before the fix, ``verify_token`` accepted ANY 32+ character string when
# ``MEMORY_OS_PASSWORD`` was set. The behaviour was equivalent to "auth
# disabled" but only for the routes guarded by ``require_auth``. Any
# attacker who knew ``auth_enabled=True`` was enough could send a garbage
# string and get in. These tests pin the fixed behaviour so the bypass
# cannot return.
# ---------------------------------------------------------------------------


def test_auth_password_not_accepted_as_bearer():
    """When MEMORY_OS_PASSWORD is set, verify_token does NOT accept the password as a bearer token.

    Password-as-bearer was removed in v0.3.0.x: the password is only
    accepted through the /login form (with optional TOTP).  This prevents
    long-lived bearer tokens that equal the login password.
    """
    s = Settings(
        memory_os_token=None,
        memory_os_password="super-secret-password-123",
        password_totp_auth=True,
    )
    assert verify_token("super-secret-password-123", s) is False


def test_auth_password_rejects_wrong_password():
    """A 32+ char wrong string must be rejected even when password is set."""
    s = Settings(
        memory_os_token=None,
        memory_os_password="the-real-password-12345",
        password_totp_auth=True,
    )
    # 40 chars long but not the configured password.
    assert verify_token("totally-wrong-but-long-enough-string-xyz", s) is False


def test_auth_password_rejects_random_32char_string():
    """The previous bypass accepted any string of length >= 32; pin that it does NOT.

    The provided string is intentionally a fresh secrets.token_hex-style
    value with no relationship to the configured password.
    """
    s = Settings(
        memory_os_token=None,
        memory_os_password="the-real-password-12345",
        password_totp_auth=True,
    )
    random_garbage = "9f3a8c1e7b6d4f2a05c819e3a7d6b8c1"  # 32 chars, unrelated
    assert len(random_garbage) >= 32
    assert verify_token(random_garbage, s) is False


def test_auth_password_totp_flag_off_falls_back_to_token(monkeypatch):
    """``PASSWORD_TOTP_AUTH=off`` makes the password path inert.

    With both ``MEMORY_OS_PASSWORD`` and ``MEMORY_OS_TOKEN`` configured
    and the flag turned off:

    * the bearer token ``MEMORY_OS_TOKEN`` is accepted (legacy path), and
    * the password is rejected even if it matches.

    This pins the rollback contract from :mod:`openclaw_memory_os.auth`.
    """
    monkeypatch.delenv("MEMORY_OS_TOKEN", raising=False)
    monkeypatch.delenv("MEMORY_OS_PASSWORD", raising=False)
    monkeypatch.setenv("MEMORY_OS_PASSWORD", "hunter2-password")
    monkeypatch.setenv("MEMORY_OS_TOKEN", "legacy-shared-bearer-token")
    monkeypatch.setenv("PASSWORD_TOTP_AUTH", "off")

    # Bust the cached settings so the monkeypatched env is picked up.
    from openclaw_memory_os.config import get_settings

    get_settings.cache_clear()
    try:
        s = Settings(
            memory_os_token="legacy-shared-bearer-token",
            memory_os_password="hunter2-password",
            password_totp_auth=False,
        )
        assert s.password_totp_auth is False
        # Bearer token path: still works.
        assert verify_token("legacy-shared-bearer-token", s) is True
        # Password path: rejected when the flag is off.
        assert verify_token("hunter2-password", s) is False
    finally:
        get_settings.cache_clear()
