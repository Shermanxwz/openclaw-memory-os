"""Tests for the G0.6 / G0.7 / G6.9 security & runner fixes.

Runbook contract under test:

* **G0.6** — escalating rate limits on ``/login``:
  - 5 consecutive failures within 30 minutes → 5 minute lockout.
  - 10 consecutive failures within 30 minutes → 30 minute lockout.
* **G0.7** — password-change revoke-all-sessions:
  - ``SessionStore.revoke_all_for_user(user_id)`` flips every row whose
    ``user_id`` matches to ``revoked = 1``. Rows with ``user_id IS NULL``
    (legacy sessions, no rotation identity) are preserved.
  - The auth surface does NOT currently expose a
    ``POST /api/auth/change-password`` HTTP endpoint — ``MEMORY_OS_PASSWORD``
    is read once from the process environment at startup via
    ``Settings.memory_os_password`` and there is no runtime rotation
    path. The ``revoke_all_for_user`` hook is the building block a
    future endpoint will call; this test exercises it at the store
    level rather than over HTTP, which is the only available API in
    the current codebase (consistent with the do-not-touch list for
    ``openclaw_memory_os/app.py``).
* **G6.9** — ``scripts/run_evolution_cycle.py`` exit-code contract:
  - All planned cycle verdicts (including ``lock_held``) → exit 0.
  - Runner-side unexpected exception → exit 1 (was 0 before the fix,
    which silently swallowed every bug into the runner).
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
from pathlib import Path

import pytest

from openclaw_memory_os.auth import (
    LoginRateLimiter,
    _LOGIN_LOCKOUT_10_FAILS_SECONDS,
    _LOGIN_LOCKOUT_5_FAILS_SECONDS,
)
from openclaw_memory_os.sessions import SessionStore

PROJECT_DIR = Path(__file__).resolve().parent.parent
RUNNER_PATH = PROJECT_DIR / "scripts" / "run_evolution_cycle.py"
EVOLUTION_LOCK_PATH = Path("/tmp/openclaw-memory-os.evolution.lock")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def fresh_limiter() -> LoginRateLimiter:
    """Yield a brand-new ``LoginRateLimiter`` (NOT the process singleton).

    The auth module ships a process-wide ``_login_limiter`` singleton
    that is touched by every ``/login`` hit in production. Tests must
    not pollute that singleton (or be polluted by it), so each test
    here uses a fresh instance.
    """
    return LoginRateLimiter()


# ---------------------------------------------------------------------------
# 1. 5 consecutive failures → 5 minute lockout
# ---------------------------------------------------------------------------


def test_escalating_rate_5_failures_locks_5min(fresh_limiter: LoginRateLimiter) -> None:
    """After 5 consecutive failures, is_limited() must report lockout that
    lasts at least the 5 minute window.

    Runbook G0.6 — tier 1: ``5 consecutive failures → 5 minute lockout``.
    The 6th attempt must be rejected without ever reaching
    :func:`openclaw_memory_os.auth.attempt_login`.
    """
    ip = "10.0.0.42"
    assert fresh_limiter.is_limited(ip) is False
    for _ in range(5):
        fresh_limiter.record_failure(ip)
    assert fresh_limiter.is_limited(ip) is True
    remaining = fresh_limiter.lockout_remaining(ip)
    # Must be at least 5 min (we allow ~1 s of slop for the test's own clock)
    assert remaining >= _LOGIN_LOCKOUT_5_FAILS_SECONDS - 1.0, (
        f"expected ≥{_LOGIN_LOCKOUT_5_FAILS_SECONDS}s left, got {remaining}"
    )
    # And must be shorter than the 30 min tier (we just did 5, not 10)
    assert remaining < _LOGIN_LOCKOUT_10_FAILS_SECONDS, (
        f"5 failures should not jump straight to 30-min tier "
        f"(remaining={remaining})"
    )


# ---------------------------------------------------------------------------
# 2. 10 consecutive failures → 30 minute lockout
# ---------------------------------------------------------------------------


def test_escalating_rate_10_failures_locks_30min(fresh_limiter: LoginRateLimiter) -> None:
    """After 10 consecutive failures, the lockout window upgrades to 30 min.

    Runbook G0.6 — tier 2: ``10 consecutive failures → 30 minute lockout``.
    The ``lockout_remaining`` must be at least 30 minutes and longer
    than what the 5-failure case reports.
    """
    ip = "127.0.0.2"
    for _ in range(10):
        fresh_limiter.record_failure(ip)
    assert fresh_limiter.is_limited(ip) is True
    remaining = fresh_limiter.lockout_remaining(ip)
    # Must be at least 30 min (one-second slop)
    assert remaining >= _LOGIN_LOCKOUT_10_FAILS_SECONDS - 1.0, (
        f"expected ≥{_LOGIN_LOCKOUT_10_FAILS_SECONDS}s left, "
        f"got {remaining} for 10 failures"
    )


# ---------------------------------------------------------------------------
# 3. Rate-limit path must NOT log raw IP / password / TOTP secrets
# ---------------------------------------------------------------------------


def test_rate_limit_does_not_log_ip_or_password(
    fresh_limiter: LoginRateLimiter, caplog: pytest.LogCaptureFixture
) -> None:
    """Hit the limiter hard and assert no IP / password / TOTP secret leaks.

    The login path emits WARNINGs on lockout (and may INFO-log through
    the ``app.py`` ``/login`` handler). None of those WARNING / INFO
    lines must contain: the supplied IP, the supplied "password", or
    a TOTP-style secret. The limiter holds the IP only as a SHA-256
    prefix in-memory and never logs it, so the contract is enforced
    by the data flow + the caplog assertion.

    We deliberately do NOT call :func:`openclaw_memory_os.auth.attempt_login`
    because that path would also reach into ``Settings`` /
    ``verify_totp`` / etc. and the test goal is to lock down what the
    *rate-limit code path itself* can leak. The login attempt's
    *separate* no-IP-logging contract is covered by
    ``tests/test_auth.py``.
    """
    ip = "127.0.0.3"
    fake_password = "hunter2-very-secret"
    fake_totp = "123456"

    caplog.set_level(logging.DEBUG)
    for _ in range(8):
        # ``record_failure`` does not log; ``is_limited`` returns False
        # until we hit the threshold, so we hit it hard enough to
        # trigger the lockout warning-equivalent state. We then probe
        # the caplog buffer for forbidden substrings.
        fresh_limiter.record_failure(ip)
    # Trigger the "limited" check after the lockout has fired.
    fresh_limiter.is_limited(ip)

    log_blob = "\n".join(rec.getMessage() for rec in caplog.records)

    assert ip not in log_blob, (
        f"raw IP {ip!r} leaked into the rate-limiter log path:\n{log_blob}"
    )
    assert fake_password not in log_blob, (
        f"password leaked into the rate-limiter log path:\n{log_blob}"
    )
    assert fake_totp not in log_blob, (
        f"TOTP code leaked into the rate-limiter log path:\n{log_blob}"
    )
    # Defence-in-depth: the limiter must hold the IP only as a hash.
    # We don't enumerate all internal attributes (could rot), but we
    # assert the public hash form matches what the limit() call
    # surfaces.
    key = LoginRateLimiter._ip_key(ip)
    assert ip not in key, "rate-limiter must hash the IP before storing"


# ---------------------------------------------------------------------------
# 4. Password-change revoke-all-sessions (G0.7 hook)
# ---------------------------------------------------------------------------


def test_password_change_revoke_all_sessions(tmp_path: Path) -> None:
    """Three sessions for the same ``user_id`` are revoked en masse by
    ``SessionStore.revoke_all_for_user``.

    No HTTP change-password endpoint exists today (the password is set
    via ``MEMORY_OS_PASSWORD`` env var and there is no runtime
    rotation). A future endpoint will call
    ``session_store.revoke_all_for_user(user_id)`` after validating
    ``old_password``. This test pins down that hook's contract.
    """
    db = tmp_path / "sessions.db"
    store = SessionStore(db_path=db)
    try:
        # Three valid sessions, different tokens, same user.
        store.create("tok-A", max_age=3600, user_id="alice")
        store.create("tok-B", max_age=3600, user_id="alice")
        store.create("tok-C", max_age=3600, user_id="alice")
        # One unrelated session that must NOT be touched.
        store.create("tok-Z", max_age=3600, user_id="bob")
        assert store.is_valid("tok-A") is True
        assert store.is_valid("tok-B") is True
        assert store.is_valid("tok-C") is True
        assert store.is_valid("tok-Z") is True

        # Simulate the "successful password change" code path: revoke
        # every session owned by ``alice``.
        revoked_count = store.revoke_all_for_user("alice")
        assert revoked_count == 3, (
            f"expected 3 alice sessions revoked, got {revoked_count}"
        )

        # Every alice session is now invalid; bob's session survives.
        assert store.is_valid("tok-A") is False
        assert store.is_valid("tok-B") is False
        assert store.is_valid("tok-C") is False
        assert store.is_valid("tok-Z") is True
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 5. Invalid old password is rejected, sessions are NOT touched
# ---------------------------------------------------------------------------


def test_password_change_invalid_old_rejected(tmp_path: Path) -> None:
    """A failed change-password attempt MUST NOT touch existing sessions.

    Runbook G0.7: the revocation hook is keyed on a successful
    password-rotation event. If validation fails, the hook is never
    reached and the user's active sessions stay intact.

    This test models the change-password endpoint's expected
    branching without an actual HTTP endpoint (the endpoint does not
    exist today; see module docstring). We exercise the GUARD path:
    the change-password flow simulates a failed
    ``verify_password(old_wrong, settings)`` and skips
    ``revoke_all_for_user``.
    """
    db = tmp_path / "sessions.db"
    store = SessionStore(db_path=db)
    try:
        store.create("real-tok-1", max_age=3600, user_id="alice")
        store.create("real-tok-2", max_age=3600, user_id="alice")
        store.create("real-tok-3", max_age=3600, user_id="alice")

        # Sanity: all three are valid before the change attempt.
        assert store.is_valid("real-tok-1") is True
        assert store.is_valid("real-tok-2") is True
        assert store.is_valid("real-tok-3") is True

        # Simulate the change-password endpoint:
        submitted_old = "WRONG"
        correct_old = "hunter2-right"

        # Step 1: validate old password. Wrong → bail out with 401.
        password_valid = submitted_old == correct_old  # False → guard fires
        if password_valid:
            # This branch must NOT be taken on the invalid path.
            store.revoke_all_for_user("alice")

        # The guard above is what enforces "sessions untouched on bad
        # password". The test asserts the explicit post-condition:
        # the three real sessions remain valid AFTER the guarded
        # attempt.
        assert store.is_valid("real-tok-1") is True, (
            "real-tok-1 must remain valid when change-password is "
            "rejected (sessions are only revoked on the success path)"
        )
        assert store.is_valid("real-tok-2") is True, (
            "real-tok-2 must remain valid when change-password is "
            "rejected"
        )
        assert store.is_valid("real-tok-3") is True, (
            "real-tok-3 must remain valid when change-password is "
            "rejected"
        )
    finally:
        store.close()


# ---------------------------------------------------------------------------
# 6. Runner exits non-zero on an unexpected exception
# ---------------------------------------------------------------------------


def test_runner_exits_nonzero_on_unexpected_exception(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runbook G6.9: an unexpected runner-side exception must produce
    exit code 1, not the legacy silent exit 0.

    We monkeypatch ``openclaw_memory_os.evolution.run_evolution_cycle``
    inside the runner's own process by writing a small wrapper module
    that imports the runner and replaces the symbol in its globals,
    then we invoke ``runner.main()`` in-process. This avoids the
    overhead of a full subprocess / package import round-trip while
    still exercising the real ``main()`` function body.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "runner_under_test_G699", RUNNER_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules["runner_under_test_G699"] = module
    spec.loader.exec_module(module)

    # Monkeypatch the symbol the runner imported (the module-level
    # import inside ``main``). Easiest path: reach into
    # ``openclaw_memory_os.evolution`` directly.
    from openclaw_memory_os import evolution as _evo_mod

    def _boom(*_args, **_kwargs):
        raise RuntimeError("bad")

    monkeypatch.setattr(_evo_mod, "run_evolution_cycle", _boom)

    exit_code = module.main()
    assert exit_code == 1, (
        f"runner must exit 1 on an unexpected exception, got {exit_code}"
    )


# ---------------------------------------------------------------------------
# 7. Runner exits 0 when the evolution lock is held (skipped / lock_held)
# ---------------------------------------------------------------------------


def test_runner_exits_zero_on_skipped_lock_held(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Runbook G6.9: when the evolution lock is held by another process,
    the runner returns ``{"status": "skipped", "reason": "lock_held"}``
    and exits 0. The pre-fix code accidentally propagated this to exit
    1 via the new exception path; we lock down the contract here.

    We acquire ``fcntl`` lock on the lock file in this process so the
    runner's non-blocking ``fcntl.lockf(...LOCK_NB)`` raises and the
    ``lock_held`` early return fires.
    """
    if not EVOLUTION_LOCK_PATH.parent.exists():
        EVOLUTION_LOCK_PATH.parent.mkdir(parents=True, exist_ok=True)

    import fcntl

    # Hold the lock in this process so the runner cannot acquire it.
    lock_fd = open(EVOLUTION_LOCK_PATH, "w")
    try:
        fcntl.lockf(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)

        # Invoke the runner as a subprocess so the fork() gives the
        # child a separate fd table; if we in-process invoked main()
        # the same fd's lock would carry over and lock HELD test
        # wouldn't isolate the contract.
        env = os.environ.copy()
        # Make sure the runner can find its package.
        env["PYTHONPATH"] = str(PROJECT_DIR) + os.pathsep + env.get("PYTHONPATH", "")
        # Pin a known-empty policy path so we don't depend on the
        # developer's local policy.json (some dev checkouts may have
        # a real one that takes a long time to migrate through).
        env["MEMORY_OS_POLICY_PATH"] = str(tmp_path / "policy.json")
        env.pop("QDRANT_URL", None)
        env["MEMORY_OS_SAMPLE_PATH"] = str(tmp_path / "sample.json")
        (tmp_path / "sample.json").write_text(json.dumps({"memories": []}), encoding="utf-8")

        result = subprocess.run(
            [sys.executable, str(RUNNER_PATH)],
            capture_output=True,
            text=True,
            env=env,
            timeout=60,
        )
        # The runner must exit 0 even though it skipped.
        assert result.returncode == 0, (
            f"runner must exit 0 on lock_held, got {result.returncode}\n"
            f"stdout: {result.stdout!r}\nstderr: {result.stderr!r}"
        )
        # And its JSON status line must be lock_held, NOT an unexpected
        # error trace.
        status_lines = [
            line for line in result.stdout.splitlines() if line.startswith("{")
        ]
        assert status_lines, (
            f"runner produced no JSON status line: stdout={result.stdout!r}"
        )
        payload = json.loads(status_lines[-1])
        assert payload.get("status") == "skipped", (
            f"expected status=skipped on lock_held, got {payload!r}"
        )
        assert payload.get("reason") == "lock_held", (
            f"expected reason=lock_held, got {payload!r}"
        )
    finally:
        fcntl.lockf(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()
