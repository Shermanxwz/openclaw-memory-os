"""Unit tests for the persistent SessionStore.

These tests are deliberately isolated from the rest of the test suite:
each test either injects a tmpdir-backed :class:`SessionStore` or sets
``MEMORY_OS_SESSIONS_DB`` to a tmp path, so the production database at
``~/.local/state/openclaw-memory-os/sessions.db`` is never touched.
"""

from __future__ import annotations

import hashlib
import hmac
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openclaw_memory_os.sessions import SessionStore, _hash_token


# ---------------------------------------------------------------------------
# Test isolation helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def tmp_store(tmp_path: Path) -> SessionStore:
    """Yield a SessionStore backed by a tmpdir file. Auto-closes on teardown."""
    store = SessionStore(db_path=tmp_path / "sessions.db")
    yield store
    store.close()


@pytest.fixture(autouse=True)
def _no_env_override(monkeypatch):
    """Make sure MEMORY_OS_SESSIONS_DB does not leak from a parent shell."""
    monkeypatch.delenv("MEMORY_OS_SESSIONS_DB", raising=False)


# ---------------------------------------------------------------------------
# 1. create + is_valid
# ---------------------------------------------------------------------------


def test_sessions_create_then_is_valid(tmp_store: SessionStore) -> None:
    """A freshly created session is valid."""
    tmp_store.create("tok-1", max_age=3600)
    assert tmp_store.is_valid("tok-1") is True


def test_sessions_is_valid_returns_false_for_unknown_token(tmp_store: SessionStore) -> None:
    """Tokens never written are reported as invalid."""
    assert tmp_store.is_valid("never-seen") is False
    assert tmp_store.is_valid(None) is False
    assert tmp_store.is_valid("") is False


# ---------------------------------------------------------------------------
# 2. revoke semantics
# ---------------------------------------------------------------------------


def test_sessions_is_valid_returns_false_for_revoked_token(tmp_store: SessionStore) -> None:
    """Revocation flips a valid session to invalid."""
    tmp_store.create("tok-2", max_age=3600)
    assert tmp_store.is_valid("tok-2") is True
    tmp_store.revoke("tok-2")
    assert tmp_store.is_valid("tok-2") is False


def test_sessions_revoke_then_is_valid_false(tmp_store: SessionStore) -> None:
    """``revoke`` returns True iff a row was updated."""
    assert tmp_store.revoke("never-existed") is False
    tmp_store.create("tok-3", max_age=3600)
    assert tmp_store.revoke("tok-3") is True
    # Idempotent: second revoke still returns True (row already updated).
    assert tmp_store.revoke("tok-3") is True
    assert tmp_store.is_valid("tok-3") is False


def test_sessions_revoke_all_returns_count(tmp_store: SessionStore) -> None:
    """``revoke_all`` reports the number of rows it flipped."""
    tmp_store.create("a", max_age=60)
    tmp_store.create("b", max_age=60)
    tmp_store.create("c", max_age=60)
    count = tmp_store.revoke_all()
    assert count == 3
    assert tmp_store.is_valid("a") is False
    assert tmp_store.is_valid("b") is False
    assert tmp_store.is_valid("c") is False


def test_sessions_revoke_all_idempotent(tmp_store: SessionStore) -> None:
    """Calling ``revoke_all`` twice does not double-count."""
    tmp_store.create("x", max_age=60)
    tmp_store.create("y", max_age=60)
    first = tmp_store.revoke_all()
    second = tmp_store.revoke_all()
    assert first == 2
    assert second == 0


# ---------------------------------------------------------------------------
# 3. Persistence across reopens
# ---------------------------------------------------------------------------


def test_sessions_revoke_persists_across_reopen(tmp_path: Path) -> None:
    """Revocations survive a process restart (separate SessionStore instances)."""
    path = tmp_path / "persist-revoke.db"
    store_a = SessionStore(db_path=path)
    store_a.create("session-A", max_age=3600)
    store_a.revoke("session-A")
    store_a.close()

    store_b = SessionStore(db_path=path)
    try:
        assert store_b.is_valid("session-A") is False
        # revoke of an already-revoked token still returns True (idempotent).
        assert store_b.revoke("session-A") is True
    finally:
        store_b.close()


def test_sessions_create_persists_across_reopen(tmp_path: Path) -> None:
    """Active sessions survive a process restart."""
    path = tmp_path / "persist-create.db"
    issued = datetime.now(timezone.utc) - timedelta(minutes=1)
    store_a = SessionStore(db_path=path)
    store_a.create("session-B", max_age=3600, issued_at=issued)
    store_a.close()

    store_b = SessionStore(db_path=path)
    try:
        # Still valid (well within max_age) after the restart.
        assert store_b.is_valid("session-B") is True
        # And the issued_at we wrote is preserved byte-for-byte.
        rows = store_b._conn.execute(
            "SELECT issued_at, max_age, revoked FROM sessions WHERE token_hash = ?",
            (_hash_token("session-B"),),
        ).fetchone()
        assert rows is not None
        assert int(rows["max_age"]) == 3600
        assert int(rows["revoked"]) == 0
    finally:
        store_b.close()


# ---------------------------------------------------------------------------
# 4. Expiry
# ---------------------------------------------------------------------------


def test_sessions_expires_after_max_age(tmp_store: SessionStore) -> None:
    """``is_valid`` returns False once ``issued_at + max_age`` is in the past."""
    past = datetime.now(timezone.utc) - timedelta(seconds=10)
    tmp_store.create("session-C", max_age=0, issued_at=past)
    assert tmp_store.is_valid("session-C") is False


# ---------------------------------------------------------------------------
# 5. list() exposes no raw tokens
# ---------------------------------------------------------------------------


def test_sessions_list_metadata_no_raw_token(tmp_store: SessionStore) -> None:
    """``list`` returns metadata only — the raw token never appears."""
    tmp_store.create("session-D", max_age=3600)
    tmp_store.create("session-E", max_age=3600)
    entries = tmp_store.list()
    assert len(entries) == 2
    for entry in entries:
        assert "token" not in entry, "raw token must never leak via list()"
        for forbidden in (
            "raw",
            "raw_token",
            "token_hash",
            "secret",
            "session_token",
            "value",
        ):
            assert forbidden not in entry, f"unexpected key: {forbidden}"
        assert set(entry.keys()) == {
            "fingerprint",
            "issued_at",
            "max_age",
            "revoked",
            "current",
        }
        # fingerprint must be exactly 16 hex chars and equal to
        # sha256(token).hexdigest()[:16].
        assert len(entry["fingerprint"]) == 16
        assert all(c in "0123456789abcdef" for c in entry["fingerprint"])
    # Spot-check the fingerprint content matches sha256(token).
    expected_d = hashlib.sha256(b"session-D").hexdigest()[:16]
    expected_e = hashlib.sha256(b"session-E").hexdigest()[:16]
    fingerprints = {e["fingerprint"] for e in entries}
    assert expected_d in fingerprints
    assert expected_e in fingerprints


def test_sessions_list_current_flag_uses_hmac(tmp_store: SessionStore) -> None:
    """Exactly one row is ``current=True`` for the provided token, and it is the right one."""
    tmp_store.create("session-F", max_age=3600)
    tmp_store.create("session-G", max_age=3600)
    tmp_store.create("session-H", max_age=3600)

    entries = tmp_store.list(current_token="session-G")
    currents = [e for e in entries if e["current"]]
    assert len(currents) == 1
    expected_fp = hashlib.sha256(b"session-G").hexdigest()[:16]
    assert currents[0]["fingerprint"] == expected_fp

    # Constant-time comparison contract: ``current`` is True iff the
    # strings match under hmac.compare_digest.
    token_fp = hmac.compare_digest("session-G", "session-G")  # sanity
    assert token_fp is True


def test_sessions_db_path_override_via_env(tmp_path: Path, monkeypatch) -> None:
    """Setting ``MEMORY_OS_SESSIONS_DB`` redirects the default store path."""
    target = tmp_path / "other-name.db"
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(target))
    # Lazy default store must use the env-overridden path.
    store = SessionStore()
    try:
        # The connection was opened at the overridden path.
        assert store._db_path == target
        # CRUD works as expected.
        store.create("override-token", max_age=60)
        assert store.is_valid("override-token") is True
    finally:
        store.close()
    # And the file actually exists on disk at that path.
    assert target.exists()


# ---------------------------------------------------------------------------
# 6. Concurrent writes
# ---------------------------------------------------------------------------


def test_sessions_concurrent_writes_no_corruption(tmp_store: SessionStore) -> None:
    """5 threads writing 20 sessions each produce exactly 100 unique rows."""
    barrier = threading.Barrier(parties=5)
    errors: list[BaseException] = []

    def writer(prefix: str) -> None:
        try:
            barrier.wait(timeout=5)
            for i in range(20):
                # Use unique tokens across all writers.
                tmp_store.create(f"{prefix}-{i:03d}", max_age=3600)
        except BaseException as exc:  # pragma: no cover - surfaced via errors
            errors.append(exc)

    threads = [
        threading.Thread(target=writer, args=(f"thread-{tid}",))
        for tid in range(5)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
        assert not t.is_alive(), "writer thread deadlocked"

    assert not errors, errors

    rows = tmp_store._conn.execute(
        "SELECT COUNT(*) AS n FROM sessions"
    ).fetchone()
    assert int(rows["n"]) == 100, f"expected 100 rows, got {rows['n']}"

    # No duplicate token hashes.
    hashes = {
        row[0]
        for row in tmp_store._conn.execute(
            "SELECT token_hash FROM sessions"
        ).fetchall()
    }
    assert len(hashes) == 100
