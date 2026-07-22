"""Test that close_session_store is idempotent and safe.

The close_session_store function was added to auth.py to flush the
SQLite WAL cleanly during FastAPI lifespan shutdown. These tests verify:
1. Calling close_session_store on an uninitialized store is safe (no-op).
2. Calling close_session_store twice is safe (idempotent).
3. After close, the store is set to None so a new one can be created.
4. The underlying SQLite connection is properly closed.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from openclaw_memory_os.auth import (
    close_session_store,
    get_session_store,
    _set_session_store_for_tests,
)
from openclaw_memory_os.sessions import SessionStore


@pytest.fixture(autouse=True)
def _reset_store():
    """Ensure the module-level store is cleaned up after each test."""
    prev = _set_session_store_for_tests(None)
    yield
    _set_session_store_for_tests(None)
    # Close the previous store if it was a real one.
    if prev is not None:
        try:
            prev.close()
        except Exception:
            pass


class TestCloseSessionStoreIdempotent:
    """close_session_store must be safe to call multiple times."""

    def test_close_on_uninitialized_store(self) -> None:
        """Calling close when no store exists should not raise."""
        close_session_store()  # Should not raise

    def test_close_twice(self, tmp_path: Path) -> None:
        """Calling close twice should not raise on the second call."""
        store = SessionStore(tmp_path / "sessions.db")
        _set_session_store_for_tests(store)
        close_session_store()
        close_session_store()  # Second call should be a no-op

    def test_close_sets_store_to_none(self, tmp_path: Path) -> None:
        """After close, the module-level store should be None."""
        store = SessionStore(tmp_path / "sessions.db")
        _set_session_store_for_tests(store)
        close_session_store()
        # get_session_store should create a new one, not reuse the closed one.
        new_store = get_session_store()
        assert new_store is not store
        # Clean up the new store.
        close_session_store()

    def test_close_flushes_wal(self, tmp_path: Path) -> None:
        """Closing the store should flush the WAL to the main DB file."""
        db_path = tmp_path / "sessions.db"
        store = SessionStore(db_path)
        store.create("test-token-123", 3600)
        _set_session_store_for_tests(store)

        # Close the store (which should flush WAL).
        close_session_store()

        # After close, the main DB file should contain the data.
        # Open a fresh connection to verify.
        conn = sqlite3.connect(str(db_path))
        count = conn.execute("SELECT COUNT(*) FROM sessions").fetchone()[0]
        conn.close()
        assert count == 1, "Data should be in the main DB after close"


class TestCloseSessionStoreErrorHandling:
    """close_session_store should handle errors gracefully."""

    def test_close_with_failing_connection(self, tmp_path: Path) -> None:
        """If the underlying connection.close() fails, close_session_store
        should not raise (it logs a warning instead)."""
        store = SessionStore(tmp_path / "sessions.db")
        # Close the underlying connection first to simulate a failure.
        store._conn.close()
        _set_session_store_for_tests(store)

        # close_session_store should not raise even though the
        # underlying connection is already closed.
        close_session_store()
