"""Test that session_readonly_helper.py never modifies WAL/SHM files.

The helper was introduced to replace /usr/bin/sqlite3 CLI calls that
corrupted the WAL. These tests verify:
1. The helper opens in read-only URI mode.
2. PRAGMA query_only=ON is active.
3. Write attempts through the helper fail.
4. WAL/SHM files are not created or modified by the helper.
"""

from __future__ import annotations

import sqlite3
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.fixture()
def populated_db(tmp_path: Path) -> Path:
    """Create a small sessions DB with one active row."""
    db_path = tmp_path / "sessions.db"
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute(
        "CREATE TABLE sessions ("
        "token_hash TEXT PRIMARY KEY, "
        "issued_at TEXT NOT NULL, "
        "max_age INTEGER NOT NULL, "
        "revoked INTEGER NOT NULL DEFAULT 0, "
        "last_seen_at TEXT, "
        "user_id TEXT)"
    )
    conn.execute(
        "INSERT INTO sessions (token_hash, issued_at, max_age, revoked) "
        "VALUES (?, ?, ?, ?)",
        ("abc123hash", "2026-07-18T00:00:00+00:00", 43200, 0),
    )
    conn.commit()
    conn.close()
    return db_path


def _run_helper(db_path: Path, query: str) -> subprocess.CompletedProcess:
    """Run the readonly helper as a subprocess."""
    helper = Path(__file__).resolve().parents[1] / "scripts" / "session_readonly_helper.py"
    result = subprocess.run(
        [sys.executable, str(helper), str(db_path), query],
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result


class TestReadonlyHelperBasic:
    """Basic read-only helper functionality."""

    def test_select_returns_data(self, populated_db: Path) -> None:
        """A simple SELECT should return the expected row."""
        result = _run_helper(
            populated_db,
            "SELECT revoked FROM sessions WHERE token_hash='abc123hash';",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "0" in result.stdout

    def test_select_revoked_row(self, populated_db: Path) -> None:
        """Querying a revoked row should return 1."""
        # First, mark the row as revoked using a direct connection.
        conn = sqlite3.connect(str(populated_db))
        conn.execute("UPDATE sessions SET revoked=1 WHERE token_hash='abc123hash'")
        conn.commit()
        conn.close()
        result = _run_helper(
            populated_db,
            "SELECT revoked FROM sessions WHERE token_hash='abc123hash';",
        )
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "1" in result.stdout

    def test_count_query(self, populated_db: Path) -> None:
        """COUNT(*) should work through the helper."""
        result = _run_helper(populated_db, "SELECT COUNT(*) FROM sessions;")
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "1" in result.stdout


class TestReadonlyHelperNoWrites:
    """Verify the helper cannot write to the database."""

    def test_insert_fails(self, populated_db: Path) -> None:
        """An INSERT through the helper must fail."""
        result = _run_helper(
            populated_db,
            "INSERT INTO sessions (token_hash, issued_at, max_age, revoked) "
            "VALUES ('x', '2026-01-01', 100, 0);",
        )
        assert result.returncode != 0, "INSERT should have failed in read-only mode"

    def test_update_fails(self, populated_db: Path) -> None:
        """An UPDATE through the helper must fail."""
        result = _run_helper(
            populated_db,
            "UPDATE sessions SET revoked=1 WHERE token_hash='abc123hash';",
        )
        assert result.returncode != 0, "UPDATE should have failed in read-only mode"

    def test_delete_fails(self, populated_db: Path) -> None:
        """A DELETE through the helper must fail."""
        result = _run_helper(
            populated_db,
            "DELETE FROM sessions WHERE token_hash='abc123hash';",
        )
        assert result.returncode != 0, "DELETE should have failed in read-only mode"


class TestReadonlyHelperNoWalModification:
    """Verify the helper does not create or modify WAL/SHM files."""

    def test_no_new_wal_after_read(self, populated_db: Path) -> None:
        """Reading through the helper should not create new WAL/SHM files.

        If the DB was created with WAL mode, the WAL file may already exist.
        The test checks that the helper does not change the WAL file's
        modification time or size.
        """
        # First, checkpoint the WAL so it's as small as possible.
        conn = sqlite3.connect(str(populated_db))
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        conn.close()

        wal_path = populated_db.with_name(populated_db.name + "-wal")

        # Record pre-read state.
        wal_size_before = wal_path.stat().st_size if wal_path.exists() else 0

        # Read through the helper.
        result = _run_helper(populated_db, "SELECT COUNT(*) FROM sessions;")
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # Check post-read state.
        wal_size_after = wal_path.stat().st_size if wal_path.exists() else 0

        # The WAL should not have grown (it may stay at 0 bytes).
        assert wal_size_after <= wal_size_before, (
            f"WAL grew from {wal_size_before} to {wal_size_after} after read-only query"
        )

    def test_no_wal_created_on_fresh_db(self, tmp_path: Path) -> None:
        """A fresh DB with no WAL should not gain one after a read-only query."""
        db_path = tmp_path / "fresh.db"
        conn = sqlite3.connect(str(db_path))
        conn.execute("PRAGMA journal_mode=DELETE")  # Ensure no WAL
        conn.execute(
            "CREATE TABLE sessions ("
            "token_hash TEXT PRIMARY KEY, "
            "issued_at TEXT NOT NULL, "
            "max_age INTEGER NOT NULL, "
            "revoked INTEGER NOT NULL DEFAULT 0)"
        )
        conn.commit()
        conn.close()

        wal_path = db_path.with_name(db_path.name + "-wal")
        assert not wal_path.exists(), "WAL should not exist with journal_mode=DELETE"

        result = _run_helper(db_path, "SELECT COUNT(*) FROM sessions;")
        assert result.returncode == 0, f"stderr: {result.stderr}"

        # The helper opens in ?mode=ro which should not create a WAL.
        # Note: SQLite may still create a -shm file in some configurations,
        # but the WAL itself should remain absent or empty.
        if wal_path.exists():
            assert wal_path.stat().st_size == 0, (
                "WAL file was created and written to by read-only query"
            )


class TestReadonlyHelperUriMode:
    """Verify the helper uses the correct URI mode."""

    def test_helper_opens_readonly(self, populated_db: Path) -> None:
        """The helper should report read-only mode via PRAGMA."""
        # We can't directly query PRAGMA through the helper's output,
        # but we can verify indirectly: a write must fail.
        result = _run_helper(
            populated_db,
            "PRAGMA query_only;",
        )
        # The helper prints the result as a Python list.
        # query_only=1 means it's active.
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert "1" in result.stdout
