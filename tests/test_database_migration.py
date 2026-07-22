"""Tests for the centralised schema migration runner (G8).

The runner is the single authoritative entry point for evolving the
recall_feedback SQLite schema. The tests below cover the four
invariants the runbook cares about:

* ``run_migrations`` is **idempotent**: running it twice on a
  fresh DB is a no-op the second time (and leaves only one
  history row per actual step).
* ``migration_history`` records every (from_version, to_version,
  applied_at, success) step the runner applies.
* ``run_migrations`` refuses to write to a DB whose recorded
  ``schema_version`` exceeds the runtime ``SCHEMA_VERSION``
  (rolling back to an older build must not silently corrupt
  data).
* Each migration step is **atomic**: a failing migration
  rolls back so the DB never lands in a half-applied state,
  and the failed step is recorded with ``success=0``.
"""

from __future__ import annotations

import sqlite3

import pytest

from openclaw_memory_os.migration import (
    SCHEMA_VERSION,
    FutureSchemaError,
    run_migrations,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _fresh_conn() -> sqlite3.Connection:
    """Return a brand-new in-memory SQLite connection."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    return conn


def _read_metadata_version(conn: sqlite3.Connection) -> int:
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?", ("schema_version",)
    ).fetchone()
    return int(row[0]) if row is not None else 0


def _history_count(conn: sqlite3.Connection) -> int:
    row = conn.execute("SELECT COUNT(*) FROM migration_history").fetchone()
    return int(row[0])


# ---------------------------------------------------------------------------
# 1. Idempotency: second call is a no-op.
# ---------------------------------------------------------------------------


def test_run_migrations_is_idempotent():
    """Calling ``run_migrations`` twice on a fresh DB leaves exactly
    one history row per applied step — the second call short-
    circuits on ``current == SCHEMA_VERSION``."""
    conn = _fresh_conn()

    v_first = run_migrations(conn)
    assert v_first == SCHEMA_VERSION
    history_after_first = _history_count(conn)
    # We must have applied *something* — the migration_history
    # bookkeeping is what the audit guarantees.
    assert history_after_first >= 1

    v_second = run_migrations(conn)
    assert v_second == SCHEMA_VERSION
    # Idempotent: the second call did not append a new history row.
    assert _history_count(conn) == history_after_first


def test_run_migrations_returns_schema_version():
    """The runner returns the schema version AFTER applying pending
    migrations (== SCHEMA_VERSION on a fresh DB)."""
    conn = _fresh_conn()
    assert run_migrations(conn) == SCHEMA_VERSION
    assert _read_metadata_version(conn) == SCHEMA_VERSION


# ---------------------------------------------------------------------------
# 2. migration_history records every step.
# ---------------------------------------------------------------------------


def test_migration_history_recorded():
    """After ``run_migrations``, ``migration_history`` carries at
    least one entry per applied ``from_version`` → ``to_version``
    step, each with ``success=1``."""
    conn = _fresh_conn()
    run_migrations(conn)

    rows = conn.execute(
        "SELECT from_version, to_version, success FROM migration_history ORDER BY id"
    ).fetchall()
    assert rows, "migration_history must contain at least one row"

    # Every applied step must end at SCHEMA_VERSION (or earlier for
    # intermediate steps). The migration chain is monotonic in the
    # ``to_version`` column.
    to_versions = [int(r["to_version"]) for r in rows]
    assert to_versions == sorted(to_versions)
    assert to_versions[-1] == SCHEMA_VERSION

    # All recorded rows are successful (no failure paths triggered).
    for r in rows:
        assert int(r["success"]) == 1


def test_migration_history_includes_applied_at():
    """Each row carries a non-empty ``applied_at`` timestamp so the
    audit log can reconstruct when the schema evolved."""
    conn = _fresh_conn()
    run_migrations(conn)
    rows = conn.execute(
        "SELECT applied_at FROM migration_history"
    ).fetchall()
    assert rows
    for r in rows:
        assert r["applied_at"], "applied_at must be a non-empty ISO timestamp"


# ---------------------------------------------------------------------------
# 3. Refuse to write when schema_version > SCHEMA_VERSION.
# ---------------------------------------------------------------------------


def test_reject_future_schema():
    """A DB whose recorded ``schema_version`` is higher than the
    runtime ``SCHEMA_VERSION`` is rejected with
    :class:`FutureSchemaError`. Writing into an unknown schema
    risks data corruption, so the runner refuses outright."""
    conn = _fresh_conn()
    # Lay down the bookkeeping tables so we can simulate a future
    # version scenario without tripping over the runner's table-
    # creation contract.
    conn.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "CREATE TABLE migration_history ("
        "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
        "  from_version INTEGER NOT NULL,"
        "  to_version INTEGER NOT NULL,"
        "  applied_at TEXT NOT NULL,"
        "  success INTEGER NOT NULL,"
        "  detail TEXT)"
    )
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION + 50)),
    )
    conn.commit()

    with pytest.raises(FutureSchemaError):
        run_migrations(conn)


def test_reject_future_schema_does_not_mutate_db():
    """When the runner refuses to write, the DB's ``schema_version``
    row is preserved verbatim — no silent overwrite / downgrade."""
    conn = _fresh_conn()
    conn.execute(
        "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)"
    )
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?)",
        ("schema_version", str(SCHEMA_VERSION + 10)),
    )
    conn.commit()
    with pytest.raises(FutureSchemaError):
        run_migrations(conn)
    assert _read_metadata_version(conn) == SCHEMA_VERSION + 10


# ---------------------------------------------------------------------------
# 4. Atomic migration: failure rolls back to the prior version.
# ---------------------------------------------------------------------------


def test_atomic_migration():
    """A failing migration rolls back the DB to its prior
    ``schema_version`` and the failed step is recorded with
    ``success=0``. This is the central safety guarantee the runbook
    requires: a half-applied migration must never be persisted."""
    from openclaw_memory_os import migration as mig

    original_migrations = list(mig._MIGRATIONS)
    target_version = SCHEMA_VERSION + 1
    recorded_version = SCHEMA_VERSION  # strictly less than target → runner will try to apply

    def _boom(conn):
        # Reference an undefined table to force a sqlite3.OperationalError
        # inside the per-step transaction. This is enough to exercise
        # the rollback path without depending on production schema.
        conn.execute("SELECT * FROM table_that_does_not_exist_xyz")

    # Inject a synthetic failing migration that targets a higher
    # version than the runner knows about. We temporarily raise
    # ``SCHEMA_VERSION`` so the runner considers the step pending
    # (``current == recorded_version < SCHEMA_VERSION``). The runner
    # then attempts to apply the failing step, which is exactly the
    # scenario we want to exercise.
    mig.SCHEMA_VERSION = target_version
    mig._MIGRATIONS = [
        (recorded_version, target_version, _boom),
    ]
    try:
        conn = _fresh_conn()
        # Lay down the bookkeeping tables first (the runner normally
        # creates them but we need to call it manually for this
        # adversarial scenario).
        conn.execute(
            "CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT)"
        )
        conn.execute(
            "CREATE TABLE migration_history ("
            "  id INTEGER PRIMARY KEY AUTOINCREMENT,"
            "  from_version INTEGER NOT NULL,"
            "  to_version INTEGER NOT NULL,"
            "  applied_at TEXT NOT NULL,"
            "  success INTEGER NOT NULL,"
            "  detail TEXT)"
        )
        conn.execute(
            "INSERT INTO metadata(key, value) VALUES (?, ?)",
            ("schema_version", str(recorded_version)),
        )
        conn.commit()

        with pytest.raises(sqlite3.OperationalError):
            run_migrations(conn)

        # The DB's recorded version did NOT advance — atomic rollback.
        assert _read_metadata_version(conn) == recorded_version

        # And the failed step is captured in migration_history.
        rows = conn.execute(
            "SELECT from_version, to_version, success, detail FROM migration_history"
        ).fetchall()
        assert rows, "failure must be recorded in migration_history"
        # At least one row must carry success=0 with the error message.
        failures = [r for r in rows if int(r["success"]) == 0]
        assert failures, "expected at least one failure row"
        assert failures[0]["detail"], "failure row must carry a non-empty detail"
    finally:
        mig.SCHEMA_VERSION = SCHEMA_VERSION
        mig._MIGRATIONS = original_migrations


# ---------------------------------------------------------------------------
# 5. Sanity: the runner leaves the feature tables alone.
# ---------------------------------------------------------------------------


def test_run_migrations_creates_only_bookkeeping_tables():
    """``run_migrations`` only creates the bookkeeping tables
    (``metadata``, ``migration_history``); it must not touch feature
    tables (``recall_runs`` etc.) — those are owned by
    ``recall_feedback._ensure_schema``."""
    conn = _fresh_conn()
    run_migrations(conn)
    tables = {
        r[0]
        for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    assert {"metadata", "migration_history"} <= tables
    # Feature tables must NOT be created by the migration runner —
    # they belong to the feature module's ``_ensure_schema`` step.
    assert "recall_runs" not in tables
    assert "feedback_events" not in tables
    assert "recall_results" not in tables