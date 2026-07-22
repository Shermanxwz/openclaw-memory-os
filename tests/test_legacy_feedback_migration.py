"""Tests for the Runbook G8 legacy feedback migration.

The pre-v0.3.0 audit log stored feedback events as ``action='feedback'``
rows in the ``audit_log`` table, with the actual query / memory_id /
useful payload encoded inside the ``detail`` column as a Python
``repr()``-style string. v0.3.0 normalises that signal into the
``feedback_events`` table so the offline evaluation pipeline can
replay real recall traces.

Coverage:

* ``migrate_legacy_feedback`` is idempotent: a second call on the
  same DB returns ``0`` and writes no new rows.
* A DB without ``audit_log`` is a clean no-op + marker set so the
  call site can run the migration at startup without crashing.
* Seeded ``audit_log`` rows are migrated into ``feedback_events``
  with ``migration_status = "migrated:audit"``.
* The migration fires on the first ``_get_db()`` call (the bootstrap
  path).
* Migrated rows carry every required field (query_id, candidate_key,
  useful, created_at).
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

import openclaw_memory_os.recall_feedback as rf
from openclaw_memory_os.recall_feedback import (
    _LEGACY_MIGRATION_MARKER,
    _get_db,
    migrate_legacy_feedback,
    record_feedback_v030,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_recall_db(tmp_path: Path, monkeypatch):
    """Point recall_feedback at a tmpdir so tests don't touch the real DB."""
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path))
    rf._RECALL_DB_DIR = tmp_path / "openclaw-memory-os"
    rf._RECALL_DB = rf._RECALL_DB_DIR / "recall_feedback.db"
    yield
    if rf._RECALL_DB.exists():
        rf._RECALL_DB.unlink()


@pytest.fixture
def audit_db_with_feedback(tmp_path: Path, monkeypatch):
    """Seed a fresh audit_log SQLite DB with N feedback rows.

    Mirrors the schema produced by ``openclaw_memory_os.audit``
    so the migration's read path matches what the production
    code emits. The ``detail`` payload follows the v0.2.x format
    (``query='...' memory_id='...' useful=True|False``) which the
    ``_parse_old_feedback`` regex expects.
    """
    audit_path = tmp_path / "audit_log.sqlite"
    conn = sqlite3.connect(str(audit_path))
    conn.executescript(
        """
        CREATE TABLE audit_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action    TEXT NOT NULL,
            actor     TEXT,
            memory_id TEXT,
            detail    TEXT
        );
        """
    )
    # Seed 5 feedback rows. The first column is the candidate_key
    # (memory_id) and the third is the useful flag; we mirror the
    # v0.2.x emit format ``query='...' memory_id='...' useful=True``.
    rows = [
        ("how to rotate token", "m1", True),
        ("rotate token steps", "m2", False),
        ("deploy checklist", "m3", True),
        ("deploy checklist", "m4", True),
        ("deploy checklist", "m5", False),
    ]
    for idx, (q, mid, useful) in enumerate(rows):
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, actor, memory_id, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                f"2026-07-0{idx + 1}T08:00:00+00:00",
                "feedback",
                "dashboard",
                mid,
                f"query='{q}' memory_id='{mid}' useful={useful}",
            ),
        )
    conn.commit()
    conn.close()

    # Redirect the audit store singleton to point at our tmpfile
    # so ``_open_audit_db_for_read`` opens the seeded DB rather
    # than the real ``/root/.local/share/...`` audit_log.sqlite.
    from openclaw_memory_os import audit as audit_module
    # Clear any cached default store so the new path is honoured.
    audit_module._default_store = None
    fake_store = audit_module.AuditStore(db_path=audit_path)
    audit_module._default_store = fake_store
    # Also ensure the new store uses the tmpdir (some AuditStore
    # code paths derive the default path lazily from env).
    monkeypatch.setenv("OPENCLAW_MEMORY_OS_AUDIT_PATH", str(audit_path))

    yield {
        "path": audit_path,
        "rows": rows,
    }

    # Cleanup: reset the singleton so other tests can build their
    # own AuditStore with default config.
    audit_module._default_store = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_migrate_legacy_feedback_is_idempotent(audit_db_with_feedback):
    """Calling migrate twice returns 0 the second time.

    The first call migrates the 5 seeded rows; the second call
    must short-circuit on the metadata marker and return 0.
    """
    first = migrate_legacy_feedback()
    assert first == 5, f"expected 5 rows migrated, got {first}"

    # Inspect the marker row.
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_LEGACY_MIGRATION_MARKER,),
        ).fetchone()
        assert row is not None, "marker row missing after migration"
    finally:
        conn.close()

    # Second call: marker is set, so the migration returns 0 and
    # leaves the row count unchanged.
    second = migrate_legacy_feedback()
    assert second == 0

    conn = _get_db()
    try:
        total = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
        assert total == 5, f"expected 5 rows in feedback_events, got {total}"
    finally:
        conn.close()


def test_migrate_legacy_feedback_handles_missing_audit_log(tmp_path, monkeypatch):
    """When the audit_log table doesn't exist, migration is a no-op + marker.

    The brief is explicit: a fresh install must be able to call
    ``migrate_legacy_feedback`` from ``_get_db()`` without
    crashing, even when no audit log has ever been written.
    """
    # Point the audit store at a path that doesn't exist yet.
    monkeypatch.setenv("OPENCLAW_MEMORY_OS_AUDIT_PATH", str(tmp_path / "nonexistent.sqlite"))
    from openclaw_memory_os import audit as audit_module
    audit_module._default_store = None

    result = migrate_legacy_feedback()
    assert result == 0

    # Marker must be set so a second call short-circuits cheaply.
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_LEGACY_MIGRATION_MARKER,),
        ).fetchone()
        assert row is not None, "marker not set after no-op migration"
        # Marker value is an ISO timestamp.
        assert row["value"], "marker value should not be empty"
    finally:
        conn.close()


def test_migrate_legacy_feedback_writes_to_feedback_events(audit_db_with_feedback):
    """5 seeded audit rows → 5 feedback_events rows with migration_status."""
    migrated = migrate_legacy_feedback()
    assert migrated == 5

    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT query_id, candidate_key, useful, migration_status, "
            "feedback_source, created_at FROM feedback_events "
            "ORDER BY created_at ASC"
        ).fetchall()
        assert len(rows) == 5
        # Every row must carry the migration status marker so the
        # downstream evaluator can filter them out.
        for row in rows:
            assert row["migration_status"] == "migrated:audit"
            assert row["feedback_source"] == "legacy_audit"
            assert row["created_at"], "created_at missing"
            assert row["query_id"], "query_id missing"
            assert row["candidate_key"], "candidate_key missing"
        # The 3 negative / positive split mirrors the seeded data.
        positives = sum(1 for r in rows if r["useful"] == 1)
        negatives = sum(1 for r in rows if r["useful"] == 0)
        assert positives == 3
        assert negatives == 2
    finally:
        conn.close()


def test_legacy_migration_runs_on_first_db_connect(audit_db_with_feedback):
    """The first ``_get_db()`` call after seeding the audit_log
    must run the migration and land 5 rows in feedback_events.

    Pinning the "migration fires at startup" behaviour is the
    primary Runbook G8 acceptance criterion: a fresh process that
    boots against a pre-existing audit log must see the migrated
    rows the very first time it opens the recall DB.
    """
    # Reset the recall DB state so ``_get_db()`` is a fresh open.
    rf._RECALL_DB.unlink(missing_ok=True)

    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT COUNT(*) AS c, "
            "SUM(CASE WHEN migration_status='migrated:audit' THEN 1 ELSE 0 END) AS migrated "
            "FROM feedback_events"
        ).fetchone()
        assert rows["c"] == 5, f"expected 5 migrated rows, got {rows['c']}"
        assert rows["migrated"] == 5
        # The marker must be set as part of the same bootstrap call.
        marker = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_LEGACY_MIGRATION_MARKER,),
        ).fetchone()
        assert marker is not None
    finally:
        conn.close()


def test_migrated_rows_have_correct_schema(audit_db_with_feedback):
    """Migrated rows must carry every required field the live
    ``record_feedback_v030`` writes.

    This catches schema-drift bugs where the migration lands rows
    with fewer fields than the live path, which would make the
    JOIN with recall_runs fail.
    """
    migrate_legacy_feedback()

    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM feedback_events WHERE migration_status='migrated:audit'"
        ).fetchall()
        assert len(rows) >= 1
        # Inspect the schema of feedback_events so the test
        # catches any future column removal that would break the
        # migration.
        cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback_events)").fetchall()}
        for required in (
            "id",
            "query_id",
            "candidate_key",
            "memory_id",
            "useful",
            "created_at",
            "feedback_source",
            "migration_status",
        ):
            assert required in cols, f"feedback_events missing column: {required}"

        # Each migrated row must carry the canonical field set.
        for row in rows:
            assert row["query_id"].startswith("legacy_audit:"), (
                f"unexpected query_id format: {row['query_id']!r}"
            )
            assert row["candidate_key"], "candidate_key missing"
            assert row["memory_id"], "memory_id missing"
            assert row["useful"] in (0, 1)
            assert row["created_at"]
    finally:
        conn.close()


def test_migration_skips_malformed_legacy_rows(tmp_path, monkeypatch):
    """Rows with malformed ``detail`` (no memory_id) must be skipped,
    not crashed on, and the migration must still set the marker
    so subsequent calls short-circuit.
    """
    audit_path = tmp_path / "audit_partial.sqlite"
    conn = sqlite3.connect(str(audit_path))
    conn.executescript(
        """
        CREATE TABLE audit_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action    TEXT NOT NULL,
            actor     TEXT,
            memory_id TEXT,
            detail    TEXT
        );
        """
    )
    # 1 good row + 1 bad row (no memory_id in detail).
    conn.execute(
        "INSERT INTO audit_log (timestamp, action, detail) VALUES (?, ?, ?)",
        ("2026-07-01T08:00:00+00:00", "feedback", "query='ok' memory_id='m-good' useful=True"),
    )
    conn.execute(
        "INSERT INTO audit_log (timestamp, action, detail) VALUES (?, ?, ?)",
        ("2026-07-01T08:01:00+00:00", "feedback", "garbage no parseable fields"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("OPENCLAW_MEMORY_OS_AUDIT_PATH", str(audit_path))
    from openclaw_memory_os import audit as audit_module
    audit_module._default_store = None

    migrated = migrate_legacy_feedback()
    # Only the well-formed row is migrated; the bad row is skipped.
    assert migrated == 1

    # Marker is still set so the next call is a no-op.
    conn = _get_db()
    try:
        marker = conn.execute(
            "SELECT value FROM metadata WHERE key = ?",
            (_LEGACY_MIGRATION_MARKER,),
        ).fetchone()
        assert marker is not None
        total = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
        assert total == 1
    finally:
        conn.close()


def test_migration_handles_recall_feedback_action_too(tmp_path, monkeypatch):
    """Very old builds used ``action='recall_feedback'`` instead of
    ``'feedback'``. Both must be migrated.
    """
    audit_path = tmp_path / "audit_old.sqlite"
    conn = sqlite3.connect(str(audit_path))
    conn.executescript(
        """
        CREATE TABLE audit_log (
            id        INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action    TEXT NOT NULL,
            actor     TEXT,
            memory_id TEXT,
            detail    TEXT
        );
        """
    )
    conn.execute(
        "INSERT INTO audit_log (timestamp, action, detail) VALUES (?, ?, ?)",
        ("2026-06-15T08:00:00+00:00", "feedback", "query='q1' memory_id='m-1' useful=True"),
    )
    conn.execute(
        "INSERT INTO audit_log (timestamp, action, detail) VALUES (?, ?, ?)",
        ("2026-06-15T08:01:00+00:00", "recall_feedback", "query='q2' memory_id='m-2' useful=False"),
    )
    conn.commit()
    conn.close()

    monkeypatch.setenv("OPENCLAW_MEMORY_OS_AUDIT_PATH", str(audit_path))
    from openclaw_memory_os import audit as audit_module
    audit_module._default_store = None

    migrated = migrate_legacy_feedback()
    assert migrated == 2

    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT memory_id, useful FROM feedback_events "
            "WHERE migration_status='migrated:audit' ORDER BY memory_id ASC"
        ).fetchall()
        assert {r["memory_id"] for r in rows} == {"m-1", "m-2"}
        by_id = {r["memory_id"]: r["useful"] for r in rows}
        assert by_id["m-1"] == 1
        assert by_id["m-2"] == 0
    finally:
        conn.close()


def test_migration_does_not_block_record_feedback_v030(audit_db_with_feedback):
    """After the migration runs, new feedback can still be recorded
    via ``record_feedback_v030`` (the live path). The migration
    must not leave the DB in a state that breaks subsequent writes.
    """
    migrate_legacy_feedback()
    # Pre-seed a recall_results row so the G5.1 strong-validation
    # check passes.
    rf.record_recall_run("live-q-1", "live query")
    rf.record_recall_result("live-q-1", "live:mem-1", rank=1, status="active")
    new_id = record_feedback_v030("live-q-1", "live:mem-1", True)
    assert new_id > 0

    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT query_id, migration_status FROM feedback_events ORDER BY id"
        ).fetchall()
        # The migrated rows + the live row.
        statuses = [r["migration_status"] for r in rows]
        assert "migrated:audit" in statuses
        assert None in statuses  # live row has NULL migration_status
    finally:
        conn.close()