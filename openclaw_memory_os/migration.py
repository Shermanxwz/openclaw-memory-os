"""Centralized schema migration runner for the audit + feedback SQLite DBs.

This module is the single, authoritative entry point for evolving the
SQLite schema used by ``recall_feedback.py`` (and any future stores
that share the same file). It replaces the previous pattern of
inline ``CREATE TABLE IF NOT EXISTS`` + ``ALTER TABLE ... ADD COLUMN``
blocks scattered across feature modules, and gives the project:

* a single ``SCHEMA_VERSION`` constant that every store declares,
* a ``metadata`` table that records the currently-applied version,
* a ``migration_history`` audit log of every (from_version →
  to_version, applied_at, success) step,
* per-migration transactions (each migration runs in its own
  ``BEGIN`` / ``COMMIT``; failures roll back so the DB never lands
  in a half-applied state),
* a guard that refuses to write to a DB whose ``schema_version``
  is **higher** than this code's ``SCHEMA_VERSION`` (an operator
  rolling back to an older build must either keep the new build or
  run a downgrade migration explicitly).

Design notes
------------

* **No dependency on a specific DB file.** ``run_migrations(conn)``
  operates on whatever ``sqlite3.Connection`` is handed in. The
  caller chooses the file; the migration runner only enforces the
  schema-version contract.
* **Idempotent.** Running ``run_migrations(conn)`` twice on the
  same DB is a no-op the second time — the runner detects that
  ``current_version == SCHEMA_VERSION`` and returns early.
* **Atomic.** Each migration is wrapped in ``BEGIN`` … ``COMMIT``.
  On any exception inside a migration the transaction is rolled
  back, ``migration_history`` records ``success=0`` for that step
  and the original ``schema_version`` value is preserved so the
  next call can retry from a clean state.
* **Forward-only.** Migrations advance ``current_version`` by 1
  per step. Downgrades are not part of the v0.3.0 contract; a
  higher-than-known ``schema_version`` is treated as a hard error
  so we never silently write a row into an unknown schema.

Migration list
--------------

The canonical migration list lives in ``_MIGRATIONS`` as a sequence
of small callables. Each callable receives the active connection
and the (from_version, to_version) it is expected to advance the
schema across. New migrations are appended to ``_MIGRATIONS``;
never edit a migration that has shipped.
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from typing import Callable, List, Tuple

logger = logging.getLogger(__name__)


#: The current target schema version. Every store backed by the
#: v0.3.0 migration runner must declare this same constant as its
#: "I understand the schema up to this version" marker.
SCHEMA_VERSION: int = 3


#: Stable name for the legacy feedback data-migration step (G8).
#: This is a *data* migration (not a schema migration) so it does
#: not appear in :data:`_MIGRATIONS`; it is registered separately
#: so other modules can refer to it without importing
#: ``recall_feedback`` directly. The migration itself is invoked
#: from ``recall_feedback._get_db()`` (which is the single startup
#: bootstrap path for the recall-feedback DB) and is idempotent
#: via a marker row in the ``metadata`` table.
LEGACY_FEEDBACK_MIGRATION_NAME: str = "migrate_legacy_feedback"


#: Name of the metadata table that records the currently-applied
#: schema version. The ``metadata`` table is keyed by a textual
#: ``key`` so it can host additional operational flags in the
#: future (e.g. ``wal_enabled``, ``last_audit_ts``).
METADATA_TABLE: str = "metadata"

#: Name of the audit log table that records every migration step
#: applied by :func:`run_migrations`. Rows are append-only; the
#: runner never deletes from this table.
MIGRATION_HISTORY_TABLE: str = "migration_history"


# ---------------------------------------------------------------------------
# Migration primitives
# ---------------------------------------------------------------------------


def _ensure_metadata_table(conn: sqlite3.Connection) -> None:
    """Create the ``metadata`` table if it does not yet exist."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {METADATA_TABLE} (
            key   TEXT PRIMARY KEY,
            value TEXT
        )
        """
    )


def _ensure_migration_history_table(conn: sqlite3.Connection) -> None:
    """Create the ``migration_history`` table if it does not yet exist."""
    conn.execute(
        f"""
        CREATE TABLE IF NOT EXISTS {MIGRATION_HISTORY_TABLE} (
            id            INTEGER PRIMARY KEY AUTOINCREMENT,
            from_version  INTEGER NOT NULL,
            to_version    INTEGER NOT NULL,
            applied_at    TEXT NOT NULL,
            success       INTEGER NOT NULL,
            detail        TEXT
        )
        """
    )


def _read_current_version(conn: sqlite3.Connection) -> int:
    """Return the schema version currently recorded in ``metadata``.

    Returns ``0`` when no ``schema_version`` row exists yet (fresh
    DB). The runner uses ``0`` as "pre-v1" so the first migration
    always applies on a brand-new file.
    """
    _ensure_metadata_table(conn)
    row = conn.execute(
        f"SELECT value FROM {METADATA_TABLE} WHERE key = ?",
        ("schema_version",),
    ).fetchone()
    if row is None:
        return 0
    try:
        return int(row[0])
    except (TypeError, ValueError):
        # Corrupted / unknown value — treat as pre-v1 so the runner
        # can attempt to repair by re-running migrations. This is
        # logged so operators notice the corruption.
        logger.warning(
            "migration: metadata.schema_version is non-integer (%r); treating as 0",
            row[0],
        )
        return 0


def _write_current_version(conn: sqlite3.Connection, version: int) -> None:
    """Upsert ``metadata.schema_version`` to ``version``."""
    _ensure_metadata_table(conn)
    conn.execute(
        f"""
        INSERT INTO {METADATA_TABLE} (key, value) VALUES (?, ?)
        ON CONFLICT(key) DO UPDATE SET value = excluded.value
        """,
        ("schema_version", str(int(version))),
    )


def _record_history(
    conn: sqlite3.Connection,
    *,
    from_version: int,
    to_version: int,
    success: bool,
    detail: str = "",
) -> None:
    """Append a row to ``migration_history``.

    The runner records both successful AND failed migration steps
    so the audit log captures every attempted bump.
    """
    conn.execute(
        f"""
        INSERT INTO {MIGRATION_HISTORY_TABLE}
            (from_version, to_version, applied_at, success, detail)
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            int(from_version),
            int(to_version),
            datetime.now(timezone.utc).isoformat(),
            1 if success else 0,
            detail,
        ),
    )


# ---------------------------------------------------------------------------
# Migration definitions
# ---------------------------------------------------------------------------


def _migration_v1_to_v2(conn: sqlite3.Connection) -> None:
    """No-op migration: the v0.3.0 baseline schema is already at v2.

    The ``recall_runs`` / ``recall_results`` / ``feedback_events``
    tables were created by the v0.3.0 launch with all the
    v0.3.0.x extension columns inline, and ``recall_feedback.py``
    has been carrying its own ``_SCHEMA_COLUMNS`` list for the
    ALTER-TABLE path on pre-existing databases. As of the G8
    graduation milestone we declare that baseline schema as
    ``SCHEMA_VERSION = 3``. This migration exists so that a fresh
    DB transitions cleanly from ``0`` to ``2`` and so that older
    databases carrying only ``v1`` (no ``metadata`` table yet)
    receive a no-op bump plus the bookkeeping tables that every
    future migration will rely on.
    """
    # Intentionally empty: the baseline schema is already at v2.
    # The bookkeeping tables are created by run_migrations() itself
    # so this migration only exists to bump the version counter.
    return None




def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    return conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone() is not None


def _table_columns(conn: sqlite3.Connection, table: str) -> set[str]:
    return {str(row[1]) for row in conn.execute(f"PRAGMA table_info({table})")}


def _migration_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Detach durable feedback from expiring recall traces.

    v2 tied ``feedback_events.query_id`` to the shorter-lived recall trace.
    v3 rebuilds that child table without the foreign key and records explicit
    collection-resolution provenance.  Fresh databases have no feature tables
    yet, so their current schema is created by ``recall_feedback`` afterwards.
    """
    if not _table_exists(conn, "feedback_events"):
        return

    for column, sql_type in (
        ("collection", "TEXT"),
        ("migration_status", "TEXT"),
        ("resolution_status", "TEXT"),
        ("legacy_source_key", "TEXT"),
    ):
        if column not in _table_columns(conn, "feedback_events"):
            conn.execute(
                f"ALTER TABLE feedback_events ADD COLUMN {column} {sql_type}"
            )

    fk_rows = conn.execute("PRAGMA foreign_key_list(feedback_events)").fetchall()
    has_recall_fk = any(
        str(row[2]) == "recall_runs" and str(row[3]) == "query_id"
        for row in fk_rows
    )
    if has_recall_fk:
        conn.execute("DROP TABLE IF EXISTS feedback_events_v3")
        conn.execute(
            """CREATE TABLE feedback_events_v3 (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                candidate_key TEXT NOT NULL,
                memory_id TEXT,
                collection TEXT,
                useful INTEGER NOT NULL,
                created_at TEXT NOT NULL DEFAULT (datetime('now')),
                feedback_source TEXT DEFAULT 'dashboard',
                migration_status TEXT,
                resolution_status TEXT,
                legacy_source_key TEXT
            )"""
        )
        ordered = [
            "id", "query_id", "candidate_key", "memory_id", "collection",
            "useful", "created_at", "feedback_source", "migration_status",
            "resolution_status", "legacy_source_key",
        ]
        existing = _table_columns(conn, "feedback_events")
        common = [column for column in ordered if column in existing]
        if common:
            columns = ", ".join(common)
            conn.execute(
                f"INSERT INTO feedback_events_v3 ({columns}) "
                f"SELECT {columns} FROM feedback_events"
            )
        conn.execute("DROP TABLE feedback_events")
        conn.execute("ALTER TABLE feedback_events_v3 RENAME TO feedback_events")

    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_events_query "
        "ON feedback_events(query_id)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_feedback_events_candidate "
        "ON feedback_events(candidate_key)"
    )
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_events_legacy_source "
        "ON feedback_events(legacy_source_key) "
        "WHERE legacy_source_key IS NOT NULL"
    )


#: Ordered list of migrations. Each entry is
#: ``(from_version, to_version, callable)`` and the runner applies
#: them in sequence, skipping any whose ``from_version`` is below
#: the DB's current version. Never edit a shipped migration; add a
#: new one instead.
_MIGRATIONS: List[Tuple[int, int, Callable[[sqlite3.Connection], None]]] = [
    (0, 1, _migration_v1_to_v2),
    (1, 2, _migration_v1_to_v2),
    (2, 3, _migration_v2_to_v3),
]


def _apply_one(
    conn: sqlite3.Connection,
    from_version: int,
    to_version: int,
    body: Callable[[sqlite3.Connection], None],
) -> None:
    """Run a single migration inside its own transaction.

    On success the transaction commits and ``migration_history``
    receives a ``success=1`` row. On any exception the transaction
    is rolled back and ``migration_history`` records ``success=0``
    with the error message. The original ``schema_version`` row is
    preserved on rollback so the next call can retry.
    """
    try:
        conn.execute("BEGIN")
        body(conn)
        _write_current_version(conn, to_version)
        _record_history(
            conn,
            from_version=from_version,
            to_version=to_version,
            success=True,
        )
        conn.execute("COMMIT")
    except Exception as exc:  # noqa: BLE001 - we re-raise after logging
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:  # pragma: no cover - defensive
            pass
        # Record the failure in its own transaction so the audit
        # log survives even when the migration body rolled back.
        try:
            conn.execute("BEGIN")
            _record_history(
                conn,
                from_version=from_version,
                to_version=to_version,
                success=False,
                detail=f"{type(exc).__name__}: {exc}",
            )
            conn.execute("COMMIT")
        except sqlite3.Error as log_exc:  # pragma: no cover - defensive
            logger.error("migration: failed to record history: %s", log_exc)
        logger.error(
            "migration: %s→%s failed: %s", from_version, to_version, exc
        )
        raise


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class FutureSchemaError(RuntimeError):
    """Raised when a DB's recorded schema_version exceeds ``SCHEMA_VERSION``.

    This is a hard error: writing rows into an unknown schema
    risks corrupting the data model. Operators must either keep
    the newer build or run an explicit downgrade migration (out
    of scope for v0.3.0).
    """


def run_migrations(conn: sqlite3.Connection) -> int:
    """Apply pending migrations to ``conn``.

    The runner is **idempotent** and **atomic per step**:

    1. Creates the ``metadata`` table if missing.
    2. Creates the ``migration_history`` table if missing.
    3. Reads the DB's currently-applied ``schema_version``.
    4. If the recorded version is **higher** than
       :data:`SCHEMA_VERSION`, raises :class:`FutureSchemaError`
       and refuses to write.
    5. Applies every migration in :data:`_MIGRATIONS` whose
       ``from_version`` is ``>= current_version``, in order.
    6. Each migration runs in its own transaction; failures roll
       back and are recorded with ``success=0``.

    Parameters
    ----------
    conn:
        An open ``sqlite3.Connection``. The caller owns the
        connection's lifecycle (open / close / commit policy);
        the runner never closes it. The runner does *not* issue
        a final ``COMMIT`` — it commits per-step inside the
        per-step transactions — so the caller's commit policy
        (e.g. autocommit-off) is preserved.

    Returns
    -------
    int
        The schema version **after** the call. Equals
        :data:`SCHEMA_VERSION` on a fully-migrated DB; equals
        the recorded version on a no-op call (already at target).
    """
    _ensure_metadata_table(conn)
    _ensure_migration_history_table(conn)
    conn.commit()

    current = _read_current_version(conn)

    if current > SCHEMA_VERSION:
        raise FutureSchemaError(
            f"DB schema_version={current} is newer than "
            f"runtime SCHEMA_VERSION={SCHEMA_VERSION}. "
            f"Refusing to write to a forward-only schema. "
            f"Roll forward to a newer build or run a downgrade migration."
        )

    if current == SCHEMA_VERSION:
        # Already at target — idempotent no-op. The caller can rely
        # on the return value to short-circuit downstream
        # CREATE TABLE / ALTER TABLE work safely.
        logger.debug("migration: schema already at v%d", current)
        return current

    applied = current
    for from_version, to_version, body in _MIGRATIONS:
        if from_version < applied:
            continue  # already past this migration
        _apply_one(conn, from_version, to_version, body)
        applied = to_version

    # If migrations were sparse (no entry in _MIGRATIONS for the
    # current version) we still bump straight to SCHEMA_VERSION so
    # the DB carries the canonical target. The bookkeeping tables
    # are guaranteed to exist by the time we get here.
    if applied < SCHEMA_VERSION:
        try:
            conn.execute("BEGIN")
            _write_current_version(conn, SCHEMA_VERSION)
            _record_history(
                conn,
                from_version=applied,
                to_version=SCHEMA_VERSION,
                success=True,
                detail="implicit bump to current SCHEMA_VERSION",
            )
            conn.execute("COMMIT")
            applied = SCHEMA_VERSION
        except sqlite3.Error as exc:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:  # pragma: no cover - defensive
                pass
            logger.error("migration: implicit bump failed: %s", exc)
            raise

    return applied


# ---------------------------------------------------------------------------
# WAL enforcement helper (used by ``recall_feedback.py``)
# ---------------------------------------------------------------------------


def enable_wal(conn: sqlite3.Connection) -> bool:
    """Switch the connection's journal mode to WAL.

    Returns ``True`` when WAL was successfully enabled, ``False``
    otherwise (e.g. on read-only filesystems or in-memory DBs).
    This helper is intentionally side-effect-light so callers can
    invoke it on every fresh connection without paying the cost
    of re-running the PRAGMA when the mode is already set.
    """
    try:
        cur = conn.execute("PRAGMA journal_mode=WAL")
    except sqlite3.Error as exc:
        logger.debug("migration: PRAGMA journal_mode=WAL failed: %s", exc)
        return False
    row = cur.fetchone()
    if row is None:
        return False
    mode = str(row[0] or "").lower()
    return mode == "wal"


__all__ = [
    "SCHEMA_VERSION",
    "METADATA_TABLE",
    "MIGRATION_HISTORY_TABLE",
    "FutureSchemaError",
    "run_migrations",
    "enable_wal",
    # G8 — legacy feedback migration. Lives in ``recall_feedback``
    # because the audit log + feedback_events tables are both owned
    # by that module. Exposed here as a stable import path so other
    # modules (CLI scripts, FastAPI lifespan) can call it without
    # taking a hard dependency on ``recall_feedback``.
    "LEGACY_FEEDBACK_MIGRATION_NAME",
]