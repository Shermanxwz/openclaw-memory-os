"""Persistent SQLite-backed session store for browser session cookies.

The dashboard historically kept active sessions in a module-level
``dict`` inside :mod:`openclaw_memory_os.auth`, which meant that any
service restart would silently invalidate every browser tab. This
module replaces that in-memory state with a small SQLite store so that
sessions issued before a restart remain valid until either their
``max_age`` elapses or the operator explicitly revokes them.

Security notes
==============

* **Raw tokens are never stored on disk.**  The primary key is the
  SHA-256 hash of the session token (``token_hash``).  This means that
  even if the SQLite file is leaked, an attacker cannot reconstruct
  valid session cookies from it.
* The store is **single-process per server**, but :class:`fastapi.FastAPI`
  handlers run on multiple threads, so all writes are serialized
  through a module-level :class:`threading.Lock`.
* The :meth:`SessionStore.list` API surfaces a 16-character
  SHA-256 ``fingerprint`` (derived from the *hash*, not the raw token)
  so the dashboard can show "which browser issued this session" without
  leaking the secret.
* Revocations and expirations are evaluated lazily on
  :meth:`SessionStore.is_valid` so that an offline reader process can
  inspect the DB without holding the lock.

Migration
=========

On startup the store detects the old ``token TEXT PRIMARY KEY`` schema
and migrates it in-place: each row's ``token`` value is hashed to
produce the new ``token_hash``, then the table is recreated.  Old
sessions survive the migration (the hash is deterministic), but the
raw token is permanently discarded from disk.

Configuration
=============

The DB path is resolved in this priority order:

1. ``MEMORY_OS_SESSIONS_DB`` (full file path; parent dirs are created).
2. ``$XDG_STATE_HOME/openclaw-memory-os/sessions.db``.
3. ``$MEMORY_OS_RECALL_STATE_DIR/openclaw-memory-os/sessions.db``.
4. ``~/.local/state/openclaw-memory-os/sessions.db``.
"""

from __future__ import annotations

import hashlib
import hmac
import os
import sqlite3
import stat
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _default_db_path() -> Path:
    """Resolve the on-disk path for the sessions DB.

    Honours ``MEMORY_OS_SESSIONS_DB`` when present; otherwise falls back
    to the same state directory layout used by ``recall_feedback``.
    """
    override = os.environ.get("MEMORY_OS_SESSIONS_DB", "").strip()
    if override:
        return Path(override).expanduser()

    base = Path(
        os.environ.get(
            "MEMORY_OS_RECALL_STATE_DIR",
            os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")),
        )
    )
    return base / "openclaw-memory-os" / "sessions.db"


def _hash_token(token: str) -> str:
    """Return the SHA-256 hex digest of a session token.

    This is the value stored as the primary key in the sessions table.
    Using a hash means the raw token never touches disk.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _secure_regular_file(path: Path, *, create: bool) -> None:
    """Enforce an owner-only regular file on POSIX or fail closed.

    ``os.open`` with ``O_NOFOLLOW`` (where available) prevents a session DB
    path from being redirected through a symlink. ``fchmod`` and ``fstat``
    operate on the opened descriptor, closing the pathname race between a
    chmod call and verification. Permission failures are security failures,
    not warnings: callers must not issue browser sessions against a database
    that could be read by other users.
    """
    if os.name == "nt":  # Windows ACLs are outside the POSIX mode contract.
        return
    if path.is_symlink():
        raise PermissionError(f"session database path must not be a symlink: {path}")

    flags = os.O_RDWR | (os.O_CREAT if create else 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    fd: Optional[int] = None
    try:
        fd = os.open(path, flags, 0o600)
        before = os.fstat(fd)
        if not stat.S_ISREG(before.st_mode):
            raise PermissionError(f"session database is not a regular file: {path}")
        os.fchmod(fd, 0o600)
        mode = stat.S_IMODE(os.fstat(fd).st_mode)
        if mode != 0o600:
            raise PermissionError(
                f"session database permissions are {oct(mode)}, expected 0o600: {path}"
            )
    except OSError as exc:
        raise PermissionError(f"cannot secure session database file: {path}") from exc
    finally:
        if fd is not None:
            os.close(fd)


def _prepare_session_db_file(path: Path) -> None:
    """Create the DB path privately without changing an existing parent mode."""
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    _secure_regular_file(path, create=True)


def _secure_sqlite_files(path: Path) -> None:
    """Verify the database and any WAL/SHM sidecars are owner-only."""
    candidates = (
        path,
        path.with_name(path.name + "-wal"),
        path.with_name(path.name + "-shm"),
    )
    for candidate in candidates:
        if candidate.exists() or candidate.is_symlink():
            _secure_regular_file(candidate, create=False)


_SCHEMA_SQL_V2 = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        token_hash TEXT PRIMARY KEY,
        issued_at TEXT NOT NULL,
        max_age INTEGER NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0,
        last_seen_at TEXT,
        user_id TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_revoked ON sessions(revoked);",
    "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id);",
)

# Old schema (v1) — used only for migration detection.
_SCHEMA_SQL_V1 = (
    """
    CREATE TABLE IF NOT EXISTS sessions (
        token TEXT PRIMARY KEY,
        issued_at TEXT NOT NULL,
        max_age INTEGER NOT NULL,
        revoked INTEGER NOT NULL DEFAULT 0,
        last_seen_at TEXT
    );
    """,
    "CREATE INDEX IF NOT EXISTS idx_sessions_revoked ON sessions(revoked);",
)


_write_lock = threading.Lock()


def _detect_schema_version(conn: sqlite3.Connection) -> int:
    """Return the supported sessions schema version.

    ``0`` means the table does not exist. Any existing but unrecognised table is
    rejected: treating an unknown schema as an empty database could otherwise
    overwrite operator state.
    """
    rows = conn.execute("PRAGMA table_info(sessions)").fetchall()
    if not rows:
        return 0
    columns = {
        str(row["name"] if isinstance(row, sqlite3.Row) else row[1])
        for row in rows
    }
    if "token_hash" in columns and "user_id" in columns and "token" not in columns:
        return 3
    if "token_hash" in columns and "token" not in columns:
        return 2
    if "token" in columns and "token_hash" not in columns:
        return 1
    raise sqlite3.DatabaseError(
        f"unsupported sessions schema columns: {sorted(columns)!r}"
    )


def _schema_transaction(conn: sqlite3.Connection, operation) -> None:  # type: ignore[no-untyped-def]
    """Run one schema mutation atomically and roll back every failure."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        operation()
        conn.execute("COMMIT")
    except BaseException:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        raise


def _create_v3_schema(conn: sqlite3.Connection) -> None:
    for stmt in _SCHEMA_SQL_V2:
        conn.execute(stmt)


def _create_fresh_schema(conn: sqlite3.Connection) -> None:
    _schema_transaction(conn, lambda: _create_v3_schema(conn))


def _migrate_v1_to_v2(conn: sqlite3.Connection) -> None:
    """Atomically migrate raw-token v1 rows directly to the v3 schema.

    Token hashing is completed before any DDL begins. The old table is renamed
    inside the same SQLite transaction, so an insert, DDL, or commit failure
    restores the original table and every raw row for a safe retry.
    """
    rows = conn.execute(
        "SELECT token, issued_at, max_age, revoked, last_seen_at FROM sessions"
    ).fetchall()
    transformed = []
    for row in rows:
        raw_token = row["token"] if isinstance(row, sqlite3.Row) else row[0]
        transformed.append(
            (
                _hash_token(str(raw_token)),
                row["issued_at"] if isinstance(row, sqlite3.Row) else row[1],
                int(row["max_age"] if isinstance(row, sqlite3.Row) else row[2]),
                int(row["revoked"] if isinstance(row, sqlite3.Row) else row[3]),
                row["last_seen_at"] if isinstance(row, sqlite3.Row) else row[4],
            )
        )

    def operation() -> None:
        conn.execute("DROP INDEX IF EXISTS idx_sessions_revoked")
        conn.execute("DROP INDEX IF EXISTS idx_sessions_user_id")
        conn.execute("ALTER TABLE sessions RENAME TO sessions_v1_backup")
        _create_v3_schema(conn)
        conn.executemany(
            "INSERT INTO sessions "
            "(token_hash, issued_at, max_age, revoked, last_seen_at, user_id) "
            "VALUES (?, ?, ?, ?, ?, NULL)",
            transformed,
        )
        conn.execute("DROP TABLE sessions_v1_backup")

    _schema_transaction(conn, operation)


def _migrate_v2_to_v3(conn: sqlite3.Connection) -> None:
    """Atomically add ``user_id`` and its index to a v2 database."""

    def operation() -> None:
        conn.execute("ALTER TABLE sessions ADD COLUMN user_id TEXT")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_sessions_user_id ON sessions(user_id)"
        )

    _schema_transaction(conn, operation)


class SessionStore:
    """Persistent SQLite-backed session store.

    The store is intentionally tiny: a single ``sessions`` table keyed
    by the **SHA-256 hash** of the opaque cookie token, with a ``revoked``
    flag and ``max_age`` (in seconds).  Expiry is computed lazily on read
    so that no background sweep is required.

    Parameters
    ----------
    db_path:
        Path to the SQLite file. Parent directories are created.
        Defaults to the value returned by :func:`_default_db_path`.
    """

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path: Path = Path(db_path) if db_path else _default_db_path()
        _prepare_session_db_file(self._db_path)
        self._conn: sqlite3.Connection = sqlite3.connect(
            str(self._db_path),
            check_same_thread=False,
            isolation_level=None,
            timeout=5.0,
        )
        try:
            self._conn.row_factory = sqlite3.Row
            # Use DELETE journal mode instead of WAL. The WAL mode in
            # this codebase has a stale-fd regression under Python 3.12's
            # sqlite3 binding where journal_mode changes (or even WAL
            # file opens) leave fd 18 pointing at a deleted inode, so
            # subsequent commits never reach the main DB and read-only
            # queries that bypass the stale fd see only the pre-commit
            # state. DELETE mode writes directly to the main DB file,
            # so the regression cannot occur and every commit is
            # visible to other readers. The trade-off is one fsync per
            # commit instead of a periodic WAL append, which is
            # acceptable for a session-issuance workload.
            self._conn.execute("PRAGMA journal_mode=DELETE")
            # synchronous=NORMAL fsyncs the main DB file on every commit,
            # which is the durability contract the session store requires
            # across restarts. NORMAL avoids a second fsync of the WAL
            # that FULL would add (no WAL exists in DELETE mode) while
            # still guaranteeing committed rows survive process crash.
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            with _write_lock:
                version = _detect_schema_version(self._conn)
                if version == 0:
                    _create_fresh_schema(self._conn)
                elif version == 1:
                    _migrate_v1_to_v2(self._conn)
                elif version == 2:
                    _migrate_v2_to_v3(self._conn)
                elif version != 3:  # defensive: detector currently raises first.
                    raise sqlite3.DatabaseError(
                        f"unsupported sessions schema version: {version}"
                    )
            _secure_sqlite_files(self._db_path)
        except BaseException:
            self._conn.close()
            raise

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying connection.

        Safe to call multiple times. After ``close()`` any further use
        of the store will raise :class:`sqlite3.ProgrammingError`; the
        intended use is to drop the reference and let GC reclaim it.

        On close we only hand the connection back to the binding and
        let Python's sqlite3 module close the WAL file. We do NOT run
        ``PRAGMA wal_checkpoint(TRUNCATE)`` here because that pragma
        blocks until pending writers release their locks, and during
        FastAPI lifespan shutdown those writers may already be in a
        stale-fd state where the lock would never be released cleanly.
        """
        conn = getattr(self, "_conn", None)
        if conn is None:
            return
        try:
            conn.close()
        except Exception:  # pragma: no cover - defensive
            pass

    # ------------------------------------------------------------------
    # Write path
    # ------------------------------------------------------------------

    def create(
        self,
        token: str,
        max_age: int,
        *,
        issued_at: Optional[datetime] = None,
        user_id: Optional[str] = None,
    ) -> None:
        """Insert a new session row.

        The raw ``token`` is hashed before storage; only the SHA-256
        digest is written to disk.  ``issued_at`` defaults to "now in
        UTC".  ``max_age`` is stored in seconds.  If a row already
        exists for the same token hash it is overwritten.

        ``user_id`` is an optional opaque identifier that links the
        session to a logical account (runbook G0.7). It is used by
        :meth:`revoke_all_for_user` to perform scoped session
        invalidation when the user's password is rotated. ``None``
        preserves backward compatibility with sessions issued before
        the user-scoped revoke hook shipped.
        """
        if not token:
            raise ValueError("token must be a non-empty string")
        if max_age is None or int(max_age) < 0:
            raise ValueError("max_age must be a non-negative integer")
        token_hash = _hash_token(token)
        ts = (issued_at or datetime.now(timezone.utc)).astimezone(timezone.utc)
        issued_iso = ts.isoformat()
        with _write_lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO sessions "
                "(token_hash, issued_at, max_age, revoked, last_seen_at, user_id) "
                "VALUES (?, ?, ?, 0, NULL, ?)",
                (token_hash, issued_iso, int(max_age), user_id),
            )

    def contains(self, token: Optional[str]) -> bool:
        """Return whether a hashed row exists without exposing the raw token."""
        if not token:
            return False
        row = self._conn.execute(
            "SELECT 1 FROM sessions WHERE token_hash = ?",
            (_hash_token(token),),
        ).fetchone()
        return row is not None

    def revoke(self, token: Optional[str]) -> bool:
        """Mark ``token`` as revoked. Idempotent.

        The raw token is hashed to look up the row.  Returns ``True`` if
        a row was updated, ``False`` otherwise (unknown token).
        """
        if not token:
            return False
        token_hash = _hash_token(token)
        with _write_lock:
            cur = self._conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE token_hash = ?",
                (token_hash,),
            )
            return cur.rowcount > 0

    def revoke_all(self) -> int:
        """Revoke every non-revoked row. Returns the count updated."""
        with _write_lock:
            cur = self._conn.execute(
                "UPDATE sessions SET revoked = 1 WHERE revoked = 0"
            )
            return int(cur.rowcount or 0)

    def revoke_all_for_user(self, user_id: Optional[str]) -> int:
        """Revoke every non-revoked row whose ``user_id`` matches.

        Runbook G0.7 (password change → revoke all): a successful
        password rotation must invalidate every existing session that
        belongs to the same logical account, regardless of which
        device / cookie issued it. Pre-existing sessions that have no
        ``user_id`` (``NULL``) are intentionally NOT touched — they
        were issued before the user-scoped revoke hook shipped and
        the operator can revoke them explicitly via :meth:`revoke_all`
        if needed.

        ``user_id`` is opaque to the store (string equality only). The
        caller decides the namespace — typically the same value the
        login flow stores as a session cookie attribute.

        Returns the count of rows flipped to ``revoked = 1``. A
        ``None`` or empty ``user_id`` yields zero (matches the
        behaviour of an unmatched query).
        """
        if not user_id:
            return 0
        with _write_lock:
            cur = self._conn.execute(
                "UPDATE sessions SET revoked = 1 "
                "WHERE user_id = ? AND revoked = 0",
                (user_id,),
            )
            return int(cur.rowcount or 0)

    # ------------------------------------------------------------------
    # Read path (lock-free)
    # ------------------------------------------------------------------

    def is_valid(self, token: Optional[str]) -> bool:
        """Return ``True`` iff the token exists, is not revoked, and has not expired.

        The raw token is hashed to look up the row.  Unknown, revoked,
        expired, and ``None`` tokens all return ``False``.  Expired
        tokens are NOT marked revoked automatically; they remain in the
        table until a future revoke pass prunes them.
        """
        if not token:
            return False
        token_hash = _hash_token(token)
        row = self._conn.execute(
            "SELECT issued_at, max_age, revoked FROM sessions WHERE token_hash = ?",
            (token_hash,),
        ).fetchone()
        if row is None:
            return False
        if int(row["revoked"]) != 0:
            return False
        try:
            issued_dt = datetime.fromisoformat(row["issued_at"])
        except ValueError:
            return False
        if issued_dt.tzinfo is None:
            issued_dt = issued_dt.replace(tzinfo=timezone.utc)
        max_age = int(row["max_age"])
        expires_at = issued_dt + timedelta(seconds=max_age)
        return datetime.now(timezone.utc) < expires_at

    def list(
        self, *, current_token: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Return non-secret metadata for every known session.

        The raw token never appears in the result — callers receive a
        16-character ``fingerprint`` (derived from the stored hash, not
        the raw token) plus issued/expire/revoked fields.  Rows are
        ordered by ``issued_at`` descending so the dashboard naturally
        shows the most recent session first.
        """
        rows = self._conn.execute(
            "SELECT token_hash, issued_at, max_age, revoked "
            "FROM sessions ORDER BY issued_at DESC"
        ).fetchall()
        current_hash = _hash_token(current_token) if current_token else None
        out: List[Dict[str, Any]] = []
        for row in rows:
            token_hash = row["token_hash"]
            revoked = bool(int(row["revoked"]))
            # Lazy expiry: a row whose max_age has elapsed is shown as
            # revoked so the dashboard surfaces it correctly without a
            # background sweep.
            try:
                issued_dt = datetime.fromisoformat(row["issued_at"])
            except ValueError:
                expired = True
            else:
                if issued_dt.tzinfo is None:
                    issued_dt = issued_dt.replace(tzinfo=timezone.utc)
                expired = (
                    datetime.now(timezone.utc)
                    >= issued_dt + timedelta(seconds=int(row["max_age"]))
                )
            entry_revoked = revoked or expired
            # The fingerprint is the first 16 chars of the stored hash.
            # Since the hash is already SHA-256, this is safe.
            fingerprint = token_hash[:16]
            current = bool(
                current_hash and hmac.compare_digest(current_hash, token_hash)
            )
            out.append(
                {
                    "fingerprint": fingerprint,
                    "issued_at": row["issued_at"],
                    "max_age": int(row["max_age"]),
                    "revoked": entry_revoked,
                    "current": current,
                }
            )
        return out

    # ------------------------------------------------------------------
    # Context-manager convenience (optional)
    # ------------------------------------------------------------------

    def __enter__(self) -> "SessionStore":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[no-untyped-def]
        self.close()
