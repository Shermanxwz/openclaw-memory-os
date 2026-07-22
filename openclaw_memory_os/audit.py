"""SQLite-based audit log for OpenClaw Memory OS.

Tracks all significant events: ingestion, recall tests, feedback,
consolidation, and system maintenance. The database is created
automatically on first use at a configurable path.

Schema::

    CREATE TABLE audit_log (
        id          INTEGER PRIMARY KEY AUTOINCREMENT,
        timestamp   TEXT NOT NULL,  -- ISO-8601 UTC
        action      TEXT NOT NULL,
        actor       TEXT,
        memory_id   TEXT,
        detail      TEXT
    );

All timestamps are stored as ISO-8601 UTC strings.
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import AuditLogEntry, utcnow

logger = logging.getLogger(__name__)


class AuditStore:
    """Thread-safe audit log backed by SQLite.

    Usage::

        store = AuditStore()
        store.log("ingest", memory_id="mem-001", detail="Ingested 42 chunks")
        entries = store.list_recent(limit=20)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._local = threading.local()
        self._db_path = db_path or self._default_path()
        logger.debug("AuditStore initialized at %s", self._db_path)

    @staticmethod
    def _default_path() -> Path:
        import os as _os
        # Allow the path to be overridden via env, e.g. for systemd sandboxes
        # where ProtectHome=read-only blocks writes under $HOME.
        env_path = _os.environ.get("OPENCLAW_MEMORY_OS_AUDIT_PATH")
        if env_path:
            return Path(env_path)
        data_home = Path(_os.environ.get("XDG_DATA_HOME", _os.path.expanduser("~/.local/share")))
        return data_home / "openclaw-memory-os" / "audit_log.sqlite"

    def _connection(self) -> sqlite3.Connection:
        """Get a thread-local connection (auto-creates schema on first use)."""
        conn = getattr(self._local, "conn", None)
        if conn is None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(self._db_path), timeout=10)
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute(
                """CREATE TABLE IF NOT EXISTS audit_log (
                    id        INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    action    TEXT NOT NULL,
                    actor     TEXT,
                    memory_id TEXT,
                    detail    TEXT
                )"""
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_timestamp ON audit_log(timestamp)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_audit_action ON audit_log(action)"
            )
            conn.commit()
            self._local.conn = conn
        return conn

    def log(
        self,
        action: str,
        *,
        actor: Optional[str] = None,
        memory_id: Optional[str] = None,
        detail: Optional[str] = None,
    ) -> int:
        """Insert an audit log entry. Returns the row ID."""
        conn = self._connection()
        ts = utcnow().isoformat()
        if detail is not None and len(detail) > 2000:
            detail = detail[:2000]
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, actor, memory_id, detail) VALUES (?, ?, ?, ?, ?)",
            (ts, action, actor, memory_id, detail),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug("Audit: action=%s memory_id=%s row=%d", action, memory_id, row_id)
        return row_id

    def list_recent(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        action: Optional[str] = None,
    ) -> List[AuditLogEntry]:
        """Return recent audit log entries, newest first."""
        conn = self._connection()
        if action:
            rows = conn.execute(
                "SELECT * FROM audit_log WHERE action = ? ORDER BY id DESC LIMIT ? OFFSET ?",
                (action, limit, offset),
            )
        else:
            rows = conn.execute(
                "SELECT * FROM audit_log ORDER BY id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            )
        result: List[AuditLogEntry] = []
        for r in rows:
            result.append(
                AuditLogEntry(
                    id=r["id"],
                    timestamp=datetime.fromisoformat(r["timestamp"]),
                    action=r["action"],
                    actor=r["actor"],
                    memory_id=r["memory_id"],
                    detail=r["detail"],
                )
            )
        return result

    def count(self, action: Optional[str] = None) -> int:
        conn = self._connection()
        if action:
            row = conn.execute("SELECT COUNT(*) as c FROM audit_log WHERE action = ?", (action,)).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as c FROM audit_log").fetchone()
        return row["c"] if row else 0

    # ------------------------------------------------------------------
    # v0.3.0 structured feedback methods
    # ------------------------------------------------------------------

    def record_recall_run(
        self,
        query_id: str,
        query_text: str,
        *,
        mode: str = "hybrid",
        policy_version: str = "",
        took_ms: float = 0.0,
        backend: str = "",
        total_considered: int = 0,
        fallback_used: bool = False,
        fallback_added: int = 0,
    ) -> int:
        """Insert a recall run row. Returns the row ID."""
        conn = self._connection()
        ts = utcnow().isoformat()
        conn.execute(
            "INSERT OR IGNORE INTO recall_runs "
            "(query_id, query_text, mode, policy_version, timestamp, "
            "took_ms, backend, total_considered, fallback_used, fallback_added) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query_id, query_text, mode, policy_version, ts,
             took_ms, backend, total_considered,
             int(fallback_used), fallback_added),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.debug("RecallRun: query_id=%s row=%d", query_id, row_id)
        return row_id

    def record_recall_result(
        self,
        query_id: str,
        candidate_key: str,
        *,
        collection: str = "",
        memory_id: str = "",
        score: float = 0.0,
        tier: str = "medium",
        status: str = "active",
        importance: float = 0.5,
        rank_position: int = 0,
    ) -> int:
        """Insert a recall result row. Returns the row ID."""
        conn = self._connection()
        conn.execute(
            "INSERT INTO recall_results "
            "(query_id, candidate_key, collection, memory_id, score, "
            "tier, status, importance, rank_position) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (query_id, candidate_key, collection, memory_id, score,
             tier, status, importance, rank_position),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        return row_id

    def record_feedback_event(
        self,
        query_id: str,
        candidate_key: str,
        useful: bool,
        *,
        query_text: str = "",
        note: Optional[str] = None,
        actor: Optional[str] = None,
    ) -> int:
        """Insert a structured feedback event. Returns the row ID."""
        conn = self._connection()
        ts = utcnow().isoformat()
        conn.execute(
            "INSERT INTO feedback_events "
            "(query_id, candidate_key, useful, query_text, note, actor, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (query_id, candidate_key, int(useful), query_text,
             note, actor, ts),
        )
        conn.commit()
        row_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
        logger.info(
            "FeedbackEvent: query_id=%s candidate_key=%s useful=%s row=%d",
            query_id, candidate_key, useful, row_id,
        )
        return row_id

    def cleanup_recall_runs(self, max_age_days: int = 180) -> int:
        """Delete recall_runs (and cascaded results/feedback) older than max_age_days.

        Returns the number of runs deleted.
        """
        conn = self._connection()
        cutoff = (utcnow() - __import__("datetime").timedelta(days=max_age_days)).isoformat()
        # Get IDs to delete
        rows = conn.execute(
            "SELECT query_id FROM recall_runs WHERE timestamp < ?",
            (cutoff,),
        ).fetchall()
        deleted = 0
        for r in rows:
            qid = r["query_id"]
            conn.execute("DELETE FROM feedback_events WHERE query_id = ?", (qid,))
            conn.execute("DELETE FROM recall_results WHERE query_id = ?", (qid,))
            conn.execute("DELETE FROM recall_runs WHERE query_id = ?", (qid,))
            deleted += 1
        conn.commit()
        if deleted:
            logger.info("Cleaned up %d recall_runs older than %d days", deleted, max_age_days)
        return deleted

    def get_feedback_for_query(self, query_id: str) -> List[Dict[str, Any]]:
        """Return feedback events for a specific query_id."""
        conn = self._connection()
        rows = conn.execute(
            "SELECT * FROM feedback_events WHERE query_id = ? ORDER BY id",
            (query_id,),
        ).fetchall()
        return [dict(r) for r in rows]

    def get_feedback_for_candidate(self, candidate_key: str) -> List[Dict[str, Any]]:
        """Return feedback events for a specific candidate_key."""
        conn = self._connection()
        rows = conn.execute(
            "SELECT * FROM feedback_events WHERE candidate_key = ? ORDER BY id",
            (candidate_key,),
        ).fetchall()
        return [dict(r) for r in rows]

    def count_feedback(self, *, query_id: Optional[str] = None) -> int:
        conn = self._connection()
        if query_id:
            row = conn.execute(
                "SELECT COUNT(*) as c FROM feedback_events WHERE query_id = ?",
                (query_id,),
            ).fetchone()
        else:
            row = conn.execute("SELECT COUNT(*) as c FROM feedback_events").fetchone()
        return row["c"] if row else 0

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn:
            conn.close()
            self._local.conn = None


# Module-level singleton for convenience.
_default_store: Optional[AuditStore] = None
_lock = threading.Lock()


def get_audit_store(db_path: Optional[Path] = None) -> AuditStore:
    """Get or create the module-level audit store singleton."""
    global _default_store
    if _default_store is None:
        with _lock:
            if _default_store is None:
                _default_store = AuditStore(db_path=db_path)
    return _default_store
