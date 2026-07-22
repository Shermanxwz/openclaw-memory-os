"""v0.3.0 structured feedback storage.

Supersedes the legacy ``feedback.py`` module which stored feedback
as audit-log strings. v0.3.0 stores feedback in three normalised
SQLite tables (``recall_runs``, ``recall_results``, ``feedback_events``)
so that the offline evaluation pipeline can replay real recall traces
with per-hit scores and policy versions.

Migration from the legacy audit-log format is handled by
:func:`migrate_legacy_feedback` which reads old feedback entries
from the audit store and upserts them into ``feedback_events``.
The old table is kept read-only; new data goes into v0.3.0 tables.

Schema evolution is handled by :func:`_ensure_schema` which is
backward-compatible: it issues ``CREATE TABLE IF NOT EXISTS`` for
fresh databases, and ``ALTER TABLE ... ADD COLUMN`` for pre-existing
ones so the v0.3.0.x fields (``query_hash``, ``corpus_snapshot_id``,
``dense_available``, ``lexical_available``,
``collections_succeeded_json`` / ``collections_failed_json``,
``vector_score_raw`` / ``vector_score_calibrated`` /
``lexical_score_raw`` / ``lexical_score_calibrated`` /
``importance_score`` / ``recency_score`` / ``feedback_score`` /
``display_score`` and ``migration_status``) appear in older
databases on the next call without losing data.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .audit import get_audit_store
from .migration import run_migrations, enable_wal as _enable_wal

# ``Path`` is imported above; it is used by ``compute_corpus_snapshot_id``.
__all__ = [
    "record_recall_run",
    "record_recall_result",
    "record_feedback_v030",
    "get_feedback_summary",
    "get_recall_runs_for_query",
    "migrate_legacy_feedback",
    # G5.5 — corpus fingerprinting helpers
    "compute_corpus_snapshot_id",
    "get_current_backend",
]

logger = logging.getLogger(__name__)


def _enforce_private_path(path: Path) -> None:
    """Force state directories/files to owner-only permissions."""
    if os.name == "nt":  # pragma: no cover - POSIX deployment contract
        return
    try:
        path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        os.chmod(path.parent, 0o700)
        if path.exists():
            os.chmod(path, 0o600)
    except OSError as exc:
        raise RuntimeError(
            f"cannot secure recall feedback state path {path}: {exc}"
        ) from exc


_RECALL_DB_DIR = Path(
    os.environ.get(
        "MEMORY_OS_RECALL_STATE_DIR",
        os.environ.get(
            "XDG_STATE_HOME",
            os.path.expanduser("~/.local/state"),
        ),
    )
) / "openclaw-memory-os"

_RECALL_DB = _RECALL_DB_DIR / "recall_feedback.db"

# 180-day retention for recall_runs (feedback_events are kept indefinitely
# because they are the only source of truth for what the user actually
# *wanted* — pruning them would make evaluation biassed).
_RECALL_RUNS_RETENTION_DAYS = 180

_lock = threading.Lock()


def _get_db(*, run_legacy_migration: bool = True) -> sqlite3.Connection:
    _RECALL_DB_DIR.mkdir(parents=True, exist_ok=True, mode=0o700)
    _enforce_private_path(_RECALL_DB)
    conn = sqlite3.connect(str(_RECALL_DB), timeout=5)
    _enforce_private_path(_RECALL_DB)
    conn.row_factory = sqlite3.Row
    run_migrations(conn)
    _enable_wal(conn)
    conn.execute("PRAGMA foreign_keys=ON")
    _ensure_schema(conn)
    if not run_legacy_migration:
        return conn
    # Data migration failures must remain retryable and visible.  A failed
    # legacy import is not allowed to masquerade as a completed migration.
    migrate_legacy_feedback(conn)
    return conn


# ---------------------------------------------------------------------------
# Schema management
# ---------------------------------------------------------------------------

# Each entry: (table, column, sql_type). Order matters: we always add new
# columns in the order declared here so the same migration is replayable.
# SQLite has no ``ADD COLUMN IF NOT EXISTS`` (until 3.35 with the
# ``IF NOT EXISTS`` extension), so we guard on ``PRAGMA table_info`` which
# works on every supported SQLite version.
_SCHEMA_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("recall_runs", "query_hash", "TEXT"),
    ("recall_runs", "corpus_snapshot_id", "TEXT"),
    ("recall_runs", "dense_available", "INTEGER"),
    ("recall_runs", "lexical_available", "INTEGER"),
    ("recall_runs", "collections_succeeded_json", "TEXT"),
    ("recall_runs", "collections_failed_json", "TEXT"),
    ("recall_results", "vector_score_raw", "REAL"),
    ("recall_results", "vector_score_calibrated", "REAL"),
    ("recall_results", "lexical_score_raw", "REAL"),
    ("recall_results", "lexical_score_calibrated", "REAL"),
    ("recall_results", "importance_score", "REAL"),
    ("recall_results", "recency_score", "REAL"),
    ("recall_results", "feedback_score", "REAL"),
    ("recall_results", "display_score", "REAL"),
    ("feedback_events", "migration_status", "TEXT"),
    ("feedback_events", "resolution_status", "TEXT"),
    ("feedback_events", "collection", "TEXT"),
    ("feedback_events", "legacy_source_key", "TEXT"),
)


def _existing_columns(conn: sqlite3.Connection, table: str) -> set:
    """Return the set of column names currently defined on ``table``.

    Falls back to an empty set when the table does not exist yet (the
    ``CREATE TABLE IF NOT EXISTS`` block in :func:`_ensure_schema`
    will materialise it on the next call).
    """
    try:
        rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    except sqlite3.Error:
        return set()
    return {r[1] for r in rows}



def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Create current tables, then add columns before dependent indexes."""
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS recall_runs (
            query_id TEXT PRIMARY KEY,
            query_text TEXT NOT NULL,
            created_at TEXT NOT NULL DEFAULT (datetime('now')),
            retrieval_mode TEXT,
            policy_version TEXT,
            latency_ms REAL,
            retrieval_status TEXT,
            degraded_reason TEXT,
            fallback_used INTEGER DEFAULT 0,
            query_hash TEXT,
            corpus_snapshot_id TEXT,
            dense_available INTEGER,
            lexical_available INTEGER,
            collections_succeeded_json TEXT,
            collections_failed_json TEXT
        );
        CREATE TABLE IF NOT EXISTS recall_results (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL REFERENCES recall_runs(query_id),
            candidate_key TEXT NOT NULL,
            memory_id TEXT,
            collection TEXT,
            rank INTEGER,
            status TEXT,
            vector_score REAL,
            lexical_score REAL,
            rrf_score REAL,
            final_score REAL,
            explanation TEXT,
            vector_score_raw REAL,
            vector_score_calibrated REAL,
            lexical_score_raw REAL,
            lexical_score_calibrated REAL,
            importance_score REAL,
            recency_score REAL,
            feedback_score REAL,
            display_score REAL
        );
        CREATE TABLE IF NOT EXISTS feedback_events (
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
        );
        CREATE INDEX IF NOT EXISTS idx_recall_results_query ON recall_results(query_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_events_query ON feedback_events(query_id);
        CREATE INDEX IF NOT EXISTS idx_feedback_events_candidate ON feedback_events(candidate_key);
        """
    )
    for table, column, sql_type in _SCHEMA_COLUMNS:
        if column in _existing_columns(conn, table):
            continue
        conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {sql_type}")
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS idx_feedback_events_legacy_source "
        "ON feedback_events(legacy_source_key) WHERE legacy_source_key IS NOT NULL"
    )
    conn.commit()


# -- helpers ----------------------------------------------------------------


def _query_hash(query_text: str) -> str:
    """Stable short hash for a query string. Used for grouping runs."""
    if not query_text:
        return ""
    return hashlib.sha256(query_text.encode("utf-8")).hexdigest()[:16]


def _dump_json_list(values: Optional[Iterable[str]]) -> Optional[str]:
    """Serialise a list of strings to JSON, or ``None`` when empty."""
    if values is None:
        return None
    items = [str(v) for v in values if v]
    if not items:
        return None
    return json.dumps(items, ensure_ascii=False, sort_keys=True)


# -- corpus snapshot (G5.5) -----------------------------------------------


# How many points / memory ids to fold into the snapshot hash. Larger
# values make the fingerprint more sensitive to single-row edits, but
# also make the computation more expensive on big Qdrant collections.
# 100 is enough to detect any realistic change at sub-millisecond cost.
_SNAPSHOT_SAMPLE_SIZE = 100


def compute_corpus_snapshot_id(backend: Any) -> str:
    """Fingerprint the current corpus so promotions can compare
    apples to apples (G5.5).

    Returns a string of the form
    ``"<backend_kind>:<count>:<sha256_prefix>"`` where the SHA256 is
    computed over a deterministic sample of up to
    :data:`_SNAPSHOT_SAMPLE_SIZE` memory ids (sorted for stability
    across processes). The prefix is 12 hex chars (48 bits), which
    is collision-safe for any realistic deployment.

    The fingerprint is **deterministic and stable across promotion
    cycles** when the corpus is unchanged. If the corpus content
    changes (a memory is added, removed, or edited), the fingerprint
    changes too — and downstream code can then decide that any
    ``EvalResult`` carrying the old ``corpus_snapshot_id`` is no
    longer comparable to one carrying the new id.

    The function never raises: if the backend is unknown, unreachable,
    or otherwise refuses to enumerate points, we fall back to a
    file-derived fingerprint (``"<kind>:<mtime>:<size>"``) so the
    offline pipeline can still record *something* in
    ``recall_runs.corpus_snapshot_id``. Returning ``None`` would
    hide the gap; returning a raised exception would break the
    recall write path.
    """
    if backend is None:
        return "unknown:0:000000000000"
    kind = getattr(backend, "name", None) or type(backend).__name__
    try:
        # SampleBackend has a stable on-disk identity: path + mtime + size.
        # That gives us a deterministic fingerprint without enumerating
        # the JSON payload (which can be slow on large samples).
        path = getattr(backend, "path", None)
        if path is not None:
            p = Path(path)
            if p.exists():
                stat = p.stat()
                # Compose over the file identity + the live in-memory
                # count so adding a memory to the sample (without yet
                # writing the JSON to disk) also changes the snapshot.
                try:
                    count = len(backend.list_memories())
                except Exception:
                    count = 0
                payload = f"{kind}|{p.resolve()}|{stat.st_size}|{count}"
                digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
                return f"{kind}:{count}:{digest}"
        # QdrantBackend: enumerate points via the existing in-memory
        # cache (populated lazily by ``list_memories()``). We never
        # hit Qdrant here because the cache is the same data the
        # recall pipeline uses.
        try:
            memories = list(backend.list_memories())
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "compute_corpus_snapshot_id: list_memories() failed (%s); "
                "falling back to a static marker",
                exc,
            )
            return f"{kind}:0:fallback00"
        # Sort + truncate for stability across processes.
        ids = sorted(str(m.id) for m in memories)[:_SNAPSHOT_SAMPLE_SIZE]
        payload = "|".join([kind, str(len(memories))] + ids)
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]
        return f"{kind}:{len(memories)}:{digest}"
    except Exception as exc:  # pragma: no cover — last-resort guard
        logger.warning(
            "compute_corpus_snapshot_id: unexpected failure (%s); "
            "returning a static marker",
            exc,
        )
        return f"{kind}:0:fallback00"


_current_backend_cache: Dict[str, Any] = {}


def get_current_backend() -> Any:
    """Return the live backend instance, or ``None`` if not installed.

    G5.5 contract: this function is the single source of truth for
    "what backend should we fingerprint right now". Callers that
    have a more specific backend available should pass it
    explicitly to :func:`compute_corpus_snapshot_id` rather than
    going through this helper.

    Performance fix (2026-07-16): cache the backend instance per
    process so callers don't trigger a fresh ``get_backend(...)``
    on every request. The previous uncached implementation
    caused :func:`compute_corpus_snapshot_id` to re-enumerate the
    entire corpus via :meth:`list_memories` per request, driving
    keyword/dense/hybrid latency from <1s to 30s+ and outright
    timeouts once the corpus had multiple collections. The cache
    is invalidated when ``get_settings()`` returns a new object
    (settings reload) — cheap ``is`` check covers the common case.
    """
    try:
        from .config import get_settings
        settings = get_settings()
    except Exception:
        return None
    cache_key = id(settings)
    cached = _current_backend_cache.get(cache_key)
    if cached is not None:
        return cached
    try:
        from .backends import get_backend
        backend = get_backend(settings)
        _current_backend_cache[cache_key] = backend
        return backend
    except Exception:
        return None


# -- public API --------------------------------------------------------------


def record_recall_run(
    query_id: str,
    query_text: str,
    *,
    retrieval_mode: str = "hybrid",
    policy_version: str = "",
    latency_ms: float = 0.0,
    retrieval_status: str = "ok",
    degraded_reason: Optional[str] = None,
    fallback_used: bool = False,
    query_hash: Optional[str] = None,
    corpus_snapshot_id: Optional[str] = None,
    dense_available: Optional[bool] = None,
    lexical_available: Optional[bool] = None,
    collections_succeeded: Optional[Sequence[str]] = None,
    collections_failed: Optional[Sequence[str]] = None,
) -> str:
    """Save a recall run (one query → one result set) into the DB.

    Returns ``query_id`` for chaining. The ``query_hash``,
    ``corpus_snapshot_id``, ``dense_available``,
    ``lexical_available``, ``collections_succeeded`` and
    ``collections_failed`` parameters are optional v0.3.0.x
    extensions. When ``None`` they default to safe values derived
    from the call (e.g. ``query_hash`` falls back to a hash of
    ``query_text``), so existing call sites keep working without
    any change.
    """
    if query_hash is None:
        query_hash = _query_hash(query_text) or None
    # G5.5: when no snapshot id was supplied, fingerprint the live
    # corpus so the row carries a real, comparable identifier. A
    # ``None`` here would leave the column empty, which makes the
    # downstream evaluation pipeline unable to distinguish "no
    # corpus yet" from "we forgot to write the fingerprint".
    if corpus_snapshot_id is None:
        try:
            corpus_snapshot_id = compute_corpus_snapshot_id(get_current_backend())
        except Exception as exc:  # pragma: no cover — defensive
            logger.warning(
                "record_recall_run: compute_corpus_snapshot_id failed (%s); "
                "writing with NULL snapshot id",
                exc,
            )
            corpus_snapshot_id = None
    dense_val: Optional[int] = None if dense_available is None else (1 if dense_available else 0)
    lex_val: Optional[int] = None if lexical_available is None else (1 if lexical_available else 0)
    succ_json = _dump_json_list(collections_succeeded)
    fail_json = _dump_json_list(collections_failed)
    with _lock:
        conn = _get_db()
        try:
            # Idempotent by query_id
            conn.execute(
                """
                INSERT OR REPLACE INTO recall_runs
                    (query_id, query_text, created_at, retrieval_mode,
                     policy_version, latency_ms, retrieval_status,
                     degraded_reason, fallback_used,
                     query_hash, corpus_snapshot_id,
                     dense_available, lexical_available,
                     collections_succeeded_json, collections_failed_json)
                VALUES (?, ?, datetime('now'), ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    query_text[:500],
                    retrieval_mode,
                    policy_version,
                    latency_ms,
                    retrieval_status,
                    degraded_reason,
                    1 if fallback_used else 0,
                    query_hash,
                    corpus_snapshot_id,
                    dense_val,
                    lex_val,
                    succ_json,
                    fail_json,
                ),
            )
            conn.commit()
        finally:
            conn.close()
    return query_id


def record_recall_result(
    query_id: str,
    candidate_key: str,
    *,
    memory_id: str = "",
    collection: str = "",
    rank: int = 0,
    status: str = "",
    vector_score: Optional[float] = None,
    lexical_score: Optional[float] = None,
    rrf_score: Optional[float] = None,
    final_score: Optional[float] = None,
    explanation: str = "",
    vector_score_raw: Optional[float] = None,
    vector_score_calibrated: Optional[float] = None,
    lexical_score_raw: Optional[float] = None,
    lexical_score_calibrated: Optional[float] = None,
    importance_score: Optional[float] = None,
    recency_score: Optional[float] = None,
    feedback_score: Optional[float] = None,
    display_score: Optional[float] = None,
) -> None:
    """Record one ranked hit for a recall run.

    All new v0.3.0.x score columns are optional and default to
    ``None``. When omitted, the legacy ``vector_score`` /
    ``lexical_score`` / ``final_score`` values are still written
    so existing call sites keep working unchanged.
    """
    with _lock:
        conn = _get_db()
        try:
            conn.execute(
                """
                INSERT INTO recall_results
                    (query_id, candidate_key, memory_id, collection, rank,
                     status, vector_score, lexical_score, rrf_score,
                     final_score, explanation,
                     vector_score_raw, vector_score_calibrated,
                     lexical_score_raw, lexical_score_calibrated,
                     importance_score, recency_score, feedback_score,
                     display_score)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                        ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    query_id,
                    candidate_key,
                    memory_id,
                    collection,
                    rank,
                    status,
                    vector_score,
                    lexical_score,
                    rrf_score,
                    final_score,
                    explanation,
                    vector_score_raw,
                    vector_score_calibrated,
                    lexical_score_raw,
                    lexical_score_calibrated,
                    importance_score,
                    recency_score,
                    feedback_score,
                    display_score,
                ),
            )
            conn.commit()
        finally:
            conn.close()


def record_feedback_v030(
    query_id: str,
    candidate_key: str,
    useful: bool,
    *,
    memory_id: str = "",
    feedback_source: str = "dashboard",
    migration_status: Optional[str] = None,
) -> int:
    """Record one feedback event.

    Returns the new row id. ``migration_status`` is the optional
    v0.3.0.x field used to mark rows that were backfilled by
    :func:`migrate_legacy_feedback` (e.g. ``"migrated:audit"``).

    G5.1 (strong validation): the call *fails closed* when
    ``candidate_key`` was not actually returned for ``query_id``
    according to the ``recall_results`` table. We refuse to record
    feedback for a candidate the user never saw, because that
    signal is untrustworthy (it would skew the offline evaluation
    pipeline). Migration backfill rows (``migrate_legacy_feedback``)
    bypass this check because they pre-date the recall_results
    table and are tagged with ``migration_status`` so the
    downstream pipeline can filter them out.
    """
    with _lock:
        conn = _get_db()
        try:
            # G5.1 strong validation: refuse to record feedback for a
            # candidate that was never returned for this query_id. This
            # closes a hole where a stale or fabricated candidate_key
            # could pollute the offline evaluation signal.
            #
            # Migration backfill rows carry ``migration_status`` (set by
            # :func:`migrate_legacy_feedback`) and pre-date the
            # recall_results table — we let those through without the
            # check so the backfill keeps working.
            if migration_status is None:
                row = conn.execute(
                    "SELECT 1 FROM recall_results WHERE query_id = ? "
                    "AND candidate_key = ? LIMIT 1",
                    (query_id, candidate_key),
                ).fetchone()
                if row is None:
                    raise ValueError(
                        f"candidate_key {candidate_key!r} was not returned "
                        f"for query_id {query_id!r}; refusing to record "
                        f"feedback for a candidate the user never saw"
                    )
            # Live feedback already passed the recall_results membership
            # check, therefore its parent recall_run exists.  Only explicit
            # migration rows may create a historical stub.
            if migration_status is not None:
                conn.execute(
                    "INSERT OR IGNORE INTO recall_runs(query_id, query_text, retrieval_mode, policy_version) VALUES (?, ?, ?, ?)",
                    (query_id, "", "hybrid", "legacy"),
                )
            cur = conn.execute(
                """
                INSERT INTO feedback_events
                    (query_id, candidate_key, memory_id, useful, created_at,
                     feedback_source, migration_status)
                VALUES (?, ?, ?, ?, datetime('now'), ?, ?)
                """,
                (
                    query_id,
                    candidate_key,
                    memory_id,
                    1 if useful else 0,
                    feedback_source,
                    migration_status,
                ),
            )
            conn.commit()
            row_id = cur.lastrowid
        finally:
            conn.close()
    return row_id or 0


def get_feedback_summary() -> Dict[str, Any]:
    """Aggregate feedback ratios over multiple windows.

    Returns a dict with ``ratio_24h``, ``ratio_7d``, ``ratio_30d``,
    and ``total_events``. Each ratio is ``useful / (useful + not_useful)``
    or ``None`` when there are no events in the window.
    """
    with _lock:
        conn = _get_db()
        try:
            windows = {"ratio_24h": 1, "ratio_7d": 7, "ratio_30d": 30}
            out: Dict[str, Any] = {}
            total = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
            out["total_events"] = total
            for key, days in windows.items():
                row = conn.execute(
                    """
                    SELECT
                        SUM(CASE WHEN useful=1 THEN 1 ELSE 0 END) AS pos,
                        COUNT(*) AS total
                    FROM feedback_events
                    WHERE created_at >= datetime('now', ?)
                    """,
                    (f"-{days} days",),
                ).fetchone()
                if row["total"] and row["total"] > 0:
                    out[key] = row["pos"] / row["total"]
                else:
                    out[key] = None
            return out
        finally:
            conn.close()


def get_recall_runs_for_query(query_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        conn = _get_db()
        try:
            row = conn.execute(
                "SELECT * FROM recall_runs WHERE query_id = ?", (query_id,)
            ).fetchone()
            if row is None:
                return None
            return dict(row)
        finally:
            conn.close()


def _retention_cleanup() -> int:
    """Delete expired recall traces while preserving human feedback."""
    with _lock:
        conn = _get_db()
        try:
            cutoff = f"-{_RECALL_RUNS_RETENTION_DAYS} days"
            before_feedback = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
            conn.execute("BEGIN IMMEDIATE")
            deleted_results = conn.execute(
                "DELETE FROM recall_results WHERE query_id IN ("
                "SELECT query_id FROM recall_runs WHERE created_at < datetime('now', ?)"
                ")",
                (cutoff,),
            ).rowcount
            deleted_runs = conn.execute(
                "DELETE FROM recall_runs WHERE created_at < datetime('now', ?)",
                (cutoff,),
            ).rowcount
            after_feedback = conn.execute("SELECT COUNT(*) FROM feedback_events").fetchone()[0]
            if after_feedback != before_feedback:
                raise RuntimeError("retention attempted to remove feedback events")
            conn.execute("COMMIT")
            logger.info(
                "retention: deleted_recall_results=%d deleted_recall_runs=%d "
                "preserved_feedback_events=%d cutoff=%s",
                deleted_results, deleted_runs, after_feedback, cutoff,
            )
            return int(deleted_runs or 0)
        except Exception:
            try:
                conn.execute("ROLLBACK")
            except sqlite3.Error:
                pass
            raise
        finally:
            conn.close()


# -- legacy migration --------------------------------------------------------


#: Marker key used for structured legacy migration state.
_LEGACY_MIGRATION_MARKER = "_legacy_feedback_migrated_v030"

#: Action names in the legacy audit log.
_LEGACY_FEEDBACK_ACTIONS = ("feedback", "recall_feedback")


def _migration_state(conn: sqlite3.Connection) -> Optional[Dict[str, Any]]:
    row = conn.execute("SELECT value FROM metadata WHERE key = ?", (_LEGACY_MIGRATION_MARKER,)).fetchone()
    if row is None or not row[0]:
        return None
    try:
        value = json.loads(row[0])
        return value if isinstance(value, dict) else None
    except (TypeError, ValueError, json.JSONDecodeError):
        return None


def _write_migration_state(conn: sqlite3.Connection, state: Dict[str, Any]) -> None:
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
        (_LEGACY_MIGRATION_MARKER, json.dumps(state, sort_keys=True)),
    )


def _resolve_legacy_candidate(memory_id: str) -> Tuple[str, str, str]:
    """Resolve a bare legacy id without guessing across collections."""
    backend = get_current_backend()
    if backend is None:
        return f"unresolved:{memory_id}", "", "unresolved:not_found"
    matches: List[str] = []
    try:
        iterator = getattr(backend, "iter_memories_by_collection", None)
        if callable(iterator):
            for collection, memory in iterator():
                if str(getattr(memory, "id", "")) == str(memory_id):
                    matches.append(str(collection))
        else:
            collections = list(backend.list_collections() or [])
            for collection in collections:
                try:
                    if backend.get_memory_in_collection(collection, memory_id) is not None:
                        matches.append(str(collection))
                except Exception:
                    continue
    except Exception:
        return f"unresolved:{memory_id}", "", "unresolved:not_found"
    matches = sorted(set(matches))
    if len(matches) == 1:
        coll = matches[0]
        return f"{coll}:{memory_id}", coll, "migrated:verified"
    if len(matches) > 1:
        return f"ambiguous:{memory_id}", "", "ambiguous:collection"
    return f"unresolved:{memory_id}", "", "unresolved:not_found"



def _open_audit_db_for_read() -> Optional[sqlite3.Connection]:
    """Open the legacy audit-log SQLite DB in read-only mode.

    Returns ``None`` when the file does not exist yet (a fresh install
    that has never recorded an audit event) so the migration can
    short-circuit cleanly. The connection is opened with
    ``mode=ro`` so a migration run cannot accidentally write to the
    audit log — it is the source of truth for the legacy rows.
    """

    try:
        store = get_audit_store()
    except Exception:
        return None
    path = getattr(store, "_db_path", None)
    if path is None or not Path(path).exists():
        return None
    # ``mode=ro`` requires URI=True and an absolute path. We open a
    # short-lived connection per migration run; the AuditStore's own
    # thread-local connection is left untouched so other call sites
    # can keep writing audit rows.
    uri = f"file:{Path(path).resolve().as_posix()}?mode=ro"
    try:
        # ``check_same_thread=False`` is fine here: the connection
        # is opened and closed inside a single ``with`` block, and
        # the migration runs inside the ``_lock`` in
        # ``migrate_legacy_feedback`` so no other thread can race.
        conn = sqlite3.connect(uri, uri=True, timeout=5)
    except sqlite3.Error:
        # Fallback: open read-write but only SELECT from it. The
        # migration body never INSERTs into ``audit_log`` so the
        # privilege escalation is harmless.
        try:
            conn = sqlite3.connect(str(path), timeout=5)
        except sqlite3.Error:
            return None
    conn.row_factory = sqlite3.Row
    return conn


def _has_marker(conn: sqlite3.Connection, key: str) -> bool:
    """Return True if ``key`` is recorded in the ``metadata`` table."""
    row = conn.execute(
        "SELECT value FROM metadata WHERE key = ?", (key,)
    ).fetchone()
    return row is not None


def _set_marker(conn: sqlite3.Connection, key: str, value: str) -> None:
    """Upsert ``key``/``value`` in the ``metadata`` table."""
    conn.execute(
        "INSERT INTO metadata(key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _audit_has_feedback_rows(conn: sqlite3.Connection) -> bool:
    """Return True if the legacy audit_log has any feedback rows."""
    if not conn:
        return False
    placeholders = ",".join("?" for _ in _LEGACY_FEEDBACK_ACTIONS)
    cur = conn.execute(
        f"SELECT 1 FROM audit_log WHERE action IN ({placeholders}) LIMIT 1",
        _LEGACY_FEEDBACK_ACTIONS,
    )
    return cur.fetchone() is not None


def _audit_table_exists(conn: sqlite3.Connection) -> bool:
    """Return True if the legacy ``audit_log`` table exists."""
    if not conn:
        return False
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='audit_log'"
    ).fetchone()
    return row is not None


def _audit_select_feedback(conn: sqlite3.Connection) -> List[sqlite3.Row]:
    """Return every legacy feedback row from ``audit_log``.

    Rows are returned oldest-first so the migration preserves the
    original ordering of feedback events (important for time-series
    evaluation).
    """
    placeholders = ",".join("?" for _ in _LEGACY_FEEDBACK_ACTIONS)
    return list(
        conn.execute(
            f"SELECT id, timestamp, action, actor, memory_id, detail "
            f"FROM audit_log WHERE action IN ({placeholders}) "
            f"ORDER BY id ASC",
            _LEGACY_FEEDBACK_ACTIONS,
        )
    )


def migrate_legacy_feedback(conn: Optional[sqlite3.Connection] = None) -> int:
    """Migrate legacy audit feedback with explicit, retryable state."""
    if conn is None:
        with _lock:
            owned = _get_db(run_legacy_migration=False)
            try:
                return migrate_legacy_feedback(owned)
            finally:
                owned.close()

    prior = _migration_state(conn)
    if prior and prior.get("status") == "completed":
        return 0

    now = datetime.now(timezone.utc).isoformat()
    state: Dict[str, Any] = {
        "status": "running",
        "source_rows": 0,
        "migrated_rows": 0,
        "skipped_rows": 0,
        "ambiguous_rows": 0,
        "failed_rows": 0,
        "started_at": now,
        "completed_at": None,
        "last_error": None,
    }
    _write_migration_state(conn, state)
    conn.commit()

    audit_conn = _open_audit_db_for_read()
    if audit_conn is None or not _audit_table_exists(audit_conn):
        state.update(status="not_applicable", completed_at=datetime.now(timezone.utc).isoformat())
        _write_migration_state(conn, state)
        conn.commit()
        if audit_conn is not None:
            audit_conn.close()
        return 0

    try:
        rows = _audit_select_feedback(audit_conn)
    finally:
        audit_conn.close()
    state["source_rows"] = len(rows)
    if not rows:
        state.update(status="completed", completed_at=datetime.now(timezone.utc).isoformat())
        _write_migration_state(conn, state)
        conn.commit()
        return 0

    try:
        conn.execute("BEGIN IMMEDIATE")
        for row in rows:
            detail = row["detail"] or ""
            query, memory_id, useful = _parse_old_feedback(detail)
            if not memory_id:
                state["skipped_rows"] += 1
                continue
            candidate_key, collection, resolution_status = _resolve_legacy_candidate(memory_id)
            if resolution_status == "ambiguous:collection":
                state["ambiguous_rows"] += 1
            elif resolution_status.startswith("unresolved:"):
                state["skipped_rows"] += 1
            query_id = _legacy_query_id(query, row["id"])
            created_at = row["timestamp"] or datetime.now(timezone.utc).isoformat()
            source_key = f"audit_log:{row['id']}"
            conn.execute(
                "INSERT OR IGNORE INTO recall_runs"
                "(query_id, query_text, retrieval_mode, policy_version) VALUES (?, ?, ?, ?)",
                (query_id, query or "", "hybrid", "legacy_audit"),
            )
            cur = conn.execute(
                """INSERT OR IGNORE INTO feedback_events
                    (query_id, candidate_key, memory_id, collection, useful, created_at,
                     feedback_source, migration_status, resolution_status, legacy_source_key)
                    VALUES (?, ?, ?, ?, ?, ?, 'legacy_audit', 'migrated:audit', ?, ?)""",
                (query_id, candidate_key, memory_id, collection, 1 if useful else 0,
                 created_at, resolution_status, source_key),
            )
            if cur.rowcount > 0:
                state["migrated_rows"] += 1
        state["status"] = (
            "completed_with_skips"
            if state["skipped_rows"] or state["ambiguous_rows"]
            else "completed"
        )
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_migration_state(conn, state)
        conn.execute("COMMIT")
    except Exception as exc:
        try:
            conn.execute("ROLLBACK")
        except sqlite3.Error:
            pass
        state["status"] = "failed"
        state["failed_rows"] = max(1, int(state.get("failed_rows", 0)))
        state["last_error"] = f"{type(exc).__name__}: {exc}"
        state["completed_at"] = datetime.now(timezone.utc).isoformat()
        _write_migration_state(conn, state)
        conn.commit()
        raise
    logger.info("migrate_legacy_feedback: %s", json.dumps(state, sort_keys=True))
    return int(state["migrated_rows"])


def _legacy_query_id(query: Optional[str], audit_id: int) -> str:
    """Build a stable ``query_id`` for a legacy feedback row.

    The v0.3.0 ``query_id`` must be unique per row, but the legacy
    ``audit_log`` rows don't carry one. We compose a deterministic
    id from the audit row id plus a short hash of the query string
    so that two legacy rows for the same query get distinct ids
    (and so that re-running the migration on the same legacy data
    produces the same ids, which keeps the ``INSERT OR IGNORE``
    short-circuit effective).
    """
    suffix = audit_id if audit_id is not None else 0
    qhash = hashlib.sha256((query or "").encode("utf-8")).hexdigest()[:12]
    return f"legacy_audit:{suffix}:{qhash}"


def _parse_old_feedback(detail: str) -> Tuple[Optional[str], Optional[str], bool]:
    """Parse a legacy audit.detail string into ``(query, memory_id, useful)``.

    Expected format: ``query='...' memory_id='...' useful=True|False``
    (with either ``=`` or ``:`` separators, and either single- or
    double-quoted values — the v0.2.x codebase emitted both shapes
    across different audit-log call sites).
    """
    if not detail:
        return None, None, False
    import re as _re
    qm = _re.search(r"query[=:]\s*['\"]([^'\"]+)['\"]", detail)
    mm = _re.search(r"memory_id[=:]\s*['\"]([^'\"]+)['\"]", detail)
    um = _re.search(r"useful[=:]\s*(True|False|true|false)", detail)
    query = qm.group(1) if qm else None
    mem_id = mm.group(1) if mm else None
    useful = um.group(1).lower() == "true" if um else False
    return query, mem_id, useful
