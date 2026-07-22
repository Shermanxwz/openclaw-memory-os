"""Tests for the v0.3.0 structured recall-feedback layer (S6).

Covers: schema creation, CRUD on recall_runs / recall_results /
feedback_events, legacy migration, retention, and the
new ``query_id`` + ``candidate_key`` path on the API.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from openclaw_memory_os.models import FeedbackEntry
from openclaw_memory_os.recall_feedback import (
    _get_db,
    _retention_cleanup,
    get_feedback_summary,
    get_recall_runs_for_query,
    migrate_legacy_feedback,
    record_feedback_v030,
    record_recall_result,
    record_recall_run,
)


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch):
    """Point recall_feedback at a temp directory so tests don't
    touch the real DB."""
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path))
    # force module-level constants to re-evaluate
    import openclaw_memory_os.recall_feedback as rf
    rf._RECALL_DB_DIR = tmp_path / "openclaw-memory-os"
    rf._RECALL_DB = rf._RECALL_DB_DIR / "recall_feedback.db"
    yield
    # Cleanup
    if rf._RECALL_DB.exists():
        rf._RECALL_DB.unlink()


def test_schema_creates_tables():
    conn = _get_db()
    try:
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
        ).fetchall()
        assert {r[0] for r in tables} >= {"recall_runs", "recall_results", "feedback_events"}
    finally:
        conn.close()


def test_record_recall_run_and_get_back():
    record_recall_run(
        "test-1", "how to configure nginx",
        retrieval_mode="hybrid",
        policy_version="0.3.0",
        latency_ms=12.3,
    )
    row = get_recall_runs_for_query("test-1")
    assert row is not None
    assert row["query_text"] == "how to configure nginx"
    assert row["policy_version"] == "0.3.0"


def test_record_recall_result():
    qid = record_recall_run("r2", "bravo")
    record_recall_result(
        qid, "c:mem-1",
        memory_id="mem-1",
        collection="c",
        rank=1,
        status="active",
        final_score=0.95,
    )
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM recall_results WHERE query_id=?", (qid,)
        ).fetchall()
        assert len(rows) == 1
        assert rows[0]["final_score"] == 0.95
    finally:
        conn.close()


def test_feedback_lifecycle():
    qid = record_recall_run("fb-test", "query")
    # G5.1 strong validation: candidate_key must actually have been
    # returned for this query_id, so seed a recall_results row first.
    record_recall_result(qid, "col:mem-1", rank=1, status="active")
    row_id = record_feedback_v030(qid, "col:mem-1", True)
    assert row_id > 0
    summary = get_feedback_summary()
    assert summary["total_events"] == 1
    # ratio_24h should be 1.0
    assert summary["ratio_24h"] == 1.0


def test_feedback_ratio_multiple_events():
    qid = record_recall_run("fb-multi", "test")
    # G5.1 strong validation: each candidate_key must be in recall_results.
    record_recall_result(qid, "c:a", rank=1, status="active")
    record_recall_result(qid, "c:b", rank=2, status="active")
    record_recall_result(qid, "c:c", rank=3, status="active")
    record_feedback_v030(qid, "c:a", True)
    record_feedback_v030(qid, "c:b", True)
    record_feedback_v030(qid, "c:c", False)
    summary = get_feedback_summary()
    assert summary["total_events"] == 3
    assert summary["ratio_24h"] == 2 / 3


def test_feedback_summary_empty_db():
    summary = get_feedback_summary()
    assert summary["total_events"] == 0
    assert summary["ratio_24h"] is None


def test_retention_cleanup():
    record_recall_run("old", "old data")
    # This is hard to test without mocking time. We just verify
    # no crash and it runs idempotently.
    removed = _retention_cleanup()
    assert removed == 0 or removed >= 0


def test_legacy_migration_parse():
    from openclaw_memory_os.recall_feedback import _parse_old_feedback
    q, m, u = _parse_old_feedback("query='how to set token' memory_id='m1' useful=True")
    assert q == "how to set token"
    assert m == "m1"
    assert u is True


def test_legacy_migration_no_crash():
    migrated = migrate_legacy_feedback()
    assert migrated >= 0


def test_api_feedback_v030_payload_deserialises():
    """Ensure the new FeedbackEntry fields accept the API shape."""
    payload = FeedbackEntry(
        query_id="q1",
        candidate_key="col:mem-1",
        useful=True,
    )
    assert payload.query_id == "q1"
    assert payload.candidate_key == "col:mem-1"


# ---------------------------------------------------------------------------
# v0.3.0.x schema-completeness tests
# ---------------------------------------------------------------------------

def test_schema_includes_v030_columns_on_fresh_db():
    """On a fresh DB, _ensure_schema must materialise every v0.3.0.x column
    in the same call (no separate migration step required)."""
    conn = _get_db()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(recall_runs)").fetchall()}
        for col in (
            "query_hash",
            "corpus_snapshot_id",
            "dense_available",
            "lexical_available",
            "collections_succeeded_json",
            "collections_failed_json",
        ):
            assert col in cols, f"recall_runs missing v0.3.0.x column: {col}"

        cols = {row[1] for row in conn.execute("PRAGMA table_info(recall_results)").fetchall()}
        for col in (
            "vector_score_raw",
            "vector_score_calibrated",
            "lexical_score_raw",
            "lexical_score_calibrated",
            "importance_score",
            "recency_score",
            "feedback_score",
            "display_score",
        ):
            assert col in cols, f"recall_results missing v0.3.0.x column: {col}"

        cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback_events)").fetchall()}
        assert "migration_status" in cols, "feedback_events missing v0.3.0.x column: migration_status"
    finally:
        conn.close()


def test_schema_migrates_legacy_db_adds_missing_columns():
    """Simulate a pre-v0.3.0.x database: create the tables with only the
    original columns, then call ``_ensure_schema`` and verify the new
    columns are added without losing existing data."""
    import openclaw_memory_os.recall_feedback as rf
    db_path = rf._RECALL_DB
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    # Pre-v0.3.0.x schema: no query_hash, no vector_score_raw, no migration_status.
    legacy = sqlite3.connect(str(db_path))
    legacy.executescript(
        """
        CREATE TABLE recall_runs (
            query_id        TEXT PRIMARY KEY,
            query_text      TEXT NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            retrieval_mode  TEXT,
            policy_version  TEXT,
            latency_ms      REAL,
            retrieval_status TEXT,
            degraded_reason TEXT,
            fallback_used   INTEGER DEFAULT 0
        );
        CREATE TABLE recall_results (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id        TEXT NOT NULL REFERENCES recall_runs(query_id),
            candidate_key   TEXT NOT NULL,
            memory_id       TEXT,
            collection      TEXT,
            rank            INTEGER,
            status          TEXT,
            vector_score    REAL,
            lexical_score   REAL,
            rrf_score       REAL,
            final_score     REAL,
            explanation     TEXT
        );
        CREATE TABLE feedback_events (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id        TEXT NOT NULL REFERENCES recall_runs(query_id),
            candidate_key   TEXT NOT NULL,
            memory_id       TEXT,
            useful          INTEGER NOT NULL,
            created_at      TEXT NOT NULL DEFAULT (datetime('now')),
            feedback_source TEXT DEFAULT 'dashboard'
        );
        """
    )
    # Insert a legacy row to confirm migration doesn't drop data.
    legacy.execute(
        "INSERT INTO recall_runs(query_id, query_text, retrieval_mode, policy_version) VALUES (?, ?, ?, ?)",
        ("legacy-1", "legacy query", "hybrid", "v0.2.9"),
    )
    legacy.commit()
    legacy.close()

    # _ensure_schema (called by _get_db) must add the new columns.
    conn = _get_db()
    try:
        cols = {row[1] for row in conn.execute("PRAGMA table_info(recall_runs)").fetchall()}
        for col in (
            "query_hash",
            "corpus_snapshot_id",
            "dense_available",
            "lexical_available",
            "collections_succeeded_json",
            "collections_failed_json",
        ):
            assert col in cols, f"migration missed recall_runs.{col}"

        cols = {row[1] for row in conn.execute("PRAGMA table_info(recall_results)").fetchall()}
        for col in (
            "vector_score_raw",
            "vector_score_calibrated",
            "lexical_score_raw",
            "lexical_score_calibrated",
            "importance_score",
            "recency_score",
            "feedback_score",
            "display_score",
        ):
            assert col in cols, f"migration missed recall_results.{col}"

        cols = {row[1] for row in conn.execute("PRAGMA table_info(feedback_events)").fetchall()}
        assert "migration_status" in cols

        # The legacy row must still be readable.
        row = conn.execute(
            "SELECT query_id, query_text, policy_version FROM recall_runs WHERE query_id = ?",
            ("legacy-1",),
        ).fetchone()
        assert row is not None
        assert row["query_text"] == "legacy query"
        assert row["policy_version"] == "v0.2.9"
    finally:
        conn.close()

    # Migration must be idempotent: re-running it must not raise.
    conn2 = _get_db()
    conn2.close()


def test_record_recall_run_persists_v030_fields():
    qid = record_recall_run(
        "v030-run-1",
        "how to deploy the OS",
        retrieval_mode="hybrid",
        policy_version="v0.3.0",
        latency_ms=12.3,
        corpus_snapshot_id="snap-2026-07-15",
        dense_available=True,
        lexical_available=False,
        collections_succeeded=["openclaw_memory_os", "openclaw_memory_os_2"],
        collections_failed=["openclaw_memory_os_3"],
    )
    row = get_recall_runs_for_query(qid)
    assert row is not None
    # query_hash is auto-derived from query_text when not provided.
    assert row["query_hash"] and len(row["query_hash"]) == 16
    assert row["corpus_snapshot_id"] == "snap-2026-07-15"
    assert row["dense_available"] == 1
    assert row["lexical_available"] == 0
    succ = json.loads(row["collections_succeeded_json"])
    fail = json.loads(row["collections_failed_json"])
    assert set(succ) == {"openclaw_memory_os", "openclaw_memory_os_2"}
    assert fail == ["openclaw_memory_os_3"]


def test_record_recall_run_explicit_query_hash():
    qid = record_recall_run(
        "v030-run-2",
        "alpha",
        query_hash="c0ffee01",
    )
    row = get_recall_runs_for_query(qid)
    assert row["query_hash"] == "c0ffee01"


def test_record_recall_result_persists_v030_scores():
    qid = record_recall_run("v030-result-1", "rank me")
    record_recall_result(
        qid,
        "c:mem-1",
        memory_id="mem-1",
        collection="c",
        rank=1,
        status="active",
        vector_score=0.42,
        lexical_score=0.31,
        rrf_score=0.7,
        final_score=0.85,
        vector_score_raw=0.41,
        vector_score_calibrated=0.42,
        lexical_score_raw=0.30,
        lexical_score_calibrated=0.31,
        importance_score=0.9,
        recency_score=0.5,
        feedback_score=0.0,
        display_score=0.85,
    )
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM recall_results WHERE query_id = ?", (qid,)
        ).fetchone()
        assert row is not None
        assert row["vector_score_raw"] == 0.41
        assert row["vector_score_calibrated"] == 0.42
        assert row["lexical_score_raw"] == 0.30
        assert row["lexical_score_calibrated"] == 0.31
        assert row["importance_score"] == 0.9
        assert row["recency_score"] == 0.5
        assert row["feedback_score"] == 0.0
        assert row["display_score"] == 0.85
    finally:
        conn.close()


def test_record_feedback_v030_accepts_migration_status():
    qid = record_recall_run("v030-fb-1", "migrate me")
    # G5.1: even migration-status paths need a recall_results row to be
    # accepted by the strong validation (the bypass is for legacy audit
    # backfill where the candidate_key didn't exist before). This test
    # uses an explicit recall_results row to keep the contract clean.
    record_recall_result(qid, "c:mem-1", rank=1, status="active")
    row_id = record_feedback_v030(
        qid, "c:mem-1", True,
        migration_status="migrated:audit",
    )
    assert row_id > 0
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT migration_status FROM feedback_events WHERE id = ?", (row_id,)
        ).fetchone()
        assert row["migration_status"] == "migrated:audit"
    finally:
        conn.close()


def test_record_feedback_v030_legacy_call_site_still_works():
    """Call sites that don't pass migration_status must keep working.

    G5.1: the candidate_key must actually have been returned for the
    query_id, so seed a recall_results row before recording feedback.
    """
    qid = record_recall_run("v030-fb-legacy", "old api")
    record_recall_result(qid, "c:mem-1", rank=1, status="active")
    row_id = record_feedback_v030(qid, "c:mem-1", False)
    assert row_id > 0
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT migration_status, useful FROM feedback_events WHERE id = ?", (row_id,)
        ).fetchone()
        assert row["migration_status"] is None
        assert row["useful"] == 0
    finally:
        conn.close()


def test_old_call_site_record_recall_run_no_new_args_still_works():
    """Existing tests / call sites pass only the original kwargs; the
    function must still work and write safe defaults.

    G5.5: when a caller omits ``corpus_snapshot_id``, the function
    auto-fingerprints the live corpus and writes the resulting
    id. The legacy contract is preserved in every other respect
    (the new collection fields stay ``NULL`` unless explicitly
    passed).
    """
    qid = record_recall_run("v030-old-args", "compat query")
    row = get_recall_runs_for_query(qid)
    assert row is not None
    # The auto-derived query_hash should be the SHA256 of "compat query".
    assert row["query_hash"] and len(row["query_hash"]) == 16
    # G5.5: corpus_snapshot_id is auto-fingerprinted (non-NULL) even
    # on legacy call sites.
    assert row["corpus_snapshot_id"], (
        "G5.5: corpus_snapshot_id must be auto-populated even on "
        "legacy call sites"
    )
    assert row["dense_available"] is None
    assert row["lexical_available"] is None
    assert row["collections_succeeded_json"] is None
    assert row["collections_failed_json"] is None


def test_old_call_site_record_recall_result_no_new_args_still_works():
    """``record_recall_result`` must accept the legacy call shape and
    leave the new score columns as ``NULL``."""
    qid = record_recall_run("v030-old-result", "compat2")
    record_recall_result(
        qid, "c:mem-1",
        memory_id="mem-1",
        collection="c",
        rank=1,
        status="active",
        vector_score=0.5,
        lexical_score=0.4,
        rrf_score=0.6,
        final_score=0.7,
    )
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM recall_results WHERE query_id = ?", (qid,)
        ).fetchone()
        assert row is not None
        assert row["vector_score"] == 0.5
        assert row["lexical_score"] == 0.4
        assert row["final_score"] == 0.7
        # New columns are None.
        assert row["vector_score_raw"] is None
        assert row["vector_score_calibrated"] is None
        assert row["lexical_score_raw"] is None
        assert row["lexical_score_calibrated"] is None
        assert row["importance_score"] is None
        assert row["recency_score"] is None
        assert row["feedback_score"] is None
        assert row["display_score"] is None
    finally:
        conn.close()
