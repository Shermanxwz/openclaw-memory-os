"""Tests for recall query_id ↔ feedback association (S6).

Verifies that:
1. A recall-test response carries a non-empty query_id.
2. Structured feedback can be linked back to the recall run via query_id.
3. The recall run stores query_text (not just a hash).
4. recall_runs older than 180 days are cleaned up.
5. Legacy audit_log entries remain read-compatible.
"""

from __future__ import annotations

from pathlib import Path
from tempfile import TemporaryDirectory

import pytest

from openclaw_memory_os.audit import AuditStore
from openclaw_memory_os.models import FeedbackEntry, RecallResponse
from openclaw_memory_os.recall_feedback import (
    _RECALL_RUNS_RETENTION_DAYS,
    _get_db,
    _retention_cleanup,
    get_feedback_summary,
    get_recall_runs_for_query,
    record_feedback_v030,
    record_recall_result,
    record_recall_run,
)


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch):
    """Point recall_feedback at a temp directory so tests don't
    touch the real DB."""
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path))
    import openclaw_memory_os.recall_feedback as rf
    rf._RECALL_DB_DIR = tmp_path / "openclaw-memory-os"
    rf._RECALL_DB = rf._RECALL_DB_DIR / "recall_feedback.db"
    yield
    if rf._RECALL_DB.exists():
        rf._RECALL_DB.unlink()


def test_recall_response_has_query_id():
    """RecallResponse model must carry a query_id field."""
    resp = RecallResponse(
        query="test",
        mode="hybrid",
        took_ms=10.0,
        backend="sample",
        total_considered=5,
        hits=[],
        query_id="abc-123",
        policy_version="v1",
    )
    assert resp.query_id == "abc-123"
    assert resp.policy_version == "v1"


def test_recall_response_diagnostics_default_empty():
    resp = RecallResponse(
        query="test",
        mode="hybrid",
        took_ms=10.0,
        backend="sample",
        total_considered=5,
        hits=[],
    )
    assert resp.diagnostics == {}


def test_query_id_links_feedback_to_recall_run():
    """Feedback recorded with a query_id can be traced back to the recall run."""
    qid = record_recall_run(
        "link-test-1",
        "how to set MEMORY_OS_TOKEN",
        retrieval_mode="hybrid",
        policy_version="v1",
    )
    record_recall_result(
        qid, "sample:mem-1",
        memory_id="mem-1",
        collection="sample",
        rank=1,
        status="active",
        final_score=0.9,
    )
    record_feedback_v030(qid, "sample:mem-1", True, memory_id="mem-1")

    # Verify the recall run exists
    run = get_recall_runs_for_query(qid)
    assert run is not None
    assert run["query_text"] == "how to set MEMORY_OS_TOKEN"

    # Verify feedback is linked
    conn = _get_db()
    try:
        fb = conn.execute(
            "SELECT * FROM feedback_events WHERE query_id = ?", (qid,)
        ).fetchall()
        assert len(fb) == 1
        assert fb[0]["candidate_key"] == "sample:mem-1"
        assert fb[0]["useful"] == 1
    finally:
        conn.close()


def test_query_text_stored_not_hashed():
    """query_text must be stored in full, not just as a hash."""
    qid = record_recall_run(
        "text-test-1",
        "detailed query about nginx reverse proxy configuration",
    )
    run = get_recall_runs_for_query(qid)
    assert run is not None
    # The full text must be present
    assert "detailed query about nginx reverse proxy" in run["query_text"]
    # A hash is also stored but the text is the primary field
    assert run.get("query_hash") is not None


def test_recall_runs_180_day_cleanup():
    """_retention_cleanup must target runs older than 180 days."""
    assert _RECALL_RUNS_RETENTION_DAYS == 180
    # Insert a run and verify cleanup doesn't remove recent data
    qid = record_recall_run("recent-1", "recent query")
    removed = _retention_cleanup()
    assert removed == 0  # recent data should not be removed
    # The run should still exist
    run = get_recall_runs_for_query(qid)
    assert run is not None


def test_legacy_audit_log_read_compatible():
    """Old audit_log entries with action='feedback' must still be readable."""
    with TemporaryDirectory() as tmpdir:
        db_path = Path(tmpdir) / "audit_test.sqlite"
        store = AuditStore(db_path=db_path)
        # Write a legacy feedback entry
        row_id = store.log(
            "feedback",
            actor="test-user",
            memory_id="mem-001",
            detail="query='how to set token' useful=True",
        )
        assert row_id > 0
        # Read it back
        entries = store.list_recent(action="feedback")
        assert len(entries) >= 1
        found = False
        for e in entries:
            if e.memory_id == "mem-001":
                found = True
                assert "feedback" in e.action
        assert found
        store.close()


def test_feedback_entry_model_v030_fields():
    """FeedbackEntry must accept query_id and candidate_key."""
    entry = FeedbackEntry(
        query_id="qid-1",
        candidate_key="col:mem-1",
        useful=True,
    )
    assert entry.query_id == "qid-1"
    assert entry.candidate_key == "col:mem-1"
    assert entry.useful is True


def test_feedback_entry_model_legacy_fields():
    """FeedbackEntry must still accept legacy memory_id and query."""
    entry = FeedbackEntry(
        memory_id="mem-1",
        query="test query",
        useful=False,
    )
    assert entry.memory_id == "mem-1"
    assert entry.query == "test query"


def test_multiple_feedback_events_same_query():
    """Multiple feedback events can be recorded for the same query_id."""
    qid = record_recall_run("multi-fb-1", "multi feedback test")
    # G5.1 strong validation: each candidate_key must be in recall_results.
    record_recall_result(qid, "c:mem-1", rank=1, status="active")
    record_recall_result(qid, "c:mem-2", rank=2, status="active")
    record_recall_result(qid, "c:mem-3", rank=3, status="active")
    record_feedback_v030(qid, "c:mem-1", True, memory_id="mem-1")
    record_feedback_v030(qid, "c:mem-2", False, memory_id="mem-2")
    record_feedback_v030(qid, "c:mem-3", True, memory_id="mem-3")

    summary = get_feedback_summary()
    assert summary["total_events"] == 3
    # 2 useful out of 3
    assert summary["ratio_24h"] == pytest.approx(2 / 3, abs=0.01)


def test_recall_result_stores_candidate_key():
    """recall_results must store candidate_key for feedback linkage."""
    qid = record_recall_run("ck-test-1", "candidate key test")
    record_recall_result(
        qid, "openclaw_memories:mem-42",
        memory_id="mem-42",
        collection="openclaw_memories",
        rank=1,
        status="active",
        final_score=0.88,
    )
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT candidate_key, collection FROM recall_results WHERE query_id = ?",
            (qid,),
        ).fetchone()
        assert row is not None
        assert row["candidate_key"] == "openclaw_memories:mem-42"
        assert row["collection"] == "openclaw_memories"
    finally:
        conn.close()
