"""Tests for G5.1 strong candidate-key validation in
``recall_feedback.record_feedback_v030``.

The contract under test is:

1. ``record_feedback_v030(query_id, candidate_key, ...)`` MUST
   verify that ``candidate_key`` actually appears in the
   ``recall_results`` rows for that ``query_id``. Otherwise
   feedback can target a candidate the user never saw.
2. Migration backfill rows (``migration_status`` set) bypass the
   check because they pre-date the ``recall_results`` table.
3. The ``/api/feedback`` endpoint MUST surface the failure to
   the client as HTTP 422 with a clear message.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from openclaw_memory_os.app import create_app
from openclaw_memory_os.recall_feedback import (
    _get_db,
    record_feedback_v030,
    record_recall_result,
    record_recall_run,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _isolate_db(tmp_path: Path, monkeypatch):
    """Point recall_feedback at a temp directory so tests don't
    touch the real DB. Mirrors the pattern used in
    ``test_feedback_schema.py``."""
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path))
    import openclaw_memory_os.recall_feedback as rf
    rf._RECALL_DB_DIR = tmp_path / "openclaw-memory-os"
    rf._RECALL_DB = rf._RECALL_DB_DIR / "recall_feedback.db"
    yield
    if rf._RECALL_DB.exists():
        rf._RECALL_DB.unlink()


def _client():
    """Build a FastAPI TestClient."""
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Core contract tests
# ---------------------------------------------------------------------------


def test_record_feedback_rejects_unknown_candidate_key():
    """record_feedback_v030 must raise ValueError when candidate_key
    does not appear in recall_results for this query_id.

    Setup: recall_runs(query_id='q1') with NO recall_results rows.
    Action: try to record feedback for (query_id='q1', candidate_key='c:X').
    Expectation: ValueError; no row inserted into feedback_events.
    """
    qid = record_recall_run("q1", "how to configure nginx")
    # Defensive: confirm no recall_results row exists.
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM recall_results WHERE query_id = ?", (qid,)
        ).fetchall()
        assert rows == []
    finally:
        conn.close()

    with pytest.raises(ValueError) as excinfo:
        record_feedback_v030(qid, "c:mem-X", True)

    msg = str(excinfo.value)
    assert "c:mem-X" in msg
    assert qid in msg
    assert "not returned" in msg.lower() or "never saw" in msg.lower()

    # No feedback row should have been inserted.
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM feedback_events WHERE query_id = ?", (qid,)
        ).fetchall()
        assert rows == []
    finally:
        conn.close()


def test_record_feedback_accepts_returned_candidate():
    """record_feedback_v030 must succeed when candidate_key matches a
    recall_results row for the same query_id."""
    qid = record_recall_run("q-accept", "how to deploy the OS")
    record_recall_result(
        qid, "c:mem-1",
        memory_id="mem-1",
        collection="c",
        rank=1,
        status="active",
        final_score=0.9,
    )

    row_id = record_feedback_v030(qid, "c:mem-1", True, memory_id="mem-1")
    assert row_id > 0

    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT * FROM feedback_events WHERE id = ?", (row_id,)
        ).fetchone()
        assert row is not None
        assert row["query_id"] == qid
        assert row["candidate_key"] == "c:mem-1"
        assert row["useful"] == 1
        assert row["memory_id"] == "mem-1"
    finally:
        conn.close()


def test_record_feedback_accepts_legacy_query_with_no_results():
    """Edge case: query_id exists in recall_runs but no recall_results
    rows were ever inserted (rare, but possible for legacy CLI /
    older dashboards). The brief is explicit: *fail closed* — the
    insert must be rejected with ValueError because the user never
    saw any candidate for this query.
    """
    qid = record_recall_run("q-legacy-no-results", "legacy query without hits")

    # Defensive: confirm no recall_results row exists for this qid.
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM recall_results WHERE query_id = ?", (qid,)
        ).fetchall()
        assert rows == []
    finally:
        conn.close()

    with pytest.raises(ValueError) as excinfo:
        record_feedback_v030(qid, "c:anything", True)

    # Clear "we don't have a record of returning this candidate" message.
    msg = str(excinfo.value).lower()
    assert "not returned" in msg or "never saw" in msg

    # No feedback row should have been inserted (fail-closed contract).
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM feedback_events WHERE query_id = ?", (qid,)
        ).fetchall()
        assert rows == []
    finally:
        conn.close()


def test_record_feedback_accepts_migrated_rows_without_results_check():
    """Migration backfill rows (migration_status set) bypass the
    candidate-key check, because they pre-date the recall_results
    table and the downstream pipeline filters them by status."""
    # Don't insert recall_run or recall_result rows first; the
    # migration backfill path inserts a minimal recall_runs stub
    # automatically and bypasses the recall_results check.
    row_id = record_feedback_v030(
        query_id="legacy-q-no-run",
        candidate_key="c:legacy-mem",
        useful=True,
        migration_status="migrated:audit",
    )
    assert row_id > 0
    conn = _get_db()
    try:
        row = conn.execute(
            "SELECT migration_status, candidate_key FROM feedback_events WHERE id = ?",
            (row_id,),
        ).fetchone()
        assert row["migration_status"] == "migrated:audit"
        assert row["candidate_key"] == "c:legacy-mem"
    finally:
        conn.close()


def test_record_feedback_rejects_candidate_for_unrelated_query():
    """The validation must scope by query_id: a candidate_key that
    was returned for query_id=A must NOT be accepted as feedback
    for query_id=B."""
    qid_a = record_recall_run("q-a", "query A")
    qid_b = record_recall_run("q-b", "query B")
    # candidate_key 'c:shared' is returned for A only.
    record_recall_result(
        qid_a, "c:shared",
        memory_id="shared",
        collection="c",
        rank=1,
        status="active",
        final_score=0.5,
    )

    # Feedback for (A, c:shared) is fine.
    row_id_a = record_feedback_v030(qid_a, "c:shared", True)
    assert row_id_a > 0

    # Feedback for (B, c:shared) must fail: c:shared was NOT returned
    # for query B.
    with pytest.raises(ValueError):
        record_feedback_v030(qid_b, "c:shared", False)


# ---------------------------------------------------------------------------
# HTTP endpoint test
# ---------------------------------------------------------------------------


def test_app_feedback_endpoint_returns_422_on_unknown_candidate(tmp_path: Path):
    """The /api/feedback endpoint must translate ValueError to HTTP 422
    with a clear message when the candidate_key was not returned for
    the query_id."""
    # Seed a recall_runs row so the structured_feedback path is taken,
    # but DO NOT seed a recall_results row matching the candidate_key.
    qid = record_recall_run("q-http-422", "http test query")
    # Sanity: no recall_results row exists yet for this qid.
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM recall_results WHERE query_id = ?", (qid,)
        ).fetchall()
        assert rows == []
    finally:
        conn.close()

    with _client() as c:
        r = c.post(
            "/api/feedback",
            json={
                "query_id": qid,
                "candidate_key": "c:not-returned",
                "useful": True,
            },
        )

    assert r.status_code == 422, r.text
    body = r.json()
    # FastAPI surfaces HTTPException detail under "detail".
    detail = body.get("detail") or ""
    assert "c:not-returned" in detail
    assert qid in detail
    assert "not returned" in detail.lower() or "never saw" in detail.lower()

    # And no feedback row should have been inserted (fail-closed).
    conn = _get_db()
    try:
        rows = conn.execute(
            "SELECT * FROM feedback_events WHERE query_id = ?", (qid,)
        ).fetchall()
        assert rows == []
    finally:
        conn.close()


def test_app_feedback_endpoint_accepts_returned_candidate(tmp_path: Path):
    """Sanity: when the candidate_key IS in recall_results, the endpoint
    must accept the feedback (200 OK, row inserted)."""
    qid = record_recall_run("q-http-ok", "http ok query")
    record_recall_result(
        qid, "c:mem-1",
        memory_id="mem-1",
        collection="c",
        rank=1,
        status="active",
        final_score=0.9,
    )

    with _client() as c:
        r = c.post(
            "/api/feedback",
            json={
                "query_id": qid,
                "candidate_key": "c:mem-1",
                "useful": True,
                "memory_id": "mem-1",
            },
        )

    assert r.status_code == 200, r.text
    body = r.json()
    assert body["status"] == "ok"
    assert isinstance(body["row_id"], int) and body["row_id"] > 0