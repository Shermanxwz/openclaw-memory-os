"""Tests for the feedback learning loop (issue #3).

Covers:

1. ``scripts/replay_feedback.py`` — audit log replay and weight aggregation.
2. Atomics write of weight snapshot with 0600 permissions.
3. ``rank_memories`` with ``feedback_weights`` parameter adjusts
   ``importance_boost``.
4. Missing audit log → graceful empty weights.
5. ``/api/feedback-summary`` endpoint returns correct ratios.
"""

from __future__ import annotations

import json
import os
import sqlite3
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from openclaw_memory_os.models import Memory, MemoryStatus, MemoryTier, RecallRequest
from openclaw_memory_os.config import Settings


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def mem_core() -> Memory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Memory(
        id="core-1",
        text="important core memory",
        tier=MemoryTier.CORE,
        importance=0.9,
        status=MemoryStatus.ACTIVE,
        tags=[],
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def mem_long() -> Memory:
    now = datetime(2026, 1, 1, tzinfo=timezone.utc)
    return Memory(
        id="long-1",
        text="less important long memory",
        tier=MemoryTier.LONG,
        importance=0.3,
        status=MemoryStatus.ACTIVE,
        tags=[],
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def settings() -> Settings:
    return Settings(importance_boost_scale=0.6)


def _build_audit_db(entries: list[dict]) -> sqlite3.Connection:
    """Build an in-memory audit DB with the given rows for testing."""
    conn = sqlite3.connect(":memory:")
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
    for entry in entries:
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, actor, memory_id, detail) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                entry.get("timestamp", datetime.now(timezone.utc).isoformat()),
                entry.get("action", "feedback"),
                entry.get("actor"),
                entry.get("memory_id"),
                entry.get("detail"),
            ),
        )
    conn.commit()
    return conn


# ---------------------------------------------------------------------------
# Tests for replay_feedback.py
# ---------------------------------------------------------------------------


def test_replay_ratio_three_useful_two_not(monkeypatch):
    """5 feedback entries (3 useful, 2 not) → ratio_30d = 0.6."""
    from scripts.replay_feedback import replay, write_weights

    now = datetime.now(timezone.utc)

    entries = []
    for i in range(3):
        entries.append({
            "timestamp": (now - timedelta(hours=i * 2)).isoformat(),
            "detail": f"query='test' useful=True mem_id=m{i}",
        })
    for i in range(2):
        entries.append({
            "timestamp": (now - timedelta(hours=10 + i * 2)).isoformat(),
            "detail": f"query='test' useful=False mem_id=m{i+3}",
        })

    # Build a real temp sqlite file so replay can open it by path
    tmpdir = Path(tempfile.mkdtemp())
    db_path = tmpdir / "test_audit.sqlite"
    conn = sqlite3.connect(str(db_path))
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
    for e in entries:
        conn.execute(
            "INSERT INTO audit_log (timestamp, action, actor, memory_id, detail) VALUES (?, ?, ?, ?, ?)",
            (e["timestamp"], "feedback", None, None, e["detail"]),
        )
    conn.commit()
    conn.close()

    try:
        weights = replay(db_path)
        assert weights["total_useful"] == 3
        assert weights["total_not_useful"] == 2
        assert weights["ratio_30d"] == 0.6
        assert weights["ratio_24h"] == pytest.approx(3.0 / 5.0)
        assert weights["ratio_7d"] == 0.6

        # Write the snapshot and verify 0600
        out_path = tmpdir / "feedback-weights.json"
        write_weights(weights, out_path)
        assert out_path.exists()
        assert os.stat(out_path).st_mode & 0o777 == 0o600

        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["ratio_30d"] == 0.6
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


def test_replay_no_audit_db():
    """No audit log file → graceful empty weights (no crash)."""
    from scripts.replay_feedback import replay

    missing_path = Path("/tmp/does-not-exist-xyz/audit.sqlite")
    weights = replay(missing_path)
    assert weights["total_useful"] == 0
    assert weights["total_not_useful"] == 0
    assert weights["ratio_24h"] is None
    assert weights["ratio_30d"] is None


def test_atomic_write_0600(monkeypatch):
    """Weight snapshot file has 0600 permissions."""
    from scripts.replay_feedback import write_weights

    tmpdir = Path(tempfile.mkdtemp())
    out_path = tmpdir / "test-weights.json"
    try:
        weights = {
            "ratio_24h": 0.5,
            "ratio_7d": 0.5,
            "ratio_30d": 0.5,
            "total_useful": 1,
            "total_not_useful": 1,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }
        write_weights(weights, out_path)
        assert out_path.exists()
        mode = os.stat(out_path).st_mode & 0o777
        assert mode == 0o600, f"expected 0o600 got {oct(mode)}"

        # Verify the output is valid JSON
        with open(out_path) as f:
            loaded = json.load(f)
        assert loaded["ratio_7d"] == 0.5
    finally:
        import shutil
        shutil.rmtree(tmpdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Tests for ranking with feedback_weights
# ---------------------------------------------------------------------------


def test_rank_memories_feedback_weights_adjusts_importance(mem_core, mem_long, settings):
    """When feedback_weights has ratio_7d=0.9 (high usefulness), the
    importance_boost should be amplified (×1.2).
    """
    from openclaw_memory_os.ranking import rank_memories, _feedback_weight_scale

    # Default scale = 1.0 for no weights
    assert _feedback_weight_scale(None) == 1.0

    # High usefulness → scale = 1.2
    high_weights = {"ratio_7d": 0.9}
    scale_high = _feedback_weight_scale(high_weights)
    assert scale_high == 1.2

    # Low usefulness → scale = 0.8
    low_weights = {"ratio_7d": 0.1}
    scale_low = _feedback_weight_scale(low_weights)
    assert scale_low == 0.8

    # Mid usefulness → scale = 1.0
    mid_weights = {"ratio_7d": 0.5}
    scale_mid = _feedback_weight_scale(mid_weights)
    assert scale_mid == 1.0

    # Fallback chaining: no 7d, check 30d
    fallback_weights = {"ratio_30d": 0.75}
    scale_fb = _feedback_weight_scale(fallback_weights)
    assert scale_fb == 1.2

    # Now actually verify importance_boost differs
    req = RecallRequest(query="memory", mode="hybrid", limit=10)
    high_hits, _ = rank_memories(
        [mem_core, mem_long], req, settings=settings,
        feedback_weights={"ratio_7d": 0.9},
    )
    low_hits, _ = rank_memories(
        [mem_core, mem_long], req, settings=settings,
        feedback_weights={"ratio_7d": 0.1},
    )

    # The difference in scores between core and long should be larger
    # with high feedback (amplified importance)
    core_score_high = next(h.score for h in high_hits if h.id == "core-1")
    long_score_high = next(h.score for h in high_hits if h.id == "long-1")
    core_score_low = next(h.score for h in low_hits if h.id == "core-1")
    long_score_low = next(h.score for h in low_hits if h.id == "long-1")

    gap_high = core_score_high - long_score_high
    gap_low = core_score_low - long_score_low
    assert gap_high > gap_low, (
        "With high feedback (1.2x importance), the gap between high and low "
        "importance memories should be larger than with low feedback (0.8x)."
    )


def test_build_recall_response_passes_feedback_weights(mem_core, mem_long, settings):
    """``build_recall_response`` must forward feedback_weights to
    ``rank_memories`` so the scaling takes effect."""
    from openclaw_memory_os.ranking import build_recall_response

    req = RecallRequest(query="memory", mode="hybrid", limit=10)
    resp_default = build_recall_response(
        [mem_core, mem_long], req, backend_name="test", settings=settings,
    )
    resp_weighted = build_recall_response(
        [mem_core, mem_long], req, backend_name="test", settings=settings,
        feedback_weights={"ratio_7d": 0.9},
    )

    # With high feedback (×1.2), the core memory should score higher
    # relative to the long memory.
    high_core_score = next(h.score for h in resp_weighted.hits if h.id == "core-1")
    default_core_score = next(h.score for h in resp_default.hits if h.id == "core-1")
    assert high_core_score > default_core_score, (
        "High usefulness feedback should boost the importance component, "
        "making high-importance memories score even higher."
    )


# ---------------------------------------------------------------------------
# Tests for /api/feedback-summary endpoint
# ---------------------------------------------------------------------------


def test_feedback_summary_endpoint_returns_ratios(monkeypatch):
    """The ``/api/feedback-summary`` endpoint must return a JSON response
    with the ``weights`` key containing the aggregated ratios.
    """
    from fastapi.testclient import TestClient
    from openclaw_memory_os.app import create_app
    from openclaw_memory_os.backends import SampleBackend
    from openclaw_memory_os.audit import AuditStore

    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    audit_path = tmpdir / "feedback_test.sqlite"

    # Isolated audit store — clear the singleton
    import openclaw_memory_os.audit as _audit_mod
    with _audit_mod._lock:
        old_store = _audit_mod._default_store
        _audit_mod._default_store = None

    fresh_audit = AuditStore(db_path=audit_path)
    # Must patch both app and feedback module
    monkeypatch.setattr(
        "openclaw_memory_os.app.get_audit_store", lambda *_a, **_kw: fresh_audit
    )
    monkeypatch.setattr(
        "openclaw_memory_os.feedback.get_audit_store", lambda *_a, **_kw: fresh_audit
    )

    # Ensure a backend is available
    backend = SampleBackend(
        Path(__file__).resolve().parent.parent / "data" / "sample_memories.json"
    )
    monkeypatch.setattr(
        "openclaw_memory_os.app.get_backend", lambda *_a, **_kw: backend
    )

    app = create_app()

    with TestClient(app) as c:
        # First, record some feedback so the audit log has entries
        r1 = c.post(
            "/api/feedback",
            json={"memory_id": "m1", "query": "test", "useful": True},
        )
        assert r1.status_code == 200
        r2 = c.post(
            "/api/feedback",
            json={"memory_id": "m2", "query": "test", "useful": False},
        )
        assert r2.status_code == 200

        # Now check the summary
        r = c.get("/api/feedback-summary")
        assert r.status_code == 200
        body = r.json()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    assert "weights" in body
    w = body["weights"]
    assert w["total_useful"] >= 1
    assert w["total_not_useful"] >= 1
    assert w["ratio_30d"] is not None, (
        "With at least one useful and one not-useful entry recorded, "
        "ratio_30d should be non-null."
    )

    # Restore singleton
    with _audit_mod._lock:
        _audit_mod._default_store = old_store


def test_feedback_summary_endpoint_empty(monkeypatch):
    """When no feedback has been recorded, the summary should return
    zero counts and null ratios (not crash).
    """
    from fastapi.testclient import TestClient
    from openclaw_memory_os.app import create_app
    from openclaw_memory_os.backends import SampleBackend
    from openclaw_memory_os.audit import AuditStore

    import tempfile
    tmpdir = Path(tempfile.mkdtemp())
    audit_path = tmpdir / "empty_test.sqlite"

    # Isolated empty audit store
    import openclaw_memory_os.audit as _audit_mod
    with _audit_mod._lock:
        old_store = _audit_mod._default_store
        _audit_mod._default_store = None

    fresh_audit = AuditStore(db_path=audit_path)
    monkeypatch.setattr(
        "openclaw_memory_os.app.get_audit_store", lambda *_a, **_kw: fresh_audit
    )
    monkeypatch.setattr(
        "openclaw_memory_os.feedback.get_audit_store", lambda *_a, **_kw: fresh_audit
    )

    monkeypatch.setattr(
        "openclaw_memory_os.app.get_backend", lambda *_a, **_kw: SampleBackend(
            Path(__file__).resolve().parent.parent / "data" / "sample_memories.json"
        )
    )

    app = create_app()

    with TestClient(app) as c:
        r = c.get("/api/feedback-summary")
        assert r.status_code == 200
        body = r.json()

    import shutil
    shutil.rmtree(tmpdir, ignore_errors=True)

    assert "weights" in body
    w = body["weights"]
    # With a fresh empty audit store, no feedback has been recorded
    assert w["total_useful"] == 0
    assert w["total_not_useful"] == 0

    # Restore singleton
    with _audit_mod._lock:
        _audit_mod._default_store = old_store
