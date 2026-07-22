"""Tests for recall feedback tracking."""

from __future__ import annotations

from pathlib import Path


from openclaw_memory_os.feedback import encode_feedback_body, record_feedback
from openclaw_memory_os.models import FeedbackEntry


def test_feedback_entry_model():
    entry = FeedbackEntry(memory_id="mem-001", query="test query", useful=True)
    assert entry.memory_id == "mem-001"
    assert entry.query == "test query"
    assert entry.useful is True
    assert entry.note is None


def test_feedback_entry_with_note():
    entry = FeedbackEntry(memory_id="mem-001", query="test", useful=False, note="not relevant")
    assert entry.useful is False
    assert entry.note == "not relevant"


def test_encode_feedback_body():
    fb = encode_feedback_body("mem-001", "my query", True, note="helpful!")
    assert isinstance(fb, FeedbackEntry)
    assert fb.useful is True


def test_record_feedback_stores_in_audit(tmp_path: Path):
    from openclaw_memory_os.audit import get_audit_store
    # Force a test db path
    db = tmp_path / "fb_audit.sqlite"
    store = get_audit_store(db_path=db)

    row_id = record_feedback("mem-001", "test query", True, actor="test")
    assert row_id >= 1

    entries = store.list_recent(limit=10, action="feedback")
    assert len(entries) >= 1
    assert entries[0].memory_id == "mem-001"
    assert "useful=True" in (entries[0].detail or "")


def test_record_not_useful(tmp_path: Path):
    from openclaw_memory_os.audit import get_audit_store
    db = tmp_path / "fb_audit2.sqlite"
    store = get_audit_store(db_path=db)

    record_feedback("mem-002", "bad query", False, note="wrong results")
    entries = store.list_recent(limit=10, action="feedback")
    fb_entry = next(e for e in entries if e.memory_id == "mem-002")
    assert "useful=False" in (fb_entry.detail or "")
    assert "wrong results" in (fb_entry.detail or "")
