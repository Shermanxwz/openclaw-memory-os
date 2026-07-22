"""Tests for the SQLite audit log module."""

from __future__ import annotations

from pathlib import Path


from openclaw_memory_os.audit import AuditStore, get_audit_store
from openclaw_memory_os.models import AuditLogEntry


def test_audit_store_creates_db(tmp_path: Path):
    db = tmp_path / "test_audit.sqlite"
    store = AuditStore(db_path=db)
    # Force connection creation by calling a method
    _ = store.count()
    assert db.exists()
    # Schema exists
    conn = store._connection()
    tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    assert any(r["name"] == "audit_log" for r in tables)


def test_audit_log_insert_and_count(tmp_path: Path):
    store = AuditStore(db_path=tmp_path / "test.sqlite")
    row_id = store.log("test_action", actor="test", memory_id="mem-001", detail="test detail")
    assert row_id >= 1
    assert store.count() == 1
    assert store.count(action="test_action") == 1
    assert store.count(action="nonexistent") == 0


def test_audit_list_recent_returns_entries(tmp_path: Path):
    store = AuditStore(db_path=tmp_path / "test.sqlite")
    store.log("action_a", detail="first")
    store.log("action_b", detail="second")
    entries = store.list_recent(limit=10)
    assert len(entries) == 2
    # Newest first
    assert entries[0].detail == "second"
    assert entries[1].detail == "first"
    assert all(isinstance(e, AuditLogEntry) for e in entries)


def test_audit_list_recent_filters_by_action(tmp_path: Path):
    store = AuditStore(db_path=tmp_path / "test.sqlite")
    store.log("ingest", detail="chunk 1")
    store.log("feedback", detail="useful")
    store.log("ingest", detail="chunk 2")
    entries = store.list_recent(limit=10, action="ingest")
    assert len(entries) == 2
    assert all(e.action == "ingest" for e in entries)


def test_audit_list_recent_pagination(tmp_path: Path):
    store = AuditStore(db_path=tmp_path / "test.sqlite")
    for i in range(5):
        store.log("page_test", detail=f"entry {i}")
    page1 = store.list_recent(limit=2, offset=0)
    assert len(page1) == 2
    assert page1[0].detail == "entry 4"
    page2 = store.list_recent(limit=2, offset=2)
    assert len(page2) == 2
    assert page2[0].detail == "entry 2"


def test_audit_detail_truncated(tmp_path: Path):
    store = AuditStore(db_path=tmp_path / "test.sqlite")
    long_detail = "x" * 5000
    store.log("long_test", detail=long_detail)
    entries = store.list_recent(limit=1)
    assert len(entries[0].detail or "") <= 2000


def test_audit_close_and_reopen(tmp_path: Path):
    db = tmp_path / "reopen.sqlite"
    store = AuditStore(db_path=db)
    store.log("persist", detail="data")
    store.close()
    store2 = AuditStore(db_path=db)
    assert store2.count() == 1


def test_get_audit_store_singleton(tmp_path: Path):
    s1 = get_audit_store(db_path=tmp_path / "singleton.sqlite")
    s2 = get_audit_store(db_path=tmp_path / "singleton.sqlite")
    assert s1 is s2
