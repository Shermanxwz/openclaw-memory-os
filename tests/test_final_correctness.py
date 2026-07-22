from __future__ import annotations

import importlib.util
import sqlite3
import sys
import types
from datetime import datetime, timezone
from pathlib import Path


def test_private_path_helper_and_schema_migration_are_canonical():
    source = Path("openclaw_memory_os/recall_feedback.py").read_text(encoding="utf-8")
    assert source.count("def _enforce_private_path(") == 1
    assert "_detach_feedback_events_foreign_key" not in source


def test_v3_migration_detaches_feedback_fk_and_preserves_rows():
    from openclaw_memory_os.migration import SCHEMA_VERSION, run_migrations
    assert SCHEMA_VERSION == 3
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript("""
        CREATE TABLE metadata (key TEXT PRIMARY KEY, value TEXT);
        CREATE TABLE migration_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            from_version INTEGER NOT NULL, to_version INTEGER NOT NULL,
            applied_at TEXT NOT NULL, success INTEGER NOT NULL, detail TEXT);
        INSERT INTO metadata(key,value) VALUES ('schema_version','2');
        CREATE TABLE recall_runs (query_id TEXT PRIMARY KEY);
        CREATE TABLE feedback_events (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_id TEXT NOT NULL REFERENCES recall_runs(query_id),
            candidate_key TEXT NOT NULL, memory_id TEXT, useful INTEGER NOT NULL,
            created_at TEXT, feedback_source TEXT, migration_status TEXT);
        INSERT INTO recall_runs(query_id) VALUES ('q1');
        INSERT INTO feedback_events(
            query_id,candidate_key,memory_id,useful,created_at,migration_status
        ) VALUES ('q1','alpha:1','1',1,'2026-01-01','migrated:audit');
    """)
    conn.commit()
    assert run_migrations(conn) == 3
    assert conn.execute("PRAGMA foreign_key_list(feedback_events)").fetchall() == []
    assert conn.execute("SELECT candidate_key FROM feedback_events").fetchone()[0] == "alpha:1"
    columns = {row[1] for row in conn.execute("PRAGMA table_info(feedback_events)")}
    assert {"collection", "resolution_status", "legacy_source_key"} <= columns
    history = conn.execute(
        "SELECT success FROM migration_history WHERE from_version=2 AND to_version=3"
    ).fetchone()
    assert history is not None and history[0] == 1


def test_collection_lookup_failure_never_uses_bare_id_cache():
    from openclaw_memory_os.backends import QdrantBackend
    class BrokenClient:
        def retrieve(self, *args, **kwargs):
            raise RuntimeError("unavailable")
    backend = object.__new__(QdrantBackend)
    backend._client = BrokenClient()
    backend._collection = "wanted"
    backend._secondary_collections = []
    assert backend.get_memory_in_collection("wanted", "same") is None


def test_candidate_pass_windows_never_cross_versions(monkeypatch):
    from openclaw_memory_os import evolution as evo
    moments = iter([
        datetime(2026, 1, 1, 0, 0, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 11, tzinfo=timezone.utc),
        datetime(2026, 1, 1, 0, 22, tzinfo=timezone.utc),
    ])
    monkeypatch.setattr(evo, "_now", lambda: next(moments))
    state = {"pass_windows": [], "consecutive_passes": 0, "pass_candidate_version": None}
    evo._record_cycle_result(state, "passed", candidate_version=10)
    evo._record_cycle_result(state, "passed", candidate_version=11)
    assert state["consecutive_passes"] == 1
    assert evo._two_consecutive_pass_windows(state, 11) is False
    evo._record_cycle_result(state, "passed", candidate_version=11)
    assert state["consecutive_passes"] == 2
    assert evo._two_consecutive_pass_windows(state, 11) is True


def test_strict_promotion_thresholds():
    from openclaw_memory_os import evolution as evo
    assert evo._PROMOTION_NEGATIVE_AT_5_TOLERANCE == 0.0
    assert evo._PROMOTION_FALLBACK_USEFUL_TOLERANCE == 0.0
    assert evo._PROMOTION_P95_LATENCY_MAX_MULTIPLIER == 1.15


def _load_runner():
    spec = importlib.util.spec_from_file_location(
        "final_runner_test", Path("scripts/run_evolution_cycle.py")
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_runner_returns_nonzero_for_structured_cycle_error(monkeypatch):
    runner = _load_runner()
    fake_evolution = types.ModuleType("openclaw_memory_os.evolution")
    fake_evolution.rank_fn_with_policy = lambda engine, policy: (lambda q, qid: [])
    fake_evolution.run_evolution_cycle = lambda *a, **k: {"status": "error"}
    fake_retrieval = types.ModuleType("openclaw_memory_os.retrieval_engine")
    class Engine:
        def __init__(self, backend, policy_store):
            pass
    fake_retrieval.RetrievalEngine = Engine
    monkeypatch.setitem(sys.modules, "openclaw_memory_os.evolution", fake_evolution)
    monkeypatch.setitem(sys.modules, "openclaw_memory_os.retrieval_engine", fake_retrieval)
    class Store:
        def get(self):
            return object()
    monkeypatch.setattr(runner, "_build_backend_and_store", lambda: (object(), Store(), None))
    assert runner.main() == runner.EXIT_UNEXPECTED == 1
