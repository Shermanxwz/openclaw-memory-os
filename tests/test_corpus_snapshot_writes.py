"""G5.5 \u2014 corpus_snapshot_id is written on every recall_runs insert.

Pre-G5.5 contract: the ``recall_runs.corpus_snapshot_id`` column
existed in the schema and the dashboard read it back, but every
write path (``record_recall_run``, evolution cycle, runner, the
``/api/recall-test`` handler, the ``/api/recall-handler`` handler)
left it ``NULL``. Operators could not tell whether two ``EvalResult``
rows compared the same corpus, and the G6.7 rollback triggers
could not detect a corpus change as a separate signal.

G5.5 fixes the gap:

* ``recall_feedback.compute_corpus_snapshot_id`` fingerprints the
  current corpus (``<backend_kind>:<count>:<sha256_prefix>``).
* ``record_recall_run`` auto-computes and writes the id when the
  caller did not supply one.

These tests pin the new contract end-to-end:

1. ``test_compute_snapshot_id_is_deterministic`` \u2014 calling
   ``compute_corpus_snapshot_id`` twice on the same backend
   produces the same string.
2. ``test_compute_snapshot_id_changes_when_point_count_changes`` \u2014
   adding a memory changes the count and (when applicable) the
   hash digest.
3. ``test_record_recall_run_writes_snapshot_id`` \u2014 the column
   is populated when ``record_recall_run`` is called.
4. ``test_snapshot_id_for_sample_backend_is_stable`` \u2014 the
   sample-backend fingerprint is stable across two reads.
5. ``test_snapshot_id_for_sample_backend_changes_when_file_changes`` \u2014
   a different sample file produces a different fingerprint.

The tests use the real ``SampleBackend`` (which is what ``get_backend``
returns when no Qdrant is configured \u2014 the default test
environment) and a tmp file copy so file-identity changes are
observable. ``record_recall_run`` is exercised end-to-end against
the live SQLite DB; the test cleans up its rows on teardown so
no state leaks into other tests in this file.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import List

import pytest


# ---------------------------------------------------------------------------
# Fixtures + helpers
# ---------------------------------------------------------------------------


@pytest.fixture
def isolated_recall_state_dir(monkeypatch, tmp_path):
    """Redirect ``MEMORY_OS_RECALL_STATE_DIR`` so the snapshot tests
    don't pollute the operator's real ``recall_feedback.db``.

    Each test gets a fresh tmp dir so writes from one test are
    invisible to the next. We also reload the recall-feedback
    module constants because they capture the env var at import
    time (the same gotcha Wave 5 hit in the evolution tests).
    """
    state_dir = tmp_path / "recall-state"
    state_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(state_dir))
    # The ``_RECALL_DB_DIR`` constant on the live module is read at
    # import time; rebuild it so writes go to the tmp dir.
    from openclaw_memory_os import recall_feedback as rf
    new_db_dir = state_dir / "openclaw-memory-os"
    monkeypatch.setattr(rf, "_RECALL_DB_DIR", new_db_dir)
    monkeypatch.setattr(rf, "_RECALL_DB", new_db_dir / "recall_feedback.db")
    yield state_dir


@pytest.fixture
def sample_backend_with_file(tmp_path):
    """Build a ``SampleBackend`` backed by a tmp copy of the
    bundled ``data/sample_memories.json``.

    The tmp copy is what the tests mutate (and what the
    ``test_snapshot_id_for_sample_backend_changes_when_file_changes``
    test rewrites). Tests must NOT mutate the bundled data file.
    """
    from openclaw_memory_os.backends import SampleBackend

    src = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "sample_memories.json"
    )
    assert src.exists(), f"bundled sample data missing: {src}"
    dst = tmp_path / "sample.json"
    shutil.copy(src, dst)
    backend = SampleBackend(dst)
    backend._load()  # force eager load so the cache is populated
    return backend


def _make_minimal_sample_json(out_path: Path, memories: List[dict]) -> Path:
    """Write a tiny sample JSON file with the given memories.

    Used by the "file changes" test to fabricate a corpus with a
    different size / mtime so the file-derived fingerprint
    changes. The schema matches the bundled sample (top-level
    ``memories`` list of dicts).
    """
    out_path.write_text(json.dumps({"version": 1, "memories": memories}), encoding="utf-8")
    return out_path


def _cleanup_query_ids(query_ids: List[str]) -> None:
    """Best-effort delete of recall_runs rows written during a test.

    Keeps the global recall_feedback DB tidy \u2014 without this,
    rows from one test would surface in later tests' SELECT
    queries. We use a direct connection so we don't have to
    depend on a public purge API.
    """
    if not query_ids:
        return
    from openclaw_memory_os.recall_feedback import _get_db
    conn = _get_db()
    try:
        conn.execute(
            "DELETE FROM recall_runs WHERE query_id IN ({})".format(
                ", ".join("?" for _ in query_ids)
            ),
            query_ids,
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Test 1 \u2014 determinism
# ---------------------------------------------------------------------------


def test_compute_snapshot_id_is_deterministic(sample_backend_with_file):
    """Calling ``compute_corpus_snapshot_id`` twice on the same
    backend produces the same string.

    Determinism is the foundation of the entire G5.5 contract:
    without it, two consecutive ``record_recall_run`` calls on
    the same corpus would record different snapshot ids and the
    evolution cycle would mistakenly believe the corpus changed.
    """
    from openclaw_memory_os.recall_feedback import compute_corpus_snapshot_id

    sid_a = compute_corpus_snapshot_id(sample_backend_with_file)
    sid_b = compute_corpus_snapshot_id(sample_backend_with_file)

    assert sid_a, "snapshot id must be non-empty"
    assert sid_a == sid_b, (
        f"snapshot id must be deterministic; got {sid_a!r} then {sid_b!r}"
    )
    # And it must follow the documented ``kind:count:digest`` shape.
    parts = sid_a.split(":")
    assert len(parts) == 3, f"snapshot must have 3 colon-separated parts; got {sid_a!r}"
    kind, count, digest = parts
    assert kind == "sample", kind
    assert int(count) == len(sample_backend_with_file.list_memories()), count
    assert len(digest) == 12, digest
    assert all(c in "0123456789abcdef" for c in digest), digest


# ---------------------------------------------------------------------------
# Test 2 \u2014 snapshot changes when point count changes
# ---------------------------------------------------------------------------


def test_compute_snapshot_id_changes_when_point_count_changes(
    sample_backend_with_file, tmp_path
):
    """Adding a memory changes the snapshot.

    The fingerprint encodes the live ``list_memories()`` count for
    ``SampleBackend`` (so adding a memory to the in-memory cache
    without rewriting the JSON still changes the id) and the file
    size + mtime (so writing a new JSON changes the id too). This
    test exercises the file path: rewriting the JSON to add a new
    memory must produce a different id.

    We also confirm the count field reflects the new total so
    downstream comparison code can detect a 1-row drift at a
    glance.
    """
    from openclaw_memory_os.recall_feedback import compute_corpus_snapshot_id

    sid_before = compute_corpus_snapshot_id(sample_backend_with_file)
    count_before = int(sid_before.split(":")[1])
    assert count_before > 0, "sample backend must have at least one memory"

    # Append a new memory to the sample file + force a reload.
    new_memory = {
        "id": "mem-zzzz-new",
        "text": "Synthetic memory added by the snapshot test.",
        "summary": "synthetic",
        "tier": "medium",
        "status": "active",
        "importance": 0.5,
        "tags": ["test"],
        "source": "tests/test_corpus_snapshot_writes.py",
        "created_at": "2026-07-16T00:00:00Z",
    }
    raw = json.loads(sample_backend_with_file.path.read_text(encoding="utf-8"))
    raw["memories"].append(new_memory)
    sample_backend_with_file.path.write_text(json.dumps(raw), encoding="utf-8")
    sample_backend_with_file.reload()

    sid_after = compute_corpus_snapshot_id(sample_backend_with_file)
    count_after = int(sid_after.split(":")[1])

    assert count_after == count_before + 1, (
        f"count must reflect the new memory: before={count_before} after={count_after}"
    )
    assert sid_after != sid_before, (
        f"snapshot id must change when a memory is added; "
        f"got {sid_before!r} then {sid_after!r}"
    )


# ---------------------------------------------------------------------------
# Test 3 \u2014 record_recall_run writes the snapshot id
# ---------------------------------------------------------------------------


def test_record_recall_run_writes_snapshot_id(
    isolated_recall_state_dir, sample_backend_with_file, monkeypatch
):
    """``record_recall_run`` must populate ``corpus_snapshot_id``
    even when the caller did not pass one explicitly.

    We pin both ends of the contract:

    * the column is non-null after the call (``record_recall_run``
      auto-computed it),
    * the value matches the value produced by an explicit call to
      ``compute_corpus_snapshot_id(backend)`` on the same backend
      \u2014 so the on-disk id is always reproducible by re-running
      the fingerprint against the live corpus.
    """
    from openclaw_memory_os.recall_feedback import (
        compute_corpus_snapshot_id,
        get_recall_runs_for_query,
        record_recall_run,
    )

    # Force ``get_current_backend`` to return our sample backend so
    # the auto-fingerprint uses the same corpus the test inspects.
    from openclaw_memory_os import recall_feedback as rf

    monkeypatch.setattr(rf, "get_current_backend", lambda: sample_backend_with_file)

    expected_sid = compute_corpus_snapshot_id(sample_backend_with_file)

    qid = "g55-snap-test-1"
    _cleanup_query_ids([qid])  # idempotency guard
    try:
        # The caller omits corpus_snapshot_id \u2014 the function must
        # auto-compute and write it.
        record_recall_run(qid, "snapshot write test query")

        row = get_recall_runs_for_query(qid)
        assert row is not None
        assert row["corpus_snapshot_id"], (
            "G5.5: record_recall_run must auto-populate corpus_snapshot_id "
            "when the caller omits it"
        )
        assert row["corpus_snapshot_id"] == expected_sid, (
            f"on-disk snapshot {row['corpus_snapshot_id']!r} must match "
            f"the explicit fingerprint {expected_sid!r}"
        )
        # ``query_hash`` must still be auto-derived (legacy contract).
        assert row["query_hash"] and len(row["query_hash"]) == 16
    finally:
        _cleanup_query_ids([qid])


def test_record_recall_run_honours_caller_supplied_snapshot_id(
    isolated_recall_state_dir, sample_backend_with_file, monkeypatch
):
    """When the caller passes an explicit ``corpus_snapshot_id``,
    the function must NOT overwrite it with the auto-computed
    fingerprint (callers that ship their own id have a reason).

    This protects the ``app.py`` / ``cli.py`` write paths, which
    currently pass ``diag.get('corpus_snapshot_id')`` and would
    silently lose the value if the auto-compute ran unconditionally.
    """
    from openclaw_memory_os.recall_feedback import (
        get_recall_runs_for_query,
        record_recall_run,
    )

    sentinel = "caller-supplied:999:deadbeefdead"
    qid = "g55-snap-test-caller"
    _cleanup_query_ids([qid])
    try:
        record_recall_run(
            qid,
            "caller-supplied snapshot",
            corpus_snapshot_id=sentinel,
        )
        row = get_recall_runs_for_query(qid)
        assert row is not None
        assert row["corpus_snapshot_id"] == sentinel, (
            f"caller-supplied id must be preserved; got "
            f"{row['corpus_snapshot_id']!r}, expected {sentinel!r}"
        )
    finally:
        _cleanup_query_ids([qid])


# ---------------------------------------------------------------------------
# Test 4 \u2014 sample backend fingerprint is stable
# ---------------------------------------------------------------------------


def test_snapshot_id_for_sample_backend_is_stable(tmp_path):
    """Two reads of the same ``SampleBackend`` produce the same
    fingerprint.

    Even though the fingerprint encodes file mtime + size, those
    don't change between calls when the file isn't mutated. The
    test uses a tmp copy so it can't accidentally mutate the
    bundled sample file.
    """
    from openclaw_memory_os.backends import SampleBackend
    from openclaw_memory_os.recall_feedback import compute_corpus_snapshot_id

    src = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "sample_memories.json"
    )
    dst = tmp_path / "stable.json"
    shutil.copy(src, dst)

    backend_a = SampleBackend(dst)
    backend_a._load()
    backend_b = SampleBackend(dst)
    backend_b._load()

    sid_a = compute_corpus_snapshot_id(backend_a)
    sid_b = compute_corpus_snapshot_id(backend_b)
    assert sid_a == sid_b, (
        f"sample backend fingerprint must be stable; "
        f"got {sid_a!r} then {sid_b!r}"
    )
    # And the count must match the bundled sample.
    assert int(sid_a.split(":")[1]) == len(backend_a.list_memories())


# ---------------------------------------------------------------------------
# Test 5 \u2014 sample backend fingerprint changes when the file changes
# ---------------------------------------------------------------------------


def test_snapshot_id_for_sample_backend_changes_when_file_changes(
    tmp_path,
):
    """Different sample file (different size / mtime / count) \u2192
    different fingerprint.

    We fabricate two tmp sample files: one with the bundled 15
    memories and a small extra entry, one with just the bundled
    15. Their fingerprints must differ. We also assert that the
    count field reflects the difference so a downstream
    comparison can distinguish "same count, different content"
    from "different count".
    """
    from openclaw_memory_os.backends import SampleBackend
    from openclaw_memory_os.recall_feedback import compute_corpus_snapshot_id

    # Build a smaller corpus (5 memories).
    small = [
        {
            "id": f"mem-{i:04d}",
            "text": f"small memory {i}",
            "summary": f"small {i}",
            "tier": "medium",
            "status": "active",
            "importance": 0.5,
            "tags": [],
            "source": "tests/synthetic",
            "created_at": "2026-07-16T00:00:00Z",
        }
        for i in range(5)
    ]
    # Build a larger corpus (5 + 20 extra).
    large = list(small) + [
        {
            "id": f"mem-extra-{i:04d}",
            "text": f"extra memory {i}",
            "summary": f"extra {i}",
            "tier": "medium",
            "status": "active",
            "importance": 0.5,
            "tags": [],
            "source": "tests/synthetic",
            "created_at": "2026-07-16T00:00:00Z",
        }
        for i in range(20)
    ]

    small_path = _make_minimal_sample_json(tmp_path / "small.json", small)
    large_path = _make_minimal_sample_json(tmp_path / "large.json", large)

    backend_small = SampleBackend(small_path)
    backend_small._load()
    backend_large = SampleBackend(large_path)
    backend_large._load()

    sid_small = compute_corpus_snapshot_id(backend_small)
    sid_large = compute_corpus_snapshot_id(backend_large)

    assert sid_small != sid_large, (
        f"different sample files must yield different fingerprints; "
        f"got {sid_small!r} and {sid_large!r}"
    )
    assert int(sid_small.split(":")[1]) == 5
    assert int(sid_large.split(":")[1]) == 25

    # And the same file produces the same fingerprint on two reads
    # (sanity: we don't want the test to pass for the wrong reason).
    sid_small_b = compute_corpus_snapshot_id(backend_small)
    assert sid_small == sid_small_b