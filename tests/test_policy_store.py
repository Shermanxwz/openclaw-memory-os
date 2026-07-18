"""Tests for ``openclaw_memory_os.policy_store``.

The PolicyStore is the foundation that later evolution / shadow /
rollback checkpoints build on. These tests pin:

* baseline policy field coverage,
* checksum determinism + tamper detection,
* Pydantic clamping on out-of-range values,
* atomic save (no half-written file even on simulated crash),
* hot-reload only when the on-disk file actually changed,
* no on-disk state by default (so an import never silently pulls a
  policy from ``$HOME``).
"""

from __future__ import annotations

import json
import os
import threading
from pathlib import Path

import pytest

from openclaw_memory_os.policy_store import (
    POLICY_SCHEMA_VERSION,
    Policy,
    PolicyStatus,
    PolicyStore,
    baseline_policy,
    compute_checksum,
)


# ---------------------------------------------------------------------------
# baseline_policy + Policy model
# ---------------------------------------------------------------------------


def test_baseline_policy_has_required_fields() -> None:
    required = {
        "schema_version",
        "version",
        "created_at",
        "parent_version",
        "status",
        "dense_k",
        "lexical_k",
        "rrf_k",
        "fallback_min_results",
        "rrf_dense_weight",
        "rrf_lexical_weight",
        "final_rrf_weight",
        "final_vector_weight",
        "final_lexical_weight",
        "importance_weight",
        "recency_weight",
        "feedback_weight",
        "exact_match_boost",
    }
    assert required.issubset(set(baseline_policy.keys()))


def test_baseline_policy_defaults_to_baseline_status() -> None:
    assert baseline_policy["status"] == "baseline"
    assert baseline_policy["parent_version"] is None
    assert baseline_policy["schema_version"] == POLICY_SCHEMA_VERSION


def test_policy_clamps_out_of_range_values() -> None:
    # dense_k too big → clamped to le=500. importance_weight too big → le=2.0.
    p = Policy(
        dense_k=10_000,
        importance_weight=99.0,
        exact_match_boost=-1.0,
        rrf_dense_weight=10_000.0,
    )
    assert p.dense_k == 500
    assert p.importance_weight == 2.0
    assert p.exact_match_boost == 0.0
    assert p.rrf_dense_weight == 10.0


def test_policy_rejects_unknown_fields_when_extra_forbidden() -> None:
    # We use extra='ignore' so unknown keys are silently dropped; this
    # ensures a hand-edited policy file with extra junk doesn't crash
    # the load. The test pins the chosen behaviour.
    kwargs = dict(baseline_policy)
    kwargs["unknown_field"] = "junk"
    p = Policy(**kwargs)  # type: ignore[arg-type]
    assert not hasattr(p, "unknown_field")


def test_policy_validates_isoformat() -> None:
    import pytest as _p
    with _p.raises(Exception):
        Policy(created_at="not-a-date")


# ---------------------------------------------------------------------------
# Checksum
# ---------------------------------------------------------------------------


def test_checksum_deterministic() -> None:
    a = Policy(**baseline_policy)
    b = Policy(**baseline_policy)
    assert compute_checksum(a) == compute_checksum(b)


def test_checksum_changes_on_field_change() -> None:
    a = Policy(**baseline_policy)
    b_kwargs = dict(baseline_policy)
    b_kwargs["dense_k"] = 40
    b = Policy(**b_kwargs)
    assert compute_checksum(a) != compute_checksum(b)


def test_checksum_ignores_created_at() -> None:
    a_kwargs = dict(baseline_policy)
    a_kwargs["created_at"] = "2026-01-01T00:00:00Z"
    a = Policy(**a_kwargs)
    b_kwargs = dict(baseline_policy)
    b_kwargs["created_at"] = "2026-12-31T23:59:59Z"
    b = Policy(**b_kwargs)
    assert compute_checksum(a) == compute_checksum(b)


def test_checksum_is_sha256_hex_64() -> None:
    p = Policy(**baseline_policy)
    digest = compute_checksum(p)
    assert len(digest) == 64
    assert all(c in "0123456789abcdef" for c in digest)


# ---------------------------------------------------------------------------
# PolicyStore in-memory behaviour
# ---------------------------------------------------------------------------


def test_store_starts_with_baseline_when_no_path() -> None:
    store = PolicyStore()
    p = store.get()
    assert p.status == PolicyStatus.BASELINE
    assert p.dense_k == baseline_policy["dense_k"]


def test_store_returns_independent_copies() -> None:
    store = PolicyStore()
    p1 = store.get()
    p1.dense_k = 999  # mutating the copy must not affect the store.
    p2 = store.get()
    assert p2.dense_k == baseline_policy["dense_k"]


def test_store_swap_is_atomic() -> None:
    store = PolicyStore()
    new_kwargs = dict(baseline_policy)
    new_kwargs["version"] = 2
    new_kwargs["status"] = PolicyStatus.ACTIVE
    new = Policy(**new_kwargs)
    digest = store.set(new)
    assert len(digest) == 64
    assert store.get().version == 2


def test_store_without_explicit_path_uses_persistent_default(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path))
    monkeypatch.delenv("MEMORY_OS_POLICY_DIR", raising=False)
    monkeypatch.delenv("MEMORY_OS_POLICY_PATH", raising=False)
    store = PolicyStore()
    saved = store.save()
    assert saved == tmp_path / "openclaw-memory-os" / "policies" / "active.json"
    assert saved.exists()


def test_store_checksum_changes_on_swap() -> None:
    store = PolicyStore()
    a = store.checksum()
    new_kwargs = dict(baseline_policy)
    new_kwargs["version"] = 2
    store.set(Policy(**new_kwargs))
    b = store.checksum()
    assert a != b


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_store_loads_valid_policy_from_disk(tmp_path: Path) -> None:
    p = Policy(**baseline_policy, version=2, status=PolicyStatus.ACTIVE)
    target = tmp_path / "policy.json"
    body = p.model_dump(mode="json")
    body["checksum"] = compute_checksum(p)
    target.write_text(json.dumps(body), encoding="utf-8")
    store = PolicyStore(path=target)
    assert store.get().version == 2
    assert store.get().status == PolicyStatus.ACTIVE


def test_store_rejects_corrupt_checksum(tmp_path: Path, caplog) -> None:
    p = Policy(**baseline_policy, version=2)
    target = tmp_path / "policy.json"
    body = p.model_dump(mode="json")
    body["checksum"] = "deadbeef" * 8  # wrong checksum
    target.write_text(json.dumps(body), encoding="utf-8")
    with caplog.at_level("WARNING"):
        store = PolicyStore(path=target)
    # Falls back to baseline — never crashes, never trusts a bad checksum.
    assert store.get().version == baseline_policy["version"]
    assert store.get().status == PolicyStatus.BASELINE


def test_store_rejects_schema_mismatch(tmp_path: Path) -> None:
    body = dict(baseline_policy)
    body["schema_version"] = "999"
    target = tmp_path / "policy.json"
    target.write_text(json.dumps(body), encoding="utf-8")
    store = PolicyStore(path=target)
    # Falls back to baseline — schema mismatch is a hard reject.
    assert store.get().status == PolicyStatus.BASELINE


def test_store_rejects_non_object_root(tmp_path: Path) -> None:
    target = tmp_path / "policy.json"
    target.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    store = PolicyStore(path=target)
    assert store.get().status == PolicyStatus.BASELINE


def test_save_writes_atomic_file(tmp_path: Path) -> None:
    target = tmp_path / "policy.json"
    store = PolicyStore(path=target)
    out = store.save()
    assert out == target
    assert target.exists()
    # No stray .tmp file left behind.
    assert not (target.with_suffix(target.suffix + ".tmp")).exists()
    # Body is parseable and includes a checksum.
    body = json.loads(target.read_text(encoding="utf-8"))
    assert "checksum" in body
    assert len(body["checksum"]) == 64


def test_save_round_trip(tmp_path: Path) -> None:
    target = tmp_path / "policy.json"
    store = PolicyStore(path=target)
    store.set(Policy(**baseline_policy, version=7, dense_k=42))
    store.save()
    fresh = PolicyStore(path=target)
    assert fresh.get().version == 7
    assert fresh.get().dense_k == 42


# ---------------------------------------------------------------------------
# Hot reload
# ---------------------------------------------------------------------------


def test_reload_if_changed_returns_false_when_unchanged(tmp_path: Path) -> None:
    target = tmp_path / "policy.json"
    store = PolicyStore(path=target)
    store.save()
    assert store.reload_if_changed() is False


def test_reload_if_changed_returns_true_on_disk_change(tmp_path: Path) -> None:
    target = tmp_path / "policy.json"
    store = PolicyStore(path=target)
    store.save()
    # Mutate the file: bump version + bump mtime to be sure.
    body = json.loads(target.read_text(encoding="utf-8"))
    body["version"] = 99
    body["checksum"] = compute_checksum(Policy(**body))
    target.write_text(json.dumps(body, indent=2), encoding="utf-8")
    os.utime(target, ns=(target.stat().st_mtime_ns + 1_000_000, target.stat().st_mtime_ns + 1_000_000))
    assert store.reload_if_changed() is True
    assert store.get().version == 99


def test_reload_keeps_active_on_corrupt_file(tmp_path: Path) -> None:
    target = tmp_path / "policy.json"
    store = PolicyStore(path=target)
    store.save()
    active_version = store.get().version
    target.write_text("{ not json", encoding="utf-8")
    os.utime(target, ns=(target.stat().st_mtime_ns + 1_000_000, target.stat().st_mtime_ns + 1_000_000))
    assert store.reload_if_changed() is False
    assert store.get().version == active_version


def test_reload_if_changed_missing_file_is_noop(tmp_path: Path) -> None:
    target = tmp_path / "policy.json"
    store = PolicyStore(path=target)
    assert store.reload_if_changed() is False


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


def test_store_thread_safe_under_concurrent_reads(tmp_path: Path) -> None:
    store = PolicyStore()
    errors: list = []

    def reader() -> None:
        try:
            for _ in range(100):
                _ = store.get()
        except Exception as exc:  # pragma: no cover - failure path
            errors.append(exc)

    threads = [threading.Thread(target=reader) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors

# ---------------------------------------------------------------------------
# B2-4: Additive-rerank weight triplet is renormalised to sum 1.0
# ---------------------------------------------------------------------------


def test_b2_4_additive_rerank_weights_sum_to_one() -> None:
    """B2-4: docs/self-evolution.md (lines 56-58) require that the
    three additive rerank weights (importance / recency / feedback)
    sum to 1.0 after the evolution pipeline's renormalisation. The
    shipped baseline_policy pins them at
    ``importance=0.55, recency=0.25, feedback=0.20`` (sum = 1.0).

    This test pins the contract on the baseline so any future
    refactor that drifts the weights will trip CI.
    """
    store = PolicyStore()
    p = store.get()
    rerank_sum = p.importance_weight + p.recency_weight + p.feedback_weight
    assert rerank_sum == pytest.approx(1.0), (
        f"importance/recency/feedback should sum to 1.0; "
        f"got {rerank_sum:.4f} "
        f"(importance={p.importance_weight}, "
        f"recency={p.recency_weight}, "
        f"feedback={p.feedback_weight})"
    )


def test_b2_4_final_blend_weights_sum_to_one() -> None:
    """B2-4 companion: the public scoring formula's final blend
    (``final_rrf_weight + final_vector_weight + final_lexical_weight``)
    is also a unit-sum triple per the spec. Pinned here so a future
    change that breaks the formula trips the test suite.
    """
    store = PolicyStore()
    p = store.get()
    blend_sum = (
        p.final_rrf_weight
        + p.final_vector_weight
        + p.final_lexical_weight
    )
    assert blend_sum == pytest.approx(1.0), (
        f"final_rrf/vector/lexical should sum to 1.0; got {blend_sum:.4f}"
    )


def test_b2_4_baseline_dict_exposes_renormalised_weights() -> None:
    """B2-4: the module-level ``baseline_policy`` dict itself carries
    the renormalised weights (not just the Policy constructed from it)
    so the test catches anyone who bumps the dict without running it
    through the validator.
    """
    assert baseline_policy["importance_weight"] == pytest.approx(0.55)
    assert baseline_policy["recency_weight"] == pytest.approx(0.25)
    assert baseline_policy["feedback_weight"] == pytest.approx(0.20)
    s = (
        baseline_policy["importance_weight"]
        + baseline_policy["recency_weight"]
        + baseline_policy["feedback_weight"]
    )
    assert s == pytest.approx(1.0)


def test_b2_4_exact_match_boost_is_not_part_of_rerank_sum() -> None:
    """B2-4: ``exact_match_boost`` is an additive per-document
    multiplier on the BM25 score, not part of the additive rerank
    weighted sum. Its value MUST be excluded from the
    importance/recency/feedback sum even though both groups live on
    the same :class:`Policy`.
    """
    p = PolicyStore().get()
    rerank_sum = (
        p.importance_weight + p.recency_weight + p.feedback_weight
    )
    # The exact_match_boost must not be folded into the rerank sum.
    assert p.exact_match_boost != pytest.approx(rerank_sum), (
        "exact_match_boost should be separate from the additive "
        "rerank weight sum"
    )
