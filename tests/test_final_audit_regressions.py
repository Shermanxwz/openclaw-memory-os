from __future__ import annotations

from pathlib import Path

import pytest

from openclaw_memory_os import policy_store as policy_module
from openclaw_memory_os.evolution import generate_candidates
from openclaw_memory_os.policy_store import (
    Policy,
    PolicyStatus,
    PolicyStore,
    baseline_policy,
)


def _policy(version: int, status: PolicyStatus = PolicyStatus.ACTIVE) -> Policy:
    return Policy(**baseline_policy, version=version, status=status)


def test_policy_defaults_obey_both_unit_sum_contracts():
    policy = Policy()
    assert (
        policy.final_rrf_weight
        + policy.final_vector_weight
        + policy.final_lexical_weight
    ) == pytest.approx(1.0)
    assert (
        policy.importance_weight
        + policy.recency_weight
        + policy.feedback_weight
    ) == pytest.approx(1.0)


def test_every_generated_candidate_obeys_both_unit_sum_contracts():
    candidates = generate_candidates(_policy(1), n_candidates=20, seed=19)
    assert len(candidates) == 20
    for policy in candidates:
        assert (
            policy.final_rrf_weight
            + policy.final_vector_weight
            + policy.final_lexical_weight
        ) == pytest.approx(1.0)
        assert (
            policy.importance_weight
            + policy.recency_weight
            + policy.feedback_weight
        ) == pytest.approx(1.0)


def test_failed_active_write_does_not_change_in_memory_policy(tmp_path, monkeypatch):
    target = tmp_path / "active.json"
    store = PolicyStore(path=target, initial=_policy(1))
    store.save()
    store.set_shadow(_policy(2, PolicyStatus.SHADOW))
    real_write = policy_module._atomic_json_write

    def fail_active(path: Path, payload):
        if path == target:
            raise OSError("disk full")
        return real_write(path, payload)

    monkeypatch.setattr(policy_module, "_atomic_json_write", fail_active)
    with pytest.raises(OSError, match="disk full"):
        store.promote()
    assert store.get().version == 1
    assert PolicyStore(path=target).get().version == 1


def test_candidate_unlink_failure_is_nonfatal_after_promotion(tmp_path, monkeypatch):
    target = tmp_path / "active.json"
    store = PolicyStore(path=target, initial=_policy(1))
    store.save()
    store.set_shadow(_policy(2, PolicyStatus.SHADOW))
    candidate_path = tmp_path / "candidate.json"
    real_unlink = Path.unlink

    def fail_candidate(self: Path, *args, **kwargs):
        if self == candidate_path:
            raise PermissionError("read only")
        return real_unlink(self, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", fail_candidate)
    store.promote()
    assert store.get().version == 2
    assert PolicyStore(path=target).get().version == 2


def test_failed_rollback_write_keeps_current_in_memory(tmp_path, monkeypatch):
    target = tmp_path / "active.json"
    store = PolicyStore(path=target, initial=_policy(1))
    store.save()
    store.set(_policy(2))
    store.save()
    real_write = policy_module._atomic_json_write

    def fail_active(path: Path, payload):
        if path == target:
            raise OSError("disk full")
        return real_write(path, payload)

    monkeypatch.setattr(policy_module, "_atomic_json_write", fail_active)
    with pytest.raises(OSError, match="disk full"):
        store.revert()
    assert store.get().version == 2
    assert PolicyStore(path=target).get().version == 2
