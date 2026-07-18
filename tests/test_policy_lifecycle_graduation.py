from __future__ import annotations

import json
import os
from pathlib import Path

from openclaw_memory_os.policy_store import (
    Policy, PolicyStatus, PolicyStore, baseline_policy,
)


def _policy(version: int, status=PolicyStatus.ACTIVE, parent=None):
    data = dict(baseline_policy)
    data.update(version=version, status=status, parent_version=parent)
    return Policy(**data)


def test_policy_source_has_single_canonical_helpers():
    source = Path("openclaw_memory_os/policy_store.py").read_text(encoding="utf-8")
    assert source.count("def _secure_policy_path(") == 1
    assert source.count("def _load_auxiliary_policies(") == 1


def test_candidate_envelope_and_restart_restore(tmp_path):
    store = PolicyStore(policy_dir=tmp_path)
    active = _policy(10)
    store.set(active)
    store.save(active)
    candidate = _policy(11, PolicyStatus.SHADOW, 10)
    store.set_shadow(
        candidate,
        metadata={
            "corpus_snapshot_id": "snapshot-test",
            "offline_report_id": "report-test",
            "shadow_sample_count": 7,
            "consecutive_passes": 1,
        },
    )
    raw = json.loads((tmp_path / "candidate.json").read_text(encoding="utf-8"))
    assert raw["policy"]["version"] == 11
    assert raw["corpus_snapshot_id"] == "snapshot-test"
    assert raw["shadow_sample_count"] == 7
    restarted = PolicyStore(policy_dir=tmp_path)
    assert restarted.get().version == 10
    assert restarted.get_shadow().version == 11
    assert restarted.get_shadow_metadata()["offline_report_id"] == "report-test"


def test_promote_writes_previous_active_history_and_clears_candidate(tmp_path):
    store = PolicyStore(policy_dir=tmp_path)
    active = _policy(20)
    store.set(active)
    store.save(active)
    candidate = _policy(21, PolicyStatus.SHADOW, 20)
    store.set_shadow(candidate)
    store.promote()
    assert store.get().version == 21
    assert store.get_previous().version == 20
    assert not (tmp_path / "candidate.json").exists()
    assert json.loads((tmp_path / "active.json").read_text())["version"] == 21
    assert json.loads((tmp_path / "previous.json").read_text())["version"] == 20
    assert list((tmp_path / "history").glob("v20-retired-*.json"))
    restarted = PolicyStore(policy_dir=tmp_path)
    assert restarted.get().version == 21
    assert restarted.get_previous().version == 20


def test_corrupt_active_recovers_previous_and_rollback_uses_previous(tmp_path):
    store = PolicyStore(policy_dir=tmp_path)
    active = _policy(30)
    store.set(active)
    store.save(active)
    candidate = _policy(31, PolicyStatus.SHADOW, 30)
    store.set_shadow(candidate)
    store.promote()
    (tmp_path / "active.json").write_text("{broken", encoding="utf-8")
    recovered = PolicyStore(policy_dir=tmp_path)
    assert recovered.get().version == 30
    assert recovered.recovery_reason == "policy_recovery_previous"


def test_policy_permissions_are_private(tmp_path):
    store = PolicyStore(policy_dir=tmp_path)
    active = _policy(40)
    store.set(active)
    store.save(active)
    store.set_shadow(_policy(41, PolicyStatus.SHADOW, 40))
    if os.name != "nt":
        assert (tmp_path.stat().st_mode & 0o777) == 0o700
        assert ((tmp_path / "active.json").stat().st_mode & 0o777) == 0o600
        assert ((tmp_path / "candidate.json").stat().st_mode & 0o777) == 0o600
