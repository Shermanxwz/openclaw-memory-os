"""P0-1: ``scripts/run_evolution_cycle.py`` must NOT clobber the on-disk
active policy with the shipped baseline on every invocation.

Runbook G6.10 (active policy is source of truth) — the runner's
``_build_backend_and_store`` used to call
``store.set(Policy(**baseline_policy, status="active", version=1))``
in all three branches (QdrantBackend, SampleBackend, empty-sample
fallback). That overwrote whatever policy evolution promoted in a
previous cycle, every time the runner was invoked, which is
explicitly forbidden by the runbook.

This module locks the contract down with four tests:

1. ``test_runner_does_not_call_store_set_with_baseline`` — static
   check that the runner source no longer contains the offending
   ``store.set(Policy(**baseline_policy`` substring.
2. ``test_runner_qdrant_branch_preserves_active_policy`` — a
   ``Policy(version=42)`` is written to a tmp ``policy.json``,
   the runner's ``_build_backend_and_store`` is invoked with
   ``QDRANT_URL`` set (and the backends mocked so no real Qdrant
   is touched), and we assert the store that comes back still
   reports ``version == 42`` — i.e. the runner did NOT reset it
   to ``version=1``.
3. ``test_runner_sample_branch_preserves_active_policy`` — same
   shape but exercising the ``MEMORY_OS_SAMPLE_PATH`` branch.
4. ``test_runner_no_op_when_policy_dir_missing`` — when
   ``MEMORY_OS_POLICY_PATH`` points at a non-existent file, the
   runner must rely on ``PolicyStore``'s own default-baseline
   fallback (``initial or Policy(**baseline_policy)``) rather
   than explicitly calling ``.set()``. We assert the active
   policy still has the shipped baseline ``version == 1`` and
   that the *runner* never wrote a file (the on-disk path
   remains missing because we never told ``PolicyStore`` to
   save).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from unittest.mock import MagicMock


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SCRIPT_PATH = PROJECT_ROOT / "scripts" / "run_evolution_cycle.py"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_runner_module():
    """Import ``scripts/run_evolution_cycle.py`` as a fresh module.

    We can't ``import scripts.run_evolution_cycle`` directly because
    the ``scripts/`` directory does not ship an ``__init__.py`` and
    pytest's rootdir collection does not auto-promote it. So we use
    ``importlib.util.spec_from_file_location`` + ``module_from_spec``
    and shove the resulting module into ``sys.modules`` so subsequent
    ``importlib.reload`` calls behave like a normal import.
    """
    spec = importlib.util.spec_from_file_location(
        "scripts_run_evolution_cycle_under_test", SCRIPT_PATH
    )
    assert spec is not None and spec.loader is not None, (
        f"could not build import spec for {SCRIPT_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["scripts_run_evolution_cycle_under_test"] = module
    spec.loader.exec_module(module)
    return module


def _write_active_policy(policy_path: Path, version: int) -> None:
    """Write a minimal active ``policy.json`` with the given ``version``.

    The schema matches what :class:`PolicyStore._load_from_disk`
    expects: ``schema_version``, ``version``, ``status``, plus all
    the bounded numeric tunables. ``status="active"`` so that what
    comes back from ``store.get()`` reflects the on-disk state
    without any promotion logic interfering.
    """
    body = {
        "schema_version": "1",
        "version": version,
        "created_at": "2026-07-15T00:00:00Z",
        "parent_version": None,
        "status": "active",
        "dense_k": 20,
        "lexical_k": 40,
        "rrf_k": 60,
        "fallback_min_results": 5,
        "rrf_dense_weight": 1.0,
        "rrf_lexical_weight": 1.0,
        "final_rrf_weight": 0.6,
        "final_vector_weight": 0.2,
        "final_lexical_weight": 0.2,
        "importance_weight": 0.55,
        "recency_weight": 0.25,
        "feedback_weight": 0.20,
        "exact_match_boost": 0.15,
    }
    body["checksum"] = _checksum_for(body)
    policy_path.parent.mkdir(parents=True, exist_ok=True)
    policy_path.write_text(json.dumps(body, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _checksum_for(body: dict) -> str:
    """Replicate :func:`compute_checksum` without importing it.

    The runner module already imports ``PolicyStore``; we do the
    same import here so we get the canonical hashing implementation
    rather than re-implementing it (which would silently rot if the
    canonical form ever changed).
    """
    from openclaw_memory_os.policy_store import Policy, compute_checksum

    candidate = {k: v for k, v in body.items() if k != "checksum"}
    return compute_checksum(Policy(**candidate))


# ---------------------------------------------------------------------------
# Test 1: static source-level guard
# ---------------------------------------------------------------------------


def test_runner_does_not_call_store_set_with_baseline():
    """The runner source must NOT contain ``store.set(Policy(**baseline_policy``.

    Three variants of that exact call existed before the fix (one
    per branch in ``_build_backend_and_store``). Any one of them
    would re-introduce the G6.10 violation, so we forbid all three
    by string-substring check. Docstring / comment text mentioning
    the forbidden pattern is acceptable — only the executable
    lines are the problem.
    """
    src = SCRIPT_PATH.read_text(encoding="utf-8")
    forbidden_substrings = [
        "store.set(Policy(**baseline_policy",
        "store.set(Policy(**baseline_policy, status=\"active\"",
    ]
    leaked = [needle for needle in forbidden_substrings if needle in src]
    assert not leaked, (
        "scripts/run_evolution_cycle.py still contains forbidden "
        "store.set(Policy(**baseline_policy, ...)) call(s): "
        f"{leaked!r}. G6.10 forbids the runner from overwriting the "
        "on-disk active policy with the shipped baseline."
    )


# ---------------------------------------------------------------------------
# Test 2: QdrantBackend branch preserves an active policy on disk
# ---------------------------------------------------------------------------


def test_runner_qdrant_branch_preserves_active_policy(monkeypatch, tmp_path):
    """QDRANT_URL branch must NOT clobber an on-disk active policy v42.

    We pre-write a ``policy.json`` with ``version=42`` and a fake
    ``QDRANT_URL=memory://mock``. ``_build_backend_and_store`` will:

    1. See ``MEMORY_OS_POLICY_PATH`` (set to our tmp file) and
       construct ``PolicyStore(path=...)`` which loads ``v42`` off
       disk.
    2. See ``QDRANT_URL`` and instantiate ``QdrantBackend`` — but
       we mock that class at the ``openclaw_memory_os.backends``
       module boundary so no real Qdrant client is contacted.
    3. Return ``(backend, store, None)``.

    Pre-fix behaviour: the runner called
    ``store.set(Policy(**baseline_policy, status="active", version=1))``
    which would overwrite the on-disk ``v42`` policy in-memory
    (and would persist it on next ``save``). Post-fix: the store
    returned must still report ``version == 42``.
    """
    policy_path = tmp_path / "policy.json"
    _write_active_policy(policy_path, version=42)

    monkeypatch.setenv("MEMORY_OS_POLICY_PATH", str(policy_path))
    monkeypatch.setenv("QDRANT_URL", "memory://mock-qdrant")
    monkeypatch.delenv("MEMORY_OS_SAMPLE_PATH", raising=False)

    # Mock the backend classes so the runner's ``QdrantBackend(url, ...)``
    # call does not try to instantiate a real ``QdrantClient`` (which
    # would DNS-resolve the fake ``memory://`` URL and crash).
    mock_qdrant = MagicMock(name="QdrantBackend")
    mock_qdrant.return_value = MagicMock(name="qdrant_instance")
    monkeypatch.setattr(
        "openclaw_memory_os.backends.QdrantBackend", mock_qdrant
    )
    mock_sample = MagicMock(name="SampleBackend")
    mock_sample.return_value = MagicMock(name="sample_instance")
    monkeypatch.setattr(
        "openclaw_memory_os.backends.SampleBackend", mock_sample
    )

    runner = _load_runner_module()
    backend, store, sample_path = runner._build_backend_and_store()

    # Return-shape contract: (backend, store, sample_path_or_None).
    assert sample_path is None, (
        "QDRANT_URL branch must not return a sample_path"
    )
    assert backend is mock_qdrant.return_value, (
        "QDRANT_URL branch should have constructed the mocked QdrantBackend"
    )
    # QdrantBackend.__init__ must have been called exactly once with
    # our fake URL — i.e. the runner took the QdrantBackend branch.
    assert mock_qdrant.call_count == 1
    assert mock_qdrant.call_args.args[0] == "memory://mock-qdrant"

    # The whole point: the on-disk active policy v42 must still be active.
    active = store.get()
    assert active.version == 42, (
        f"runner overwrote the on-disk active policy: "
        f"expected version=42, got version={active.version} "
        f"(status={active.status!r})"
    )
    assert active.status.value == "active"

    # And the runner must NOT have called ``store.set`` at all. If
    # it had, the PolicyStore's ``_previous`` slot would be populated
    # with the prior active (the v42 we just loaded), and the
    # ``_policy`` slot would have been replaced with the baseline v1.
    assert store.get_previous() is None, (
        "runner must not have called store.set() — _previous slot "
        "should still be None on a first-touch load."
    )


# ---------------------------------------------------------------------------
# Test 3: SampleBackend branch preserves an active policy on disk
# ---------------------------------------------------------------------------


def test_runner_sample_branch_preserves_active_policy(monkeypatch, tmp_path):
    """SampleBackend branch must NOT clobber an on-disk active policy v42.

    Symmetric to the Qdrant test, but this time we leave
    ``QDRANT_URL`` unset and set ``MEMORY_OS_SAMPLE_PATH`` to a
    tiny tmp JSON so the runner takes the SampleBackend branch.
    The pre-fix code called
    ``store.set(Policy(**baseline_policy, ...))`` here too —
    same G6.10 violation, same fix.
    """
    policy_path = tmp_path / "policy.json"
    _write_active_policy(policy_path, version=42)

    sample_path = tmp_path / "sample.json"
    sample_path.write_text(json.dumps({"memories": []}), encoding="utf-8")

    monkeypatch.setenv("MEMORY_OS_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.setenv("MEMORY_OS_SAMPLE_PATH", str(sample_path))

    mock_qdrant = MagicMock(name="QdrantBackend")
    monkeypatch.setattr(
        "openclaw_memory_os.backends.QdrantBackend", mock_qdrant
    )
    mock_sample = MagicMock(name="SampleBackend")
    mock_sample.return_value = MagicMock(name="sample_instance")
    monkeypatch.setattr(
        "openclaw_memory_os.backends.SampleBackend", mock_sample
    )

    runner = _load_runner_module()
    backend, store, returned_sample_path = runner._build_backend_and_store()

    assert returned_sample_path == str(sample_path)
    assert backend is mock_sample.return_value
    assert mock_sample.call_count == 1
    assert mock_qdrant.call_count == 0, (
        "QDRANT_URL is unset; QdrantBackend must not be instantiated."
    )

    active = store.get()
    assert active.version == 42, (
        f"SampleBackend branch overwrote the active policy: "
        f"expected version=42, got version={active.version}"
    )
    assert store.get_previous() is None


# ---------------------------------------------------------------------------
# Test 4: missing policy file → default baseline, no .set() call
# ---------------------------------------------------------------------------


def test_runner_no_op_when_policy_dir_missing(monkeypatch, tmp_path):
    """If MEMORY_OS_POLICY_PATH points at a non-existent file, the runner
    must fall back to ``PolicyStore``'s in-memory baseline default.

    The ``PolicyStore.__init__`` already starts with
    ``self._policy = initial or Policy(**baseline_policy)`` — so an
    empty / missing on-disk state is handled *inside* the
    constructor. The runner therefore must NOT call ``.set()``
    itself; if it did, that would be a re-introduction of the
    forbidden baseline-reset on the first-ever run.

    We assert three things:

    * the store's active policy is the shipped baseline (``version == 1``),
    * ``store.get_previous()`` is ``None`` (i.e. no ``.set()`` happened),
    * the on-disk file still does not exist (the runner did not
      implicitly write a policy it was never asked to).
    """
    policy_path = tmp_path / "never_created.json"
    assert not policy_path.exists(), (
        "precondition: the tmp policy path must not exist yet"
    )

    monkeypatch.setenv("MEMORY_OS_POLICY_PATH", str(policy_path))
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("MEMORY_OS_SAMPLE_PATH", raising=False)

    mock_qdrant = MagicMock(name="QdrantBackend")
    monkeypatch.setattr(
        "openclaw_memory_os.backends.QdrantBackend", mock_qdrant
    )
    mock_sample = MagicMock(name="SampleBackend")
    mock_sample.return_value = MagicMock(name="sample_instance")
    monkeypatch.setattr(
        "openclaw_memory_os.backends.SampleBackend", mock_sample
    )

    runner = _load_runner_module()
    backend, store, sample_path = runner._build_backend_and_store()

    # Without QDRANT_URL and without MEMORY_OS_SAMPLE_PATH the runner
    # falls through to the empty-sample branch (creates a tiny
    # in-process JSON file under ``PROJECT_DIR/data/_empty_sample.json``
    # and feeds it to SampleBackend). The branch under test therefore
    # is the *third* one — and that's exactly where the third
    # ``store.set(Policy(**baseline_policy, ...))`` used to live.
    assert mock_sample.call_count == 1, (
        "runner should have constructed SampleBackend for the "
        "empty-sample fallback"
    )
    assert mock_qdrant.call_count == 0

    # The store must report the shipped baseline default.
    active = store.get()
    assert active.version == 1, (
        f"expected PolicyStore's default baseline (version=1), "
        f"got version={active.version}"
    )

    # And crucially — no ``store.set()`` call was made. If the runner
    # had called ``.set(Policy(**baseline_policy, status="active", version=1))``
    # the store's ``_previous`` slot would be populated with a
    # Policy (because ``set`` always saves the prior active as
    # _previous before installing the new one). A pristine default
    # store has ``_previous`` unset (``get_previous`` returns None).
    assert store.get_previous() is None, (
        "runner must not have called store.set() — that would be a "
        "G6.10 violation even on the first-ever run."
    )

    # The runner must NOT have created the policy file out of thin
    # air. If it had, that would mean an implicit ``store.save()``
    # (which we did not ask for) — and it would also mean the
    # runner is reaching for the on-disk location when it
    # shouldn't be. ``PolicyStore.__init__`` only *creates the
    # parent directory* — never the file itself.
    assert not policy_path.exists(), (
        f"runner implicitly created {policy_path}; the missing-file "
        "fallback path must not write a policy file on its own."
    )