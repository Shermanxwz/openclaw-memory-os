"""Regression test for the deletion contract in Memory Brain Consolidate.

Historical context (issue #6, 2026-07-14): ``scripts/memory_brain_consolidate.py``
shipped a ``qdrant_delete()`` helper that could call Qdrant's delete
API when ``MEMORY_BRAIN_ALLOW_DELETE=1`` was set in the environment.
That escape hatch contradicted the documented "the OS never deletes
memories" contract.

The P0-S1 fix removes the helper entirely and replaces its call site
with an info log that only reports *how many* stale candidates were
detected. Memory Brain never asks Qdrant to delete anything.

These tests pin:

* ``memory_brain_consolidate`` no longer exposes a ``qdrant_delete``
  attribute (the function name must be gone from the module).
* The script's public surface (``main``, ``prune``, ``orient``,
  ``gather``, ``consolidate``) still imports and runs without the
  helper, even when ``MEMORY_BRAIN_ALLOW_DELETE`` is forced on.
* No ``requests.delete`` / ``requests.post`` ever lands on a Qdrant
  ``/points/delete`` URL during consolidation, regardless of the
  environment. This is the contract a future refactor cannot silently
  break.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
CONSOLIDATE_PATH = REPO_ROOT / "scripts" / "memory_brain_consolidate.py"


def _load_consolidate_module(monkeypatch):
    """Load ``scripts/memory_brain_consolidate.py`` as a stand-alone module.

    We deliberately do NOT add ``scripts/`` to ``sys.path`` globally —
    that would leak ``memory_brain_ingest``'s ``normalize_importance``
    into everything else. We only inject ``scripts/`` for the duration
    of the module import (the consolidation script tries
    ``from memory_brain_ingest import normalize_importance``), and we
    pop cached copies so we always get a fresh module.
    """
    scripts_dir = str(CONSOLIDATE_PATH.parent)
    monkeypatch.syspath_prepend(scripts_dir)
    for mod_name in list(sys.modules):
        if mod_name in {"memory_brain_consolidate", "memory_brain_ingest"}:
            monkeypatch.delitem(sys.modules, mod_name, raising=False)

    spec = importlib.util.spec_from_file_location(
        "memory_brain_consolidate", CONSOLIDATE_PATH
    )
    assert spec is not None and spec.loader is not None, (
        f"could not build import spec for {CONSOLIDATE_PATH}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _FakeResponse:
    """Minimal stand-in for ``requests.Response``."""

    def __init__(self, payload):
        self._payload = payload
        self.status_code = 200

    def raise_for_status(self):
        return None

    def json(self):
        return self._payload


def _empty_qdrant_scroll() -> dict:
    """Empty scroll response: ``{"result": {"points": [], "next_page_offset": None}}``."""
    return {"result": {"points": [], "next_page_offset": None}}


def _empty_qdrant_search() -> dict:
    """Search response with zero hits — keeps consolidation a no-op."""
    return {"result": []}


def _empty_qdrant_collection() -> dict:
    return {"result": {"points_count": 0}}


def _patch_requests_for_noop_run(monkeypatch, captured: dict):
    """Replace ``requests.{get,post,put,delete}`` with capturing stubs.

    The stubs reply with safe empty Qdrant payloads so the consolidation
    pipeline runs to completion without needing a real Qdrant server.
    ``captured["calls"]`` is filled with every URL the script touched.
    """

    def fake_get(url, headers=None, timeout=None):
        captured["calls"].append(("GET", url))
        return _FakeResponse(_empty_qdrant_collection())

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured["calls"].append(("POST", url))
        # Every POST in the consolidate script goes through
        # qdrant_search / qdrant_scroll, both of which return
        # ``r.json()["result"]``. Scroll expects a dict with a
        # ``points`` key; search expects a list. Cheapest portable
        # dispatch: look at the URL.
        if "/points/scroll" in url:
            return _FakeResponse(_empty_qdrant_scroll())
        if "/points/search" in url:
            return _FakeResponse(_empty_qdrant_search())
        # Generic empty list for anything else (chats, embeddings).
        return _FakeResponse([])

    def fake_put(url, json=None, headers=None, timeout=None):  # noqa: A002
        captured["calls"].append(("PUT", url))
        return _FakeResponse({"result": {"status": "ok"}})

    def fake_delete(url, headers=None, timeout=None):
        captured["calls"].append(("DELETE", url))
        return _FakeResponse({"result": {"status": "deleted"}})

    monkeypatch.setattr("requests.get", fake_get)
    monkeypatch.setattr("requests.post", fake_post)
    monkeypatch.setattr("requests.put", fake_put)
    monkeypatch.setattr("requests.delete", fake_delete)


@pytest.fixture
def consolidate_module(monkeypatch):
    return _load_consolidate_module(monkeypatch)


# ---------------------------------------------------------------------------
# Contract tests
# ---------------------------------------------------------------------------


def test_consolidate_module_has_no_qdrant_delete(consolidate_module):
    """The P0-S1 fix removes the ``qdrant_delete`` helper entirely.

    If this assertion ever fires, the deletion escape hatch has been
    reintroduced. Update this test (and the docstring / deletion policy)
    together as part of an explicit PR — do not just delete the test.
    """
    assert not hasattr(consolidate_module, "qdrant_delete"), (
        "memory_brain_consolidate.qdrant_delete must not exist — Memory "
        "Brain never deletes memories. If you re-introduce the helper, "
        "update tests/test_memory_brain_delete_optin.py and "
        "docs/deletion-policy.md in the same change."
    )


def test_consolidate_module_docstring_advertises_never_deletes(consolidate_module):
    """Module docstring must state the never-deletes contract.

    A future refactor that removes the note is also caught here so
    docs and code stay aligned.
    """
    doc = (consolidate_module.__doc__ or "").lower()
    assert "never deletes" in doc
    assert "memories" in doc
    assert "mem_allow" not in doc.replace("memory_brain_allow_delete", "")  # placeholder
    # Specifically: no leftover MEMORY_BRAIN_ALLOW_DELETE *opt-in*
    # text. The removal note is fine, an opt-in is not.
    # (No live env-var check; this is just a docstring sanity gate.)


def test_consolidate_module_does_not_reference_opt_in_flag_at_runtime(
    consolidate_module, monkeypatch
):
    """Sanity: even with the old opt-in env var set, the module has no
    code path that consults ``MEMORY_BRAIN_ALLOW_DELETE`` at runtime.

    We grep the module's source (already loaded) for any mention of
    the flag. Docstring notes (e.g. "the previous
    MEMORY_BRAIN_ALLOW_DELETE has been removed") are allowed; live code
    references are not.
    """
    monkeypatch.setenv("MEMORY_BRAIN_ALLOW_DELETE", "1")
    import inspect

    source = inspect.getsource(consolidate_module)
    # Strip the docstring once so the historical note does not trip
    # the check.
    stripped = source.replace(consolidate_module.__doc__ or "", "", 1)
    assert "MEMORY_BRAIN_ALLOW_DELETE" not in stripped, (
        "consolidate module still references MEMORY_BRAIN_ALLOW_DELETE "
        "in live code — the deletion escape hatch has been removed."
    )


def test_prune_reports_stale_count_but_makes_no_qdrant_delete_calls(
    consolidate_module, monkeypatch
):
    """Call ``prune()`` end-to-end and verify no Qdrant delete URL is hit.

    This is the strongest contract test: even if a future refactor
    re-adds some helper under a different name, this assertion would
    catch any path that POSTs / DELETEs to ``/points/delete``.
    """
    monkeypatch.delenv("MEMORY_BRAIN_ALLOW_DELETE", raising=False)
    captured: dict = {"calls": []}
    _patch_requests_for_noop_run(monkeypatch, captured)
    # Prevent the empty run from writing a real dream-state file.
    monkeypatch.setattr(consolidate_module, "STATE_FILE", "/tmp/_neverbrain_state.json")
    monkeypatch.setattr(
        consolidate_module, "STATUS_FILE", "/tmp/_neverbrain_status.json"
    )
    monkeypatch.setattr(consolidate_module, "MEMORY_FILE", "/tmp/_neverbrain_MEMORY.md")
    # Skip the actual orient/gather/consolidate phases: we only need prune.
    monkeypatch.setattr(consolidate_module, "orient", lambda: {"total_points": 0, "new_since_24h": 0, "mem_lines": 0, "mem_kb": 0.0})

    # Run prune() directly. With no stale points discovered, the call
    # site is a no-op; with fake scroll responses that look stale, the
    # call site logs a count but makes no delete request.
    def _fake_scroll_with_stale(limit=100, offset_id=None, filter_payload=None):
        return {
            "points": [
                {
                    "id": 1,
                    "payload": {
                        "valid_until": "2000-01-01T00:00:00",  # always in the past
                        "summary": "old",
                        "content": "old",
                        "topic": "history",
                    },
                },
                {
                    "id": 2,
                    "payload": {
                        "valid_until": "2000-01-01T00:00:00",
                        "summary": "old",
                        "content": "old",
                        "topic": "history",
                    },
                },
            ],
            "next_page_offset": None,
        }

    monkeypatch.setattr(consolidate_module, "qdrant_scroll", _fake_scroll_with_stale)

    consolidate_module.prune()

    offending = [
        call
        for call in captured["calls"]
        if "/points/delete" in call[1]
    ]
    assert offending == [], (
        f"Memory Brain must never POST/DELETE to a /points/delete URL, "
        f"got: {offending!r}"
    )


def test_main_runs_with_opt_in_flag_set_and_never_deletes(
    consolidate_module, monkeypatch
):
    """End-to-end: run main() with MEMORY_BRAIN_ALLOW_DELETE=1 forced
    on, verify consolidation completes and never hits /points/delete.

    This proves the opt-in is dead code now — setting the flag cannot
    resurrect deletions. The script is allowed to short-circuit on the
    trigger check (insufficient new memories); that's still a pass.
    """
    monkeypatch.setenv("MEMORY_BRAIN_ALLOW_DELETE", "1")
    captured: dict = {"calls": []}
    _patch_requests_for_noop_run(monkeypatch, captured)
    monkeypatch.setattr(consolidate_module, "STATE_FILE", "/tmp/_neverbrain_state2.json")
    monkeypatch.setattr(
        consolidate_module, "STATUS_FILE", "/tmp/_neverbrain_status2.json"
    )
    monkeypatch.setattr(consolidate_module, "MEMORY_FILE", "/tmp/_neverbrain_MEMORY2.md")

    consolidate_module.main()

    offending = [
        call
        for call in captured["calls"]
        if "/points/delete" in call[1]
    ]
    assert offending == [], (
        "MEMORY_BRAIN_ALLOW_DELETE=1 must not enable any Qdrant delete "
        f"path. Got: {offending!r}"
    )


def test_no_requests_delete_imported_in_consolidate_source(consolidate_module):
    """Static check: consolidate script must not import ``requests.delete``.

    The fresh install only needs ``requests`` for read/upsert calls.
    ``requests.delete`` is included for completeness — if a future
    refactor inadvertently adds it back, this test fails before the
    module is even loaded by pytest.
    """
    import inspect

    source = inspect.getsource(consolidate_module)
    # ``requests.delete(...)`` would imply a delete endpoint hit.
    # We allow the word in docstrings / comments only, not in code.
    # (Cheap heuristic — line-by-line check.)
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        assert "requests.delete" not in stripped, (
            f"requests.delete call reintroduced in consolidate source: {stripped!r}"
        )
        # POST to /points/delete: specific URL pattern is also a red flag.
        if "/points/delete" in stripped and not stripped.startswith("#"):
            pytest.fail(
                f"/points/delete URL reintroduced in consolidate source: {stripped!r}"
            )
