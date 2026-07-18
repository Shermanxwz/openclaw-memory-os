"""Shared pytest fixtures."""

from __future__ import annotations

from pathlib import Path

import pytest

# Ensure the project root is importable when pytest is run from the repo root.
import sys
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openclaw_memory_os import auth  # noqa: E402
from openclaw_memory_os.config import reset_settings_cache  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Isolate tests from real auth/qdrant state and the lexical cache.

    The auth/qdrant stripping keeps the API hermetic (no real Qdrant
    connection, no bearer-token leakage). The lexical-cache redirect
    is critical: without it, ``app.state`` lifespan eagerly loads a
    ~100MB lexical-index cache built from the real Qdrant collection,
    which makes every TestClient boot re-tokenize tens of thousands
    of CJK n-grams and drives memory to multi-GB while pytest hangs
    for minutes per file.

    We deliberately leave ``MEMORY_OS_RECALL_STATE_DIR`` pointing at
    the real feedback DB. Tests that exercise the evolution cycle
    need the real ``recall_feedback.db`` schema (``recall_runs`` /
    ``recall_results`` / ``feedback_events``) to be reachable, and
    redirecting it to a tmpdir would mask genuine schema bugs by
    making the cycle error out with ``no such table`` instead.
    """
    monkeypatch.delenv("MEMORY_OS_TOKEN", raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_COLLECTION", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    monkeypatch.delenv("MEMORY_OS_LEXICAL_CACHE_DIR", raising=False)
    monkeypatch.delenv("MEMORY_OS_SESSIONS_DB", raising=False)
    monkeypatch.setenv("MEMORY_OS_LEXICAL_CACHE_DIR", str(tmp_path / "lexical-index"))
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(tmp_path / "sessions.db"))
    auth._reset_auth_state_for_tests()
    reset_settings_cache()
    yield
    reset_settings_cache()
    auth._reset_auth_state_for_tests()
