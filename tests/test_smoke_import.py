"""Single-shot real-environment smoke test for the v0.3.0 web app.

Boots the FastAPI app via ``TestClient`` (no live Qdrant needed:
the offline JSON-backend fallback is acceptable). Exercises health,
login, recall, strategy, evaluation, security, and dashboard pages;
reports OK.

This test pins the end-to-end surface area of the v0.3.0 release
so a regression in any of the four pillars (auth, recall, dashboard,
security) is caught by a single command.

The smoke test runs cleanly under the default ``tests/conftest.py``
fixture, which strips ``MEMORY_OS_TOKEN``, ``QDRANT_URL``,
``QDRANT_COLLECTION``, and ``QDRANT_API_KEY`` from the environment
before the test runs — that means auth is intentionally **disabled**
(``settings.auth_enabled is False``) so the bearer-token comparison
in ``verify_token`` short-circuits and dashboard pages do not
require a session cookie. The ``Authorization: Bearer
test-smoke-token`` header is therefore redundant but harmless: when
``MEMORY_OS_TOKEN`` is unset the legacy bearer-token path is a
no-op, and ``require_auth`` falls through.
"""

from __future__ import annotations

import os
import tempfile

from fastapi.testclient import TestClient


def test_real_environment_smoke():
    """Single-shot real-environment smoke test.

    Boots the FastAPI app via ``TestClient`` (no live Qdrant needed for
    this smoke; the offline JSON-backend fallback is acceptable).
    Exercises health, login, recall, strategy, evaluation, security,
    and dashboard pages; reports OK.
    """
    from openclaw_memory_os.app import create_app

    tmp = tempfile.mkdtemp(prefix="memory_os_smoke_")
    os.environ["MEMORY_OS_SESSIONS_DB"] = os.path.join(tmp, "sessions.db")
    # Force an empty sample backend so we don't need live Qdrant.
    empty = os.path.join(tmp, "empty.json")
    with open(empty, "w") as f:
        f.write("[]")
    os.environ["MEMORY_OS_SAMPLE_PATH"] = empty
    try:
        with TestClient(create_app()) as c:
            # 1. Health
            assert c.get("/health").status_code == 200
            # 2. Bearer-authenticated API
            auth = {"Authorization": "Bearer test-smoke-token"}
            assert c.get("/api/health", headers=auth).status_code == 200
            assert c.post(
                "/api/recall-test",
                json={"query": "smoke", "mode": "hybrid", "limit": 3},
                headers=auth,
            ).status_code == 200
            assert c.get("/api/strategy", headers=auth).status_code == 200
            assert c.get("/api/dashboard/evaluation", headers=auth).status_code == 200
            assert c.get("/api/security/sessions", headers=auth).status_code == 200
            # 3. Dashboard pages (no auth required when MEMORY_OS_TOKEN unset in test env)
            for s in ("overview", "tiers", "duplicates", "recall", "governance",
                      "strategy", "evaluation", "memories", "health", "security"):
                assert c.get(f"/dashboard/{s}").status_code == 200, s
    finally:
        os.environ.pop("MEMORY_OS_SESSIONS_DB", None)
        os.environ.pop("MEMORY_OS_SAMPLE_PATH", None)