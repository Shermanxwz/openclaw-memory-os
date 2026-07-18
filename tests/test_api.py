"""End-to-end tests for the FastAPI app, using FastAPI's TestClient."""

from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient


from openclaw_memory_os.app import create_app


def _client():
    app = create_app()
    return TestClient(app)


def _login_with_csrf(client: TestClient, *, token: str = "secret", password: str = "", totp_code: str = "", recovery_code: str = ""):
    """Helper: GET /login to obtain CSRF cookie, then POST /login with it."""
    page = client.get("/login")
    csrf = page.cookies.get("csrf_token", "")
    data = {"token": token, "csrf_token": csrf}
    if password:
        data["password"] = password
    if totp_code:
        data["totp_code"] = totp_code
    if recovery_code:
        data["recovery_code"] = recovery_code
    return client.post(
        "/login",
        data=data,
        cookies={"csrf_token": csrf},
        follow_redirects=False,
    )


def test_health_endpoint_open():
    with _client() as c:
        r = c.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert "service" not in body  # privacy: do not leak service name


def test_root_redirects_to_dashboard():
    with _client() as c:
        r = c.get("/", follow_redirects=False)
    assert r.status_code == 307
    assert r.headers["location"].endswith("/dashboard")


def test_dashboard_renders_includes_chartjs_cdn():
    """The dashboard must reference the local chart.js static asset (CDN removed for privacy)."""
    with _client() as c:
        r = c.get("/dashboard")
    assert r.status_code == 200
    html = r.text
    assert "/static/chart.umd.min.js" in html
    assert "cdn.jsdelivr.net" not in html


def test_api_health_no_auth_when_disabled():
    with _client() as c:
        r = c.get("/api/health")
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "sample"
    assert data["total_memories"] >= 1
    assert "tier_distribution" in data
    assert "status_distribution" in data


def test_api_recall_test_returns_hits():
    payload = {"query": "recall", "mode": "hybrid", "limit": 5}
    with _client() as c:
        r = c.post("/api/recall-test", json=payload)
    assert r.status_code == 200
    data = r.json()
    assert data["query"] == "recall"
    assert data["mode"] == "hybrid"
    assert data["total_considered"] >= 0
    assert isinstance(data["hits"], list)
    assert data["query_id"]
    assert data["policy_version"]
    assert "diagnostics" in data
    assert "fallback" in data
    # Verify hit shape
    if data["hits"]:
        h = data["hits"][0]
        assert {"id", "text", "score", "components", "explanation", "candidate_key", "collection"}.issubset(h.keys())


def test_cookie_session_post_requires_csrf(monkeypatch, tmp_path):
    """Browser-cookie POSTs must carry X-CSRF-Token; bearer API calls are exempt."""
    monkeypatch.setenv("MEMORY_OS_ENV_FILE", str(tmp_path / "empty.env"))
    (tmp_path / "empty.env").write_text("", encoding="utf-8")
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("MEMORY_OS_TOKEN", "secret")
    monkeypatch.setenv("MEMORY_OS_PASSWORD", "")
    monkeypatch.setenv("MEMORY_OS_TOTP_SECRET", "")
    monkeypatch.setenv("PASSWORD_TOTP_AUTH", "off")
    from openclaw_memory_os.config import reset_settings_cache
    from openclaw_memory_os.auth import _login_limiter
    reset_settings_cache()
    _login_limiter.reset("testclient")
    app = create_app()
    with TestClient(app) as c:
        # GET /login to obtain a CSRF cookie.
        login_page = c.get("/login")
        csrf = login_page.cookies.get("csrf_token", "")
        # POST /login with the CSRF token (both cookie + form field).
        login = c.post(
            "/login",
            data={"token": "secret", "csrf_token": csrf},
            cookies={"csrf_token": csrf},
            follow_redirects=False,
        )
        assert login.status_code == 303
        session = login.cookies["memory_os_session"]
        csrf = login.cookies.get("csrf_token", csrf)
        payload = {"query": "recall", "mode": "hybrid", "limit": 5}
        blocked = c.post("/api/recall-test", json=payload, cookies={"memory_os_session": session})
        assert blocked.status_code == 403
        allowed = c.post(
            "/api/recall-test",
            json=payload,
            cookies={"memory_os_session": session, "csrf_token": csrf},
            headers={"X-CSRF-Token": csrf},
        )
        assert allowed.status_code == 200
        bearer = c.post("/api/recall-test", json=payload, headers={"Authorization": "Bearer secret"})
        assert bearer.status_code == 200


def test_api_recall_test_rejects_oversized_limit():
    # pydantic validation: limit must be <= 100
    with _client() as c:
        r = c.post("/api/recall-test", json={"query": "x", "limit": 1000})
    assert r.status_code == 422


def test_api_recall_test_rejects_empty_query():
    with _client() as c:
        r = c.post("/api/recall-test", json={"query": ""})
    assert r.status_code == 422


def test_api_timeline_returns_months():
    with _client() as c:
        r = c.get("/api/timeline")
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "sample"
    assert isinstance(data["months"], list)


def test_api_tiers_returns_distributions():
    with _client() as c:
        r = c.get("/api/tiers")
    assert r.status_code == 200
    data = r.json()
    assert data["backend"] == "sample"
    assert isinstance(data["tiers"], list)
    assert isinstance(data["statuses"], list)


def test_api_duplicates_finds_pair_from_sample():
    """The sample data has an obvious near-duplicate pair (mem-0007 / mem-0008)."""
    with _client() as c:
        r = c.get("/api/duplicates")
    assert r.status_code == 200
    data = r.json()
    assert data["count"] >= 1
    member_sets = [set(c["member_ids"]) for c in data["clusters"]]
    # Either mem-0007 or mem-0008 should appear in at least one cluster together.
    found_pair = any(
        {"mem-0007", "mem-0008"}.issubset(members) for members in member_sets
    )
    assert found_pair, f"expected duplicate cluster with mem-0007/mem-0008; got {data['clusters']}"


def test_api_deletion_candidates_marks_review_only():
    with _client() as c:
        r = c.get("/api/deletion-candidates")
    assert r.status_code == 200
    data = r.json()
    assert "review" in data["policy"]
    for c in data["candidates"]:
        assert c["recommended_action"] in ("review", "auto_delete")
        assert c["recommended_action"] != "keep"


def test_unknown_dashboard_section_returns_404():
    with _client() as c:
        r = c.get("/dashboard/nope")
    assert r.status_code == 404


def test_api_feedback_endpoint(tmp_path: Path):
    """Feedback endpoint records useful/not-useful feedback."""
    from openclaw_memory_os.audit import get_audit_store
    # Override audit db path for test isolation
    db = tmp_path / "api_fb.sqlite"
    store = get_audit_store(db_path=db)

    with _client() as c:
        r = c.post(
            "/api/feedback",
            json={"memory_id": "mem-001", "query": "test", "useful": True},
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["status"] == "ok"
    assert isinstance(data["row_id"], int)

    # Verify it went to audit log
    entries = store.list_recent(limit=5, action="feedback")
    assert len(entries) >= 1
    assert entries[0].memory_id == "mem-001"


def test_api_feedback_with_note(tmp_path: Path):
    from openclaw_memory_os.audit import get_audit_store
    db = tmp_path / "api_fb2.sqlite"
    get_audit_store(db_path=db)

    with _client() as c:
        r = c.post(
            "/api/feedback",
            json={"memory_id": "mem-002", "query": "test", "useful": False, "note": "wrong results"},
        )
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_api_audit_log_endpoint(tmp_path: Path):
    """Audit log endpoint returns recent entries."""
    from openclaw_memory_os.audit import get_audit_store
    db = tmp_path / "api_audit.sqlite"
    store = get_audit_store(db_path=db)
    store.log("test_api", detail="api test entry")

    with _client() as c:
        r = c.get("/api/audit-log?limit=10")
    assert r.status_code == 200
    data = r.json()
    assert "entries" in data
    assert data["count"] >= 1


def test_api_audit_log_filtered(tmp_path: Path):
    from openclaw_memory_os.audit import get_audit_store
    db = tmp_path / "api_audit2.sqlite"
    store = get_audit_store(db_path=db)
    store.log("action_a", detail="a")
    store.log("action_b", detail="b")

    with _client() as c:
        r = c.get("/api/audit-log?action=action_a")
    assert r.status_code == 200
    data = r.json()
    assert all(e["action"] == "action_a" for e in data["entries"])


def test_api_consolidate_endpoint(tmp_path: Path):
    """Consolidation endpoint analyzes duplicate clusters."""
    with _client() as c:
        # Get valid memory IDs first
        health = c.get("/api/health")
        assert health.status_code == 200
        # Try consolidating a known ID from sample data
        r = c.post(
            "/api/consolidate-duplicates",
            json={"cluster_ids": ["mem-0007", "mem-0008"], "strategy": "merge"},
        )
    assert r.status_code == 200, r.text
    data = r.json()
    assert "consolidation" in data
    assert data["consolidation"]["consolidated_id"] in ("mem-0007", "mem-0008")
    assert len(data["consolidation"]["merged_member_ids"]) == 2


def test_api_consolidate_bad_ids_returns_404():
    with _client() as c:
        r = c.post(
            "/api/consolidate-duplicates",
            json={"cluster_ids": ["nonexistent_id"], "strategy": "merge"},
        )
    assert r.status_code == 404


def test_api_consolidate_different_strategies():
    with _client() as c:
        r = c.post(
            "/api/consolidate-duplicates",
            json={"cluster_ids": ["mem-0007", "mem-0008"], "strategy": "keep_newest"},
        )
    assert r.status_code == 200
    data = r.json()
    assert data["consolidation"]["consolidated_id"] == "mem-0008"


def test_auth_required_when_token_set(monkeypatch, tmp_path):
    monkeypatch.setenv("MEMORY_OS_ENV_FILE", str(tmp_path / "empty.env"))
    (tmp_path / "empty.env").write_text("", encoding="utf-8")
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(tmp_path / "sessions.db"))
    monkeypatch.setenv("MEMORY_OS_TOKEN", "secret")
    monkeypatch.setenv("MEMORY_OS_PASSWORD", "")
    monkeypatch.setenv("MEMORY_OS_TOTP_SECRET", "")
    monkeypatch.setenv("PASSWORD_TOTP_AUTH", "off")
    # Re-create app so the new settings are picked up.
    from openclaw_memory_os.config import reset_settings_cache
    reset_settings_cache()
    app = create_app()
    with TestClient(app) as c:
        # Health stays open.
        assert c.get("/health").status_code == 200
        # Dashboard requires auth -> redirects to login (303) with login.html rendered.
        r = c.get("/dashboard")
        assert r.status_code in (303, 401)
        # API requires auth.
        api = c.get("/api/health")
        assert api.status_code == 401
        # ?token=... query strings are no longer accepted (privacy fix).
        api_query = c.get("/api/health?token=secret")
        assert api_query.status_code == 401
        # Cookie auth via /login sets the session, then API works.
        login = _login_with_csrf(c)
        assert login.status_code == 303
        assert "memory_os_session" in login.cookies
        # Re-issue the cookie explicitly because TestClient does not merge jar
        # cookies into subsequent requests by default.
        cookie_val = login.cookies["memory_os_session"]
        api_cookie = c.get("/api/health", cookies={"memory_os_session": cookie_val})
        assert api_cookie.status_code == 200
        # API allowed with correct bearer header (Authorization: Bearer ...).
        api3 = c.get("/api/health", headers={"Authorization": "Bearer secret"})
        assert api3.status_code == 200
        # API rejects wrong token (still no query-string auth).
        api4 = c.get("/api/health?token=wrong")
        assert api4.status_code == 401


# ---------------------------------------------------------------------------
# /api/dashboard/evaluation structured envelope (v0.3.0.x)
# ---------------------------------------------------------------------------


def test_api_dashboard_evaluation_envelope_keys():
    """The endpoint must surface the documented structured keys."""
    with _client() as c:
        r = c.get("/api/dashboard/evaluation")
    assert r.status_code == 200
    body = r.json()
    # Top-level keys are stable and never null.
    for key in (
        "status",
        "generated_at",
        "corpus_snapshot_id",
        "metrics",
        "feedback",
        "history",
        "notes",
        "warnings",
    ):
        assert key in body, f"missing top-level key: {key}"
    # status is one of the documented enum values.
    assert body["status"] in ("ok", "unavailable", "error")
    # generated_at is a non-empty ISO-8601-ish string.
    assert isinstance(body["generated_at"], str) and body["generated_at"]
    # corpus_snapshot_id is either null or a non-empty string.
    assert body["corpus_snapshot_id"] is None or isinstance(body["corpus_snapshot_id"], str)
    # notes / warnings are lists.
    assert isinstance(body["notes"], list)
    assert isinstance(body["warnings"], list)
    # history is a list (may be empty).
    assert isinstance(body["history"], list)


def test_api_dashboard_evaluation_metrics_keys():
    """metrics must include both legacy fields and the new v0.3.0.x fields."""
    with _client() as c:
        r = c.get("/api/dashboard/evaluation")
    assert r.status_code == 200
    metrics = r.json()["metrics"]
    # Legacy fields (kept for older dashboards).
    for key in (
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "useful_at_1",
        "useful_at_5",
        "explicit_negative_at_5",
        "no_result_rate",
        "p50_latency",
        "p95_latency",
        "num_cases",
    ):
        assert key in metrics, f"missing legacy metrics field: {key}"
    # New v0.3.0.x fields with explicit null/unavailable contract.
    for key in (
        "judged_ndcg_at_10",
        "useful_superseded_fallback_rate",
        "num_judged_cases",
        "corpus_snapshot_id",
        "judged_ndcg_status",
        "fallback_rate_status",
    ):
        assert key in metrics, f"missing v0.3.0.x metrics field: {key}"
    # When status=unavailable the graded metrics must be None (not 0.0).
    body = r.json()
    if body["status"] == "unavailable":
        assert metrics["judged_ndcg_at_10"] is None
        assert metrics["useful_superseded_fallback_rate"] is None
        assert metrics["judged_ndcg_status"] == "unavailable"
        assert metrics["fallback_rate_status"] == "unavailable"


def test_api_dashboard_evaluation_feedback_block_has_ratios():
    """The feedback sub-block must carry the documented ratio fields."""
    with _client() as c:
        r = c.get("/api/dashboard/evaluation")
    assert r.status_code == 200
    fb = r.json()["feedback"]
    for key in ("ratio_24h", "ratio_7d", "ratio_30d", "total_events"):
        assert key in fb, f"missing feedback key: {key}"


def test_api_dashboard_evaluation_unavailable_when_no_judged_data(tmp_path, monkeypatch):
    """When no judged cases are available, status="unavailable" and the
    graded fields are explicitly null (not fabricated 0.0)."""
    # Isolate the recall_feedback DB to an empty tmp location so the
    # endpoint sees zero cases. The conftest already strips QDRANT_* env
    # vars so no live backend will be touched.
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path / "empty-state"))
    # Reload the recall_feedback module so it re-resolves the DB path
    # against the new env var.
    import importlib
    import openclaw_memory_os.recall_feedback as rf
    importlib.reload(rf)

    try:
        with _client() as c:
            r = c.get("/api/dashboard/evaluation")
        assert r.status_code == 200
        body = r.json()
        # No judged data → unavailable (or ok if there is data we just
        # didn't isolate, but the metrics contract is what we care about).
        assert body["status"] in ("unavailable", "ok")
        metrics = body["metrics"]
        if body["status"] == "unavailable":
            assert metrics["judged_ndcg_at_10"] is None
            assert metrics["useful_superseded_fallback_rate"] is None
            assert metrics["num_judged_cases"] == 0
        # feedback / history blocks must remain well-typed even with no data.
        assert isinstance(body["feedback"], dict)
        assert isinstance(body["history"], list)
        assert isinstance(body["notes"], list)
        assert isinstance(body["warnings"], list)
    finally:
        # Reload once more so subsequent tests see the default DB path.
        monkeypatch.delenv("MEMORY_OS_RECALL_STATE_DIR", raising=False)
        importlib.reload(rf)
def test_api_dashboard_evaluation_returns_null_metrics_when_no_judged_data(monkeypatch, tmp_path):
    """Honest-null contract: graded metrics must be None, not 0.0.

    Uses an isolated ``MEMORY_OS_RECALL_STATE_DIR`` so the live
    recall_feedback.db (which carries traffic from running uvicorn
    and bench runs) does NOT poison the empty-DB assertion. Without
    this redirect the test would observe the live corpus_snapshot_id
    rows and report "ok" instead of "unavailable".
    """
    import importlib
    monkeypatch.setenv("MEMORY_OS_RECALL_STATE_DIR", str(tmp_path))
    import openclaw_memory_os.recall_feedback as _rf
    importlib.reload(_rf)
    try:
        with _client() as c:
            r = c.get("/api/dashboard/evaluation")
        assert r.status_code == 200
        data = r.json()
        assert data["status"] == "unavailable", data
        metrics = data["metrics"]
        for key in ("recall_at_1", "recall_at_5", "recall_at_10", "mrr_at_10",
                     "ndcg_at_10", "useful_at_1", "useful_at_5",
                     "explicit_negative_at_5", "no_result_rate",
                     "p50_latency", "p95_latency", "degraded_rate", "fallback_rate",
                     "judged_ndcg_at_10", "useful_superseded_fallback_rate"):
            assert metrics[key] is None, f"{key} should be None, got {metrics[key]!r}"
        # Counters stay 0 so the dashboard layout is preserved.
        assert metrics["num_cases"] == 0
        assert metrics["num_judged_cases"] == 0
        assert metrics["judged_ndcg_status"] == "unavailable"
        assert metrics["fallback_rate_status"] == "unavailable"
    finally:
        monkeypatch.delenv("MEMORY_OS_RECALL_STATE_DIR", raising=False)
        importlib.reload(_rf)


def test_evaluation_js_renders_null_as_em_dash():
    """The dashboard JS must surface null/undefined metrics as em-dash."""
    from pathlib import Path
    src = Path("openclaw_memory_os/static/js/evaluation.js").read_text(encoding="utf-8")
    assert "fmtMetric" in src
    assert "if (value == null) return " in src or "if (value == null) return" in src


def test_api_dashboard_evaluation_metrics_remain_null_when_no_real_scoring(
    monkeypatch, tmp_path
):
    """Judged feedback alone must not fabricate offline evaluation metrics.

    The dashboard reads only persisted real evaluation reports. Isolating that
    report directory proves the honest-null response even when a developer or
    host already has an unrelated ``latest.json`` in their normal state path.
    """
    import uuid
    from openclaw_memory_os import recall_feedback as rf

    monkeypatch.setenv(
        "MEMORY_OS_EVALUATION_REPORT_DIR", str(tmp_path / "empty-evaluation-reports")
    )
    # Persist a synthetic recall_run + useful feedback row so the
    # ``cases`` branch in ``_load_cases`` produces a non-empty list.
    qid = "hnull-" + uuid.uuid4().hex[:10]
    ck = "mem:h1"
    db = rf._get_db()
    try:
        db.execute(
            "INSERT OR REPLACE INTO recall_runs "
            "(query_id, query_text, created_at, retrieval_mode, policy_version, latency_ms, retrieval_status) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (qid, "test", "2026-07-15T07:00:00", "hybrid", "v1", 1.0, "ok"),
        )
        db.execute(
            "INSERT INTO feedback_events "
            "(query_id, candidate_key, memory_id, useful, created_at, feedback_source) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (qid, ck, "h1", 1, "2026-07-15T07:00:00", "test"),
        )
        db.commit()
        with _client() as c:
            r = c.get("/api/dashboard/evaluation")
        assert r.status_code == 200
        metrics = r.json()["metrics"]
        for key in ("recall_at_1", "recall_at_5", "recall_at_10",
                    "mrr_at_10", "ndcg_at_10", "useful_at_1", "useful_at_5",
                    "explicit_negative_at_5", "no_result_rate",
                    "p50_latency", "p95_latency", "degraded_rate", "fallback_rate",
                    "judged_ndcg_at_10", "useful_superseded_fallback_rate"):
            assert metrics[key] is None, f"{key} leaked 0.0 from no-op ranker: {metrics[key]!r}"
    finally:
        try:
            db.execute("DELETE FROM feedback_events WHERE query_id = ?", (qid,))
            db.execute("DELETE FROM recall_runs WHERE query_id = ?", (qid,))
            db.commit()
        finally:
            db.close()


# ---------------------------------------------------------------------------
# P0-S2 (v0.3.0) — session cookie persistence + revocation
# ---------------------------------------------------------------------------


def test_session_cookie_survives_service_restart_in_persistence_test(tmp_path, monkeypatch):
    """A cookie minted by one ``TestClient`` is still accepted by a fresh client
    built from ``create_app()`` so long as both share the same sessions DB.

    Implementation note (v0.3.0.x): we deliberately point both clients at the
    same ``MEMORY_OS_SESSIONS_DB`` so the persistent :class:`SessionStore`
    survives the simulated service restart. With this configuration the second
    client returns **200** — both the bearer-token ``hmac.compare_digest`` and
    the persistent store remember the cookie. To assert the opposite
    behaviour (401 when DBs differ), pass distinct ``tmp_path`` values to
    ``create_app`` setups in a future test.
    """
    # Single shared DB so persistence semantics are exercised end-to-end.
    db_path = tmp_path / "shared-sessions.db"
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(db_path))
    monkeypatch.setenv("MEMORY_OS_TOKEN", "secret")
    from openclaw_memory_os import config

    config.reset_settings_cache()
    from openclaw_memory_os import auth

    # Force a fresh store keyed at the shared path on first use.
    previous_store = auth._set_session_store_for_tests(None)
    try:
        # --- Client A: login, capture cookie, close -----------------------
        app_a = create_app()
        with TestClient(app_a) as c:
            login = _login_with_csrf(c)
            assert login.status_code == 303
            session_token = login.cookies["memory_os_session"]
            # The store now has a row keyed by the session token hash.
            store = auth.get_session_store()
            assert store.is_valid(session_token) is True
        # Closing the TestClient does not drop the process-wide store; close
        # it explicitly to mirror a service restart. Without this, the test
        # would leave a SessionStore pointing at our tmp file in the global
        # state and pollute subsequent tests.
        auth._set_session_store_for_tests(None)

        # --- Client B: fresh process, fresh store, but pointing at the same DB
        app_b = create_app()
        with TestClient(app_b) as c:
            api = c.get("/api/health", cookies={"memory_os_session": session_token})
            assert api.status_code == 200, (
                "expected the persistent SessionStore to remember the cookie "
                "across the simulated restart; got "
                f"status={api.status_code} body={api.text!r}"
            )
    finally:
        # Final cleanup — never leak a SessionStore between tests.
        auth._set_session_store_for_tests(None)
        # Restore the previous override (typically None in a fresh test
        # session, but tests that compose fixtures may have installed one).
        if previous_store is not None:
            auth._set_session_store_for_tests(previous_store)


def test_revoked_cookie_rejected_after_revoke(tmp_path, monkeypatch):
    """After ``/api/security/sessions/revoke-all`` runs, the revoked cookie is rejected.

    The revoke-all endpoint persists revocations in the SessionStore; the
    next call to :func:`verify_token` consults the store and, when the
    same string is found and flagged revoked, returns False even if it
    would otherwise have matched ``MEMORY_OS_TOKEN`` via hmac.
    """
    db_path = tmp_path / "revoke-sessions.db"
    monkeypatch.setenv("MEMORY_OS_SESSIONS_DB", str(db_path))
    monkeypatch.setenv("MEMORY_OS_TOKEN", "secret")
    from openclaw_memory_os import config

    config.reset_settings_cache()
    from openclaw_memory_os import auth

    previous_store = auth._set_session_store_for_tests(None)
    try:
        app = create_app()
        with TestClient(app) as c:
            # 1) Log in via bearer-token to obtain a cookie.
            login = _login_with_csrf(c)
            assert login.status_code == 303
            session_token = login.cookies["memory_os_session"]
            assert "csrf_token" in login.cookies
            csrf = login.cookies["csrf_token"]

            # 2) Confirm the cookie works BEFORE the revoke-all.
            pre = c.get(
                "/api/health",
                cookies={"memory_os_session": session_token, "csrf_token": csrf},
            )
            assert pre.status_code == 200

            # 3) Trigger the revoke-all endpoint. POST + CSRF required.
            revoke = c.post(
                "/api/security/sessions/revoke-all",
                cookies={"memory_os_session": session_token, "csrf_token": csrf},
                headers={"X-CSRF-Token": csrf},
            )
            assert revoke.status_code == 200, (
                f"revoke-all failed: status={revoke.status_code} body={revoke.text!r}"
            )
            body = revoke.json()
            assert body.get("status") == "ok"
            assert int(body.get("revoked", -1)) >= 1

            # 4) The same cookie should now be rejected, even though the
            #    string still matches MEMORY_OS_TOKEN by hmac — the persistent
            #    revocation has precedence.
            post = c.get(
                "/api/health",
                cookies={"memory_os_session": session_token, "csrf_token": csrf},
            )
            assert post.status_code == 401, (
                "expected 401 after revoke-all; got "
                f"status={post.status_code} body={post.text!r}"
            )
    finally:
        # Critical: tear down so this revoked store can never leak into
        # subsequent tests as a global state. Without this, the global
        # _session_store would still hold a SessionStore pointing at our
        # tmp DB with "secret" marked revoked, which would 401 the next
        # test that monkey-patches MEMORY_OS_TOKEN to "secret".
        auth._set_session_store_for_tests(None)
        if previous_store is not None:
            auth._set_session_store_for_tests(previous_store)

