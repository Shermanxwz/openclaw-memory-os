"""Web console v0.3.0 UI compliance checks.

These tests do NOT execute JS — they assert on the *contract* the
browser code can rely on:

  * ``/static/js/common.js`` exists and is wired into the page.
  * Every per-section module is loaded from the local path
    (``/static/js/<file>.js``); no CDN reference.
  * Chart.js is the local ``/static/chart.umd.min.js`` copy, never a
    ``cdn.jsdelivr.net`` reference.
  * The Recall form exposes the inputs the JS attaches handlers to.
  * The dashboard renders the right section containers / buttons for
    Strategy (live state + action buttons), Memories (read-only),
    Security (sessions table + revoke-all) and Governance
    (review-only copy, no "delete all" framing).
  * Server-side endpoints used by the new modules
    (``/api/dashboard/{strategy,evaluation,memories}``,
    ``/api/security/sessions``) respond with the expected shape.

Backend / recall_feedback / schema work is out of scope here.
"""

from __future__ import annotations

import os

import pytest
from fastapi.testclient import TestClient

from openclaw_memory_os.app import create_app


def _client():
    return TestClient(create_app())


REQUIRED_MODULES = (
    "common.js",
    "overview.js",
    "tiers_duplicates.js",
    "recall.js",
    "governance.js",
    "strategy.js",
    "evaluation.js",
    "memories.js",
    "security.js",
    "health.js",
    "dashboard.js",
)

# Sections that the server will render. Mirrors the allow-list in app.py.
SECTIONS = (
    "overview", "tiers", "duplicates", "recall", "governance",
    "strategy", "evaluation", "memories", "health", "security",
)

JS_FILE_DIR = "openclaw_memory_os/static/js"


def _dashboard_html(section: str) -> str:
    with _client() as c:
        r = c.get(f"/dashboard/{section}")
    assert r.status_code == 200, f"section={section!r} failed to render: {r.status_code}"
    return r.text


def _js(name: str) -> str:
    path = os.path.join(JS_FILE_DIR, name)
    with open(path, encoding="utf-8") as fh:
        return fh.read()


# ---------------------------------------------------------------------------
# 1. Static module inclusion + no-CDN contract
# ---------------------------------------------------------------------------


def test_dashboard_overview_includes_every_required_module():
    """One consolidated test: every JS module is on the page, in order."""
    html = _dashboard_html("overview")
    last = -1
    for module in REQUIRED_MODULES:
        needle = "/static/js/" + module
        idx = html.find(needle)
        assert idx > 0, f"missing local module: {needle!r}"
        assert idx > last, f"module {module!r} appeared before its dependency"
        last = idx


@pytest.mark.parametrize("module", REQUIRED_MODULES)
def test_dashboard_every_section_includes_required_module(module):
    """All sections reference every JS module (modules are shared)."""
    needle = "/static/js/" + module
    for section in SECTIONS:
        html = _dashboard_html(section)
        assert needle in html, f"section {section!r} missing module {module!r}"


def test_dashboard_has_no_cdn_script_reference():
    """No CDN reference leaks into the rendered dashboard HTML."""
    for section in SECTIONS:
        html = _dashboard_html(section)
        for forbidden in ("cdn.jsdelivr.net", "unpkg.com", "cdnjs.cloudflare.com", "https://cdn"):
            assert forbidden not in html, f"{forbidden!r} found in dashboard HTML ({section!r})"
        # Local chart.js is referenced as a static asset.
        assert "/static/chart.umd.min.js" in html


def test_dashboard_has_chart_loaded_once_with_defer():
    html = _dashboard_html("overview")
    # Chart loads before dashboard.js (which instantiates it).
    assert html.count("/static/chart.umd.min.js") == 1
    assert "<script src=\"/static/chart.umd.min.js\" defer></script>" in html


# ---------------------------------------------------------------------------
# 2. Per-section container / button contracts
# ---------------------------------------------------------------------------


def test_recall_page_has_form_inputs():
    html = _dashboard_html("recall")
    for needle in (
        'id="q"',
        'id="mode"',
        'id="limit"',
        'id="runBtn"',
        'id="recallMeta"',
        'id="recallHits"',
    ):
        assert needle in html, f"missing recall element: {needle!r}"


def test_strategy_page_has_live_state_block_and_action_container():
    html = _dashboard_html("strategy")
    for needle in (
        'id="strategy"',
        'id="strategy-state"',
        'data-strategy-state',
        'data-strategy-actions',
        '策略状态 (live)',
    ):
        assert needle in html, f"missing strategy element: {needle!r}"


def test_evaluation_page_renders_feedback_history_metric_blocks():
    html = _dashboard_html("evaluation")
    for needle in (
        'id="evaluation"',
        'id="evaluation-feedback"',
        'id="evaluation-metrics"',
        'id="evaluation-history"',
        'data-evaluation-block="feedback"',
        'data-evaluation-block="metrics"',
        'data-evaluation-block="history"',
    ):
        assert needle in html, f"missing evaluation element: {needle!r}"


def test_memories_page_marks_readonly_and_no_physical_delete_button():
    html = _dashboard_html("memories")
    for needle in (
        'id="memories"',
        'id="memories-meta"',
        'id="memories-list"',
        'data-memories-meta',
        'data-memories-list',
    ):
        assert needle in html, f"missing memories element: {needle!r}"
    # "read-only" wording must appear somewhere on the page.
    assert "只读" in html or "read-only" in html
    # No "一键删除" / "delete-all" framing — the OS is read-only.
    for forbidden in ("一键删除", "delete all", "Mark all delete", "Delete all", "Delete ALL"):
        assert forbidden not in html


def test_security_page_has_sessions_table_and_revoke_all_button():
    html = _dashboard_html("security")
    for needle in (
        'id="security-sessions"',
        'id="security-events"',
        'id="revokeAllBtn"',
        'data-security-block="sessions"',
        'data-security-block="events"',
        'data-security-action="revoke-all"',
    ):
        assert needle in html, f"missing security element: {needle!r}"


def test_governance_page_uses_review_only_language():
    html = _dashboard_html("governance")
    for needle in (
        'id="governance"',
        'id="delList"',
        'id="markAllDel"',
        'data-governance-action="mark-all"',
    ):
        assert needle in html, f"missing governance element: {needle!r}"
    # The page must NOT frame the bulk button as "delete all" — it
    # must use the review-only contract.
    assert "一键删除" not in html
    assert "Mark all delete" not in html
    # "review-only" or equivalent reads as the explicit contract.
    assert "review-only" in html.lower()


# ---------------------------------------------------------------------------
# 3. Endpoint contracts used by the new modules
# ---------------------------------------------------------------------------


def test_api_dashboard_strategy_returns_state_policy_checksum():
    with _client() as c:
        r = c.get("/api/dashboard/strategy")
    assert r.status_code == 200
    body = r.json()
    assert "policy" in body
    assert "state" in body
    assert "checksum" in body


def test_api_dashboard_evaluation_returns_feedback_block():
    with _client() as c:
        r = c.get("/api/dashboard/evaluation")
    assert r.status_code == 200
    body = r.json()
    assert "feedback" in body
    assert "metrics" in body
    assert "history" in body
    fb = body["feedback"]
    # The function returns ratios for 24h / 7d / 30d.
    for key in ("ratio_24h", "ratio_7d", "ratio_30d", "total_events"):
        assert key in fb, f"missing evaluation field: {key}"


def test_api_dashboard_memories_returns_collections_and_readonly_policy():
    with _client() as c:
        r = c.get("/api/dashboard/memories")
    assert r.status_code == 200
    body = r.json()
    assert "collections" in body
    assert "memories" in body
    assert "policy" in body
    assert "read-only" in body["policy"].lower()


def test_api_security_sessions_returns_sessions_and_events():
    with _client() as c:
        r = c.get("/api/security/sessions")
    assert r.status_code == 200
    body = r.json()
    assert "sessions" in body
    assert "events" in body


# ---------------------------------------------------------------------------
# 4. JavaScript module sanity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    ["common.js", "overview.js", "tiers_duplicates.js",
     "recall.js", "governance.js", "strategy.js",
     "evaluation.js", "memories.js", "security.js",
     "health.js", "dashboard.js"],
)
def test_static_js_files_have_balanced_brackets(name):
    """Cheap ASCII-only balance check: counted braces / parens / brackets."""
    source = _js(name)
    assert source.strip(), f"{name} is empty"
    assert source.count("{") == source.count("}"), f"{name} brace mismatch"
    assert source.count("(") == source.count(")"), f"{name} paren mismatch"
    assert source.count("[") == source.count("]"), f"{name} bracket mismatch"


def test_common_js_exposes_ocmemory_namespace():
    src = _js("common.js")
    # Required exports.
    for ident in (
        "getCookie", "csrfHeaders", "authHeaders", "getJSON",
        "postJSON", "escapeHTML", "el",
        "window.OCMemory",
        'CSRF_COOKIE:',
        '"csrf_token"',
    ):
        assert ident in src, f"common.js missing export: {ident!r}"


def test_recall_js_emits_data_query_id_and_candidate_key_attrs():
    src = _js("recall.js")
    assert "global.OCMemory.attachRecallForm" in src
    assert 'data-query-id' in src
    assert 'data-candidate-key' in src


def test_governance_js_keeps_review_only_actions_honest():
    src = _js("governance.js")
    # The governance surface is review-only: it must not imply physical
    # deletion, and it must not call an unrelated consolidation endpoint
    # just to make a button look stateful.
    assert "review-only" in src.lower() or "仅审核" in src
    assert "/api/consolidate-duplicates" not in src
    assert "/api/delete" not in src


def test_security_js_wires_revoke_all_button():
    src = _js("security.js")
    assert "revokeAllBtn" in src
    assert "/api/security/sessions/revoke-all" in src


def test_strategy_js_calls_evolution_endpoints_via_postJSON():
    src = _js("strategy.js")
    for path in (
        "/api/evolution/pause",
        "/api/evolution/resume",
        "/api/evolution/candidate/reject",
        "/api/evolution/rollback",
    ):
        assert path in src
    # Pause / resume / reject / rollback buttons must each appear.
    for label in ("strategy-pause-btn", "strategy-resume-btn",
                  "strategy-reject-btn", "strategy-rollback-btn"):
        assert label in src


def test_evaluation_js_renders_feedback_ratio_cards():
    src = _js("evaluation.js")
    # The module must render the ratio cards the HTML declares.
    for key in ("ratio_24h", "ratio_7d", "ratio_30d", "total_events"):
        assert key in src


def test_memories_js_groups_by_collection_and_marks_readonly():
    src = _js("memories.js")
    assert "collection" in src
    assert "read-only" in src


def test_health_js_updates_data_health_tiles():
    src = _js("health.js")
    for tile in ("qdrant", "ollama", "lexical", "policy", "feedback", "memoryos"):
        assert tile in src
