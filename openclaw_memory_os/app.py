"""FastAPI application for OpenClaw Memory OS.

Run locally with::

    uvicorn openclaw_memory_os.app:app --host 0.0.0.0 --port 7788

The app exposes a small set of JSON endpoints and a server-rendered
dashboard. All non-health endpoints require a bearer token when
``MEMORY_OS_TOKEN`` is set.
"""

from __future__ import annotations

import logging
import os
import secrets
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import Depends, FastAPI, Form, HTTPException, Request, Response, status
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .analytics import (
    _build_deletion_candidates as build_deletion_candidates,
    _read_autonomous_governance_status,
    _estimate_duplicate_clusters as estimate_duplicate_clusters,
    build_health_summary,
    monthly_counts,
    status_distribution,
    tier_distribution,
)
from .audit import get_audit_store
from .policy_store import PolicyStore
from .evolution import _load_evolution_state
from .auth import (
    CSRF_COOKIE_NAME,
    SESSION_COOKIE_NAME,
    attempt_login,
    clear_session_cookie,
    close_session_store,
    extract_token,
    generate_csrf_token,
    issue_csrf_cookie,
    require_auth,
    list_sessions,
    revoke_all_sessions,
    set_session_cookie,
    verify_csrf,
    verify_token,
)
from .auth import _login_limiter  # rate limiter singleton
from .backends import MemoryBackend, get_backend
from .config import Settings, get_settings, reset_settings_cache
from .consolidation import consolidate_cluster
from .models import (
    ConsolidationRequest,
    FeedbackEntry,
    RecallRequest,
    RecallResponse,
    ReclassifyRequest,
)
from .ranking import build_recall_response
from .retrieval_engine import RetrievalEngine, build_recall_response_v030

logger = logging.getLogger(__name__)

TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"


def _ensure_csrf(request: Request, response: Response) -> str:
    """Return the current CSRF token, setting the cookie on ``response`` if missing.

    The token is stored in a non-HttpOnly cookie so the page's JS can
    read it and echo it back via the ``X-CSRF-Token`` header on
    state-changing requests.
    """
    existing = request.cookies.get(CSRF_COOKIE_NAME)
    if existing:
        return existing
    return issue_csrf_cookie(response)


def _issue_csrf_token(request: Request) -> str:
    """Return the current CSRF token, or mint a fresh one (no response side-effect).

    Used by render paths that don't yet have a Response object. The
    cookie is attached on the eventual TemplateResponse; if the caller
    has a Response in hand, prefer :func:`_ensure_csrf` so the cookie
    is actually written.
    """
    existing = request.cookies.get(CSRF_COOKIE_NAME)
    if existing:
        return existing
    return generate_csrf_token()


def _build_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    templates = Jinja2Templates(directory=str(TEMPLATES_DIR))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Build the backend once at startup so the sample file is parsed eagerly.
        app.state.backend = get_backend(settings)
        app.state.settings = settings
        # Policy persistence is part of the serving contract. Permission or
        # recovery failures must stop startup rather than retrying the identical
        # constructor and pretending an in-memory fallback exists.
        app.state.policy_store = PolicyStore()
        logger.info("OpenClaw Memory OS started with backend=%s", app.state.backend.name)
        # Sync pre-warm: force cache load before accepting requests
        try:
            mems = app.state.backend.list_memories()
            logger.info("Memory cache pre-warmed (%d entries)", len(mems))
        except Exception as e:
            logger.warning("Cache warm failed: %s", e)
        # v0.3.0.x: BM25 lexical index — eagerly load from cache if
        # one exists so the first hybrid/keyword request avoids a
        # cold build. If no cache file is present (or it is
        # malformed), fall back to None and the lazy-load path in
        # the recall handler will build the index on first request.
        app.state.lexical_index = None
        app.state._lexical_cache_dir = None
        try:
            from .lexical import BM25Index
            _lexical_cache_dir = Path(
                os.environ.get(
                    "MEMORY_OS_LEXICAL_CACHE_DIR",
                    str(
                        Path(
                            os.environ.get(
                                "MEMORY_OS_RECALL_STATE_DIR",
                                os.environ.get(
                                    "XDG_STATE_HOME",
                                    os.path.expanduser("~/.local/state"),
                                ),
                            )
                        )
                        / "openclaw-memory-os"
                        / "lexical-index"
                    ),
                )
            )
            _idx = BM25Index.load(_lexical_cache_dir)
            if _idx is not None and len(_idx) > 0:
                app.state.lexical_index = _idx
                app.state._lexical_cache_dir = _lexical_cache_dir
                logger.info(
                    "BM25 lexical index eagerly loaded from cache (%d docs)",
                    len(_idx),
                )
        except Exception as exc:
            logger.debug("BM25 eager load skipped: %s", exc)
        yield
        # Persist the lexical index on shutdown so the next startup
        # can load it from cache instead of rebuilding.
        try:
            idx = getattr(app.state, "lexical_index", None)
            cache_dir = getattr(app.state, "_lexical_cache_dir", None)
            if idx is not None and cache_dir is not None:
                idx.save(cache_dir)
                logger.info("BM25 lexical index saved to cache on shutdown")
        except Exception:
            pass
        # Close the session store so the SQLite WAL is flushed cleanly
        # before the process exits. Without this, a killed process can
        # leave the WAL in an inconsistent state, causing session
        # persistence failures across restarts.
        try:
            close_session_store()
        except Exception:
            pass
        logger.info("OpenClaw Memory OS shutting down.")

    app = FastAPI(
        title="OpenClaw Memory OS",
        version="0.3.0",
        description=(
            "governance-layer dashboard and recall-testing layer for OpenClaw-style "
            "memory stores. This project does NOT delete memories; deletion flows "
            "produce review-only candidate lists."
        ),
        lifespan=lifespan,
    )

    static_dir = Path(__file__).resolve().parent / "static"
    if static_dir.exists():
        app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    # --- Health ------------------------------------------------------------

    @app.get("/health", tags=["meta"])
    def health() -> dict:
        return {"status": "ok"}

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/dashboard", status_code=status.HTTP_307_TEMPORARY_REDIRECT)

    # --- Login / logout ----------------------------------------------------

    @app.post("/login", include_in_schema=False)
    def login(
        request: Request,
        token: str = Form(default=""),
        password: str = Form(default=""),
        totp_code: str = Form(default=""),
        csrf_token: str = Form(default=""),
        recovery_code: str = Form(default=""),
    ) -> RedirectResponse:
        """Two-step login: Password+TOTP (preferred) or legacy bearer token.

        CSRF is enforced: the submitted ``csrf_token`` form field must match
        the ``csrf_token`` cookie.  Without this check a cross-origin form
        POST could log the victim in to an attacker-controlled session.

        Rate-limited: after 5 failed attempts per IP within 60 seconds,
        further attempts are rejected with HTTP 429.
        """
        client_ip = request.client.host if request.client else "unknown"
        # --- Rate limiting ---
        if settings.auth_enabled and _login_limiter.is_limited(client_ip):
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "auth_enabled": True,
                    "section": "overview",
                    "token": "",
                    "csrf_token": _issue_csrf_token(request),
                    "error": "登录尝试过于频繁，请稍后再试。",
                },
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            )
        # --- CSRF enforcement ---
        if settings.auth_enabled and not verify_csrf(request, csrf_token):
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "auth_enabled": True,
                    "section": "overview",
                    "token": "",
                    "csrf_token": _issue_csrf_token(request),
                    "error": "CSRF token 无效，请刷新页面重试。",
                },
                status_code=status.HTTP_403_FORBIDDEN,
            )
        submitted = (token or "").strip()
        if not submitted:
            header_token = extract_token(request, settings)
            if header_token:
                submitted = header_token
        if settings.auth_enabled and not attempt_login(
            token=submitted,
            password=(password or "").strip(),
            totp_code=(totp_code or "").strip(),
            recovery_code=(recovery_code or "").strip(),
            settings=settings,
        ):
            _login_limiter.record_failure(client_ip)
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "auth_enabled": True,
                    "section": "overview",
                    "token": "",
                    "csrf_token": _issue_csrf_token(request),
                    "error": "凭据无效。",
                },
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        # Success — reset rate limiter for this IP.
        _login_limiter.reset(client_ip)
        # Always issue a fresh random session token — never reuse the
        # submitted bearer token or password as the session cookie.
        # This ensures the session cookie value differs from both
        # MEMORY_OS_TOKEN and MEMORY_OS_PASSWORD.
        session_token = secrets.token_urlsafe(48)
        response = RedirectResponse(
            url="/dashboard/overview", status_code=status.HTTP_303_SEE_OTHER
        )
        set_session_cookie(response, session_token, settings)
        issue_csrf_cookie(response)
        return response

    @app.post("/logout", include_in_schema=False)
    def logout(
        request: Request,
        csrf_token: str = Form(default=""),
    ) -> RedirectResponse:
        """Log out: enforce CSRF, revoke the session in the persistent store, then clear cookies.

        CSRF is enforced the same way as ``/login``: the submitted
        ``csrf_token`` form field must match the ``csrf_token`` cookie.
        Without this check a cross-origin form POST could log the victim
        out of an attacker-controlled session, breaking availability
        (a DoS vector on top of which login-CSRF attacks often chain).
        """
        # --- CSRF enforcement ---
        if settings.auth_enabled and not verify_csrf(request, csrf_token):
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "auth_enabled": True,
                    "section": "overview",
                    "token": "",
                    "csrf_token": _issue_csrf_token(request),
                    "error": "CSRF token 无效，请刷新页面重试。",
                },
                status_code=status.HTTP_403_FORBIDDEN,
            )
        # Logout revokes the browser session cookie specifically. A caller may
        # also send an Authorization header; that static/API credential must neither
        # shadow the cookie nor prevent the cookie's persistent revocation.
        session_token = request.cookies.get(SESSION_COOKIE_NAME)
        response = RedirectResponse(url="/login", status_code=status.HTTP_303_SEE_OTHER)
        clear_session_cookie(response, token=session_token)
        return response

    @app.get("/login", response_class=HTMLResponse, include_in_schema=False, tags=["ui"])
    def login_page(request: Request, next: str = "/dashboard/overview") -> HTMLResponse:
        # One CSRF token for both cookie and hidden form field.
        csrf = generate_csrf_token()
        resp = templates.TemplateResponse(
            request,
            "login.html",
            {
                "auth_enabled": settings.auth_enabled,
                "section": "overview",
                "token": "",
                "csrf_token": csrf,
                "next": next,
            },
        )
        issue_csrf_cookie(resp, csrf)
        return resp

    # --- Dashboard HTML ----------------------------------------------------

    def _render_dashboard(request: Request, section: str = "overview") -> HTMLResponse:
        # Dashboard authentication is cookie / bearer only. Query-string
        # tokens are intentionally not accepted (they would land in nginx
        # access logs, browser history, and Referer headers).
        token = extract_token(request, settings)
        if settings.auth_enabled and not verify_token(token, settings):
            # Render the login page instead of a JSON 401.
            return templates.TemplateResponse(
                request,
                "login.html",
                {
                    "auth_enabled": True,
                    "section": section,
                    "token": "",
                    "csrf_token": _issue_csrf_token(request),
                },
                status_code=status.HTTP_401_UNAUTHORIZED,
            )
        backend: MemoryBackend = app.state.backend
        from .models import AutonomousGovernanceJob
        governance_status = _read_autonomous_governance_status()

        # v0.3.0: strategy card data — use the shared PolicyStore
        # instance from app.state (created in lifespan) so the
        # dashboard stays consistent with the API recall path. Falls
        # back to a fresh PolicyStore if app.state hasn't been
        # initialised yet (defensive — only hit in unit tests).
        _ps = getattr(app.state, "policy_store", None) or PolicyStore()
        _pol = _ps.get()
        _evo = _load_evolution_state()
        _policy_version = f"v{_pol.version}"
        _promo_count = _evo.get("promotion_count_30d", 0)
        _rollback_count = _evo.get("consecutive_rollbacks", 0)
        _shadow_count = len(_evo.get("shadow_comparisons", []))

        resp = templates.TemplateResponse(
            request,
            "dashboard.html",
            {
                "auth_enabled": settings.auth_enabled,
                "section": section,
                "backend": backend.name,
                "token": "",
                "autonomous_governance": AutonomousGovernanceJob.for_dashboard(
                    last_run=governance_status.get("last_run"),
                    last_result=governance_status.get("last_result"),
                    last_summary=governance_status.get("last_summary"),
                ),
                "policy_version": _policy_version,
                "promotion_count_30d": _promo_count,
                "consecutive_rollbacks": _rollback_count,
                "shadow_comparisons": _shadow_count,
                "titles": {"overview":"健康看板", "tiers":"层级分类", "duplicates":"去重审核", "recall":"召回测试", "governance":"自主治理", "strategy":"检索策略", "health":"系统健康", "security":"安全"},
                "csrf_token": _issue_csrf_token(request),
            },
        )
        return resp

    @app.get("/dashboard", response_class=HTMLResponse, tags=["ui"])
    def dashboard(request: Request) -> HTMLResponse:
        return _render_dashboard(request, "overview")

    @app.get("/dashboard/{section}", response_class=HTMLResponse, tags=["ui"])
    def dashboard_section(section: str, request: Request) -> HTMLResponse:
        allowed = {"overview", "tiers", "duplicates", "recall", "governance", "strategy", "evaluation", "memories", "health", "security"}
        if section not in allowed:
            raise HTTPException(status_code=404, detail="Unknown dashboard section")
        return _render_dashboard(request, section)

    # --- JSON API ----------------------------------------------------------

    def _backend() -> MemoryBackend:
        return app.state.backend

    def require_csrf_for_cookie_session(request: Request) -> None:
        """Enforce CSRF for browser-cookie state-changing requests.

        Non-browser API clients using ``Authorization: Bearer`` are exempt so
        CLI/scripts can keep using bearer auth without a CSRF cookie dance.
        When auth is disabled (demo/test mode), this is also a no-op.
        """
        if not settings.auth_enabled:
            return
        auth = request.headers.get("authorization") or request.headers.get("Authorization")
        if auth and auth.strip().lower().startswith("bearer "):
            return
        if not request.cookies.get("memory_os_session"):
            return
        if not verify_csrf(request):
            raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="Invalid CSRF token.")

    @app.get("/api/health", tags=["api"], dependencies=[Depends(require_auth)])
    def api_health() -> dict:
        backend = _backend()
        summary = build_health_summary(backend)
        data = summary.model_dump(mode="json")
        data["_auth"] = {
            "enabled": settings.auth_enabled,
            "has_password": bool(settings.memory_os_password),
            "has_totp": bool(settings.memory_os_totp_secret),
            "has_token": bool(settings.memory_os_token),
        }
        return data
    @app.get("/api/timeline", tags=["api"], dependencies=[Depends(require_auth)])
    def api_timeline() -> dict:
        backend = _backend()
        return {
            "backend": backend.name,
            "months": [m.model_dump(mode="json") for m in monthly_counts(backend.list_memories())],
        }

    @app.get("/api/tiers", tags=["api"], dependencies=[Depends(require_auth)])
    def api_tiers() -> dict:
        backend = _backend()
        return {
            "backend": backend.name,
            "tiers": [t.model_dump(mode="json") for t in tier_distribution(backend.list_memories())],
            "statuses": [s.model_dump(mode="json") for s in status_distribution(backend.list_memories())],
        }

    @app.get("/api/duplicates", tags=["api"], dependencies=[Depends(require_auth)])
    def api_duplicates() -> dict:
        backend = _backend()
        clusters = estimate_duplicate_clusters(backend.list_memories())
        return {
            "backend": backend.name,
            "clusters": [c.model_dump(mode="json") for c in clusters],
            "count": len(clusters),
        }

    @app.get("/api/deletion-candidates", tags=["api"], dependencies=[Depends(require_auth)])
    def api_deletion_candidates() -> dict:
        backend = _backend()
        candidates = build_deletion_candidates(backend.list_memories())
        return {
            "backend": backend.name,
            "count": len(candidates),
            "candidates": [c.model_dump(mode="json") for c in candidates],
            "policy": "review-only; no physical deletion is performed by this OS.",
        }

    @app.get("/api/strategy", tags=["api"], dependencies=[Depends(require_auth)])
    def api_strategy() -> dict:
        """Return the current retrieval policy and evolution state."""
        store = getattr(app.state, "policy_store", None) or PolicyStore()
        policy = store.get()
        state: dict = {"policy_version": f"v{policy.version}"}
        try:
            evo = _load_evolution_state()
            state["last_promotion_at"] = evo.get("last_promotion_at")
            state["promotion_count_30d"] = evo.get("promotion_count_30d", 0)
            state["consecutive_rollbacks"] = evo.get("consecutive_rollbacks", 0)
            state["shadow_comparisons"] = len(evo.get("shadow_comparisons", []))
        except Exception:
            pass
        # v0.3.0.x feature flag: surface SHADOW_ENABLED so dashboards
        # can display whether the shadow-comparison stage is active.
        state["shadow_enabled"] = bool(getattr(settings, "shadow_enabled", True))
        return {
            "state": state,
            "policy": policy.model_dump(mode="json"),
            "checksum": store.checksum(),
        }

    @app.get("/api/dashboard/strategy", tags=["api"], dependencies=[Depends(require_auth)])
    def api_dashboard_strategy() -> dict:
        """Alias used by the v0.3.0 web-console contract."""
        return api_strategy()

    @app.get("/api/dashboard/evaluation", tags=["api"], dependencies=[Depends(require_auth)])
    def api_dashboard_evaluation() -> dict:
        """Return the latest persisted *real* offline evaluation report.

        This endpoint never executes retrieval and never creates metrics with an
        empty ranker.  A fresh installation returns ``status=unavailable``.
        """
        from .evaluation_reports import (
            list_evaluation_reports,
            load_latest_evaluation_report,
            unavailable_envelope,
        )
        from .recall_feedback import get_feedback_summary

        latest = load_latest_evaluation_report()
        payload = dict(latest) if latest is not None else unavailable_envelope()
        payload["feedback"] = get_feedback_summary()
        payload["history"] = [
            {
                "report_id": report.get("report_id"),
                "generated_at": report.get("generated_at"),
                "status": report.get("status"),
                "corpus_snapshot_id": report.get("corpus_snapshot_id"),
                "policy": report.get("policy", {}),
                "decision": report.get("decision", {}),
            }
            for report in list_evaluation_reports(limit=5)
        ]
        return payload

    @app.get("/api/dashboard/memories", tags=["api"], dependencies=[Depends(require_auth)])
    def api_dashboard_memories(limit: int = 100, status_filter: str | None = None) -> dict:
        """Read-only collection-aware memory listing for the Memories page."""
        backend = _backend()
        memories = backend.list_memories()
        if status_filter:
            wanted = status_filter.lower()
            memories = [m for m in memories if m.status.value.lower() == wanted]
        return {
            "backend": backend.name,
            "collections": backend.list_collections(),
            "count": len(memories),
            "memories": [m.model_dump(mode="json") for m in memories[: max(1, min(limit, 500))]],
            "policy": "read-only; no physical deletion is performed by this OS.",
        }

    @app.get("/api/security/sessions", tags=["api", "security"], dependencies=[Depends(require_auth)])
    def api_security_sessions(request: Request) -> dict:
        """Return non-secret session metadata for the Security page."""
        token = extract_token(request, settings)
        events = get_audit_store().list_recent(limit=20, action="security")
        return {
            "sessions": list_sessions(current_token=token),
            "events": [e.model_dump(mode="json") for e in events],
        }

    @app.post(
        "/api/security/sessions/revoke-all",
        tags=["api", "security"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_security_revoke_all() -> dict:
        try:
            count = revoke_all_sessions()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Session revocation unavailable.",
            ) from exc
        get_audit_store().log("security", detail=f"revoked_all_sessions count={count}")
        return {"status": "ok", "revoked": count}

    @app.post(
        "/api/evolution/pause",
        tags=["api", "evolution"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_evolution_pause() -> dict:
        from .evolution import _load_evolution_state, _save_evolution_state
        # v0.3.0.x feature flag: EVOLUTION_ENABLED=off makes every
        # evolution endpoint a safe no-op (``status="disabled"``) so
        # policy is effectively static and we never mutate state.
        if not bool(getattr(settings, "evolution_enabled", True)):
            return {
                "status": "disabled",
                "paused": True,
                "reason": "evolution_enabled=off",
                "state": _load_evolution_state(),
            }
        state = _load_evolution_state()
        state["paused"] = True
        _save_evolution_state(state)
        get_audit_store().log("evolution", detail="paused via api")
        return {"status": "ok", "paused": True, "state": state}

    @app.post(
        "/api/evolution/resume",
        tags=["api", "evolution"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_evolution_resume() -> dict:
        from .evolution import _load_evolution_state, _save_evolution_state
        # v0.3.0.x feature flag: see ``api_evolution_pause``.
        if not bool(getattr(settings, "evolution_enabled", True)):
            return {
                "status": "disabled",
                "paused": False,
                "reason": "evolution_enabled=off",
                "state": _load_evolution_state(),
            }
        state = _load_evolution_state()
        state["paused"] = False
        _save_evolution_state(state)
        get_audit_store().log("evolution", detail="resumed via api")
        return {"status": "ok", "paused": False, "state": state}

    @app.post(
        "/api/evolution/candidate/reject",
        tags=["api", "evolution"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_evolution_candidate_reject() -> dict:
        from datetime import datetime, timezone
        from .evolution import _load_evolution_state, _save_evolution_state
        if not bool(getattr(settings, "evolution_enabled", True)):
            return {
                "status": "disabled",
                "reason": "evolution_enabled=off",
                "state": _load_evolution_state(),
            }
        store = getattr(app.state, "policy_store", None) or PolicyStore()
        try:
            rejected_version = store.reject_shadow()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Candidate rejection could not be persisted.",
            ) from exc
        state = _load_evolution_state()
        state["shadow_comparisons"] = []
        state["candidate_rejected_at"] = datetime.now(timezone.utc).isoformat()
        state["candidate_version"] = None
        _save_evolution_state(state)
        get_audit_store().log(
            "evolution", detail=f"candidate rejected via api version={rejected_version}"
        )
        return {
            "status": "ok",
            "rejected": rejected_version is not None,
            "rejected_version": rejected_version,
            "state": state,
        }


    @app.post(
        "/api/evolution/rollback",
        tags=["api", "evolution"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_evolution_rollback() -> dict:
        from datetime import datetime, timezone
        from .evolution import _load_evolution_state, _save_evolution_state
        if not bool(getattr(settings, "evolution_enabled", True)):
            return {
                "status": "disabled",
                "reason": "evolution_enabled=off",
                "policy_version": None,
                "checksum": None,
                "state": _load_evolution_state(),
            }
        store = getattr(app.state, "policy_store", None) or PolicyStore()
        target = "previous" if store.get_previous() is not None else "baseline"
        try:
            checksum = store.revert()
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Policy rollback could not be persisted.",
            ) from exc
        restored = store.get()
        state = _load_evolution_state()
        state["last_manual_rollback_at"] = datetime.now(timezone.utc).isoformat()
        state["consecutive_rollbacks"] = int(state.get("consecutive_rollbacks", 0)) + 1
        _save_evolution_state(state)
        get_audit_store().log(
            "evolution",
            detail=(
                f"manual rollback target={target} version={restored.version} "
                f"checksum={checksum}"
            ),
        )
        return {
            "status": "ok",
            "rollback_target": target,
            "policy_version": f"v{restored.version}",
            "checksum": checksum,
            "state": state,
        }



    @app.post(
        "/api/recall-test",
        response_model=RecallResponse,
        tags=["api"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_recall_test(request: Request, payload: RecallRequest) -> RecallResponse:
        """Run a recall test against the configured backend.

        v0.3.0: the endpoint now wires ``RetrievalEngine.retrieve``
        as the canonical recall pipeline (dense + lexical + RRF +
        feature rerank + Active-first / Superseded-fallback contract).
        The legacy scorer is available only through the explicit
        ``RETRIEVAL_ENGINE_V2=off`` rollback flag. Unexpected failures in the
        canonical engine return HTTP 503 so lifecycle, collection, and expiry
        contracts are never bypassed silently.

        Every successful response is persisted via the structured
        ``recall_runs`` / ``recall_results`` tables so the offline
        evaluation pipeline can replay the run with the same
        policy version. Persistence failures are logged but never
        abort the request — the user-facing response stays intact.
        """
        backend = _backend()

        # Hot-reload the policy file if an admin dropped a new one
        # into place since the last request. This is the production
        # caller of ``PolicyStore.reload_if_changed`` (G3 in the v0.3.0
        # gap plan). Safe to call on every request: it's a single
        # ``stat`` and a no-op when the mtime is unchanged.
        policy_store: PolicyStore = getattr(app.state, "policy_store", None) or PolicyStore()
        try:
            policy_store.reload_if_changed()
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("policy reload_if_changed failed: %s", exc)
        policy = policy_store.get()

        # v0.3.0.x feature flag: RETRIEVAL_ENGINE_V2=off forces the
        # legacy ``ranking.build_recall_response`` scorer so an operator
        # can revert to the v0.2.x behaviour without redeploying.
        # When the flag is on (default) we use the unified
        # ``RetrievalEngine`` and fall back to legacy only when the
        # engine path raises (defensive fallback, see below).
        use_legacy = not bool(getattr(settings, "retrieval_engine_v2", True))

        if use_legacy:
            # Legacy path: dense mode still wants backend.search
            # candidates (issue #2), other modes use the full corpus.
            if (payload.mode or "").lower() == "dense":
                candidates = backend.search(
                    payload.query,
                    limit=max(payload.limit * 4, 40),
                )
                response = build_recall_response(
                    backend.list_memories(),
                    payload,
                    backend_name=backend.name,
                    settings=settings,
                    dense_candidates=candidates,
                )
            else:
                response = build_recall_response(
                    backend.list_memories(),
                    payload,
                    backend_name=backend.name,
                    settings=settings,
                )
            # Best-effort: stamp the active policy version so the
            # offline evaluation pipeline can replay the same query
            # under the same policy (legacy path doesn't emit
            # diagnostics).
            try:
                response.policy_version = f"v{policy.version}"
            except Exception:
                pass
        else:
            # Build the engine and translate status_filter from request flags.
            statuses: List[str] = []
            # Default to Active-only. The engine itself re-issues the
            # second pass for Superseded fallback when the active pass
            # yields < fallback_min_results hits.
            if not payload.include_superseded:
                statuses = ["active"]
            else:
                statuses = ["active", "superseded"]
            if payload.include_expired:
                statuses.append("expired")

            # Lazy-load the BM25 index on first hybrid/keyword request.
            # The index is cached on app.state so subsequent requests reuse it.
            lexical_index = getattr(request.app.state, "lexical_index", None)
            if lexical_index is None:
                try:
                    from .lexical import BM25Index
                    from .retrieval_engine import _records_from_backend
                    lexical_cache_dir = Path(
                        os.environ.get(
                            "MEMORY_OS_LEXICAL_CACHE_DIR",
                            str(
                                Path(
                                    os.environ.get(
                                        "MEMORY_OS_RECALL_STATE_DIR",
                                        os.environ.get(
                                            "XDG_STATE_HOME",
                                            os.path.expanduser("~/.local/state"),
                                        ),
                                    )
                                )
                                / "openclaw-memory-os"
                                / "lexical-index"
                            ),
                        )
                    )
                    # Try cache first
                    lexical_index = BM25Index.load(lexical_cache_dir)
                    if lexical_index is None or len(lexical_index) == 0:
                        # Build from backend
                        records = _records_from_backend(
                            backend,
                            backend.list_memories(),
                        )
                        lexical_index = BM25Index()
                        for r in records:
                            lexical_index.add(r)
                        try:
                            lexical_index.save(lexical_cache_dir)
                        except Exception:
                            pass
                    request.app.state.lexical_index = lexical_index
                    request.app.state._lexical_cache_dir = lexical_cache_dir
                    logger.info(
                        "BM25 lexical index loaded (%d docs)",
                        len(lexical_index),
                    )
                except Exception as exc:
                    logger.warning("BM25 lexical index lazy-load failed: %s", exc)
                    lexical_index = None
            engine = RetrievalEngine(
                backend,
                policy_store,
                lexical_index=lexical_index,
            )
            try:
                result = engine.retrieve(
                    payload.query,
                    mode=payload.mode or "hybrid",
                    limit=payload.limit,
                    status_filter=statuses,
                )
                response = build_recall_response_v030(
                    payload, result, policy=policy, started_ms=0.0,
                )
                # Attach the engine diagnostics envelope (dense_available,
                # lexical_available, candidate_count, *_ms timing) so
                # dashboards / replay tooling can show what actually
                # happened. The schema accepts an arbitrary Dict.
                try:
                    response.diagnostics = result.diagnostics.model_dump(mode="json")
                except Exception:
                    pass
                # Ensure the response carries the active policy version
                # for downstream consumers (dashboards / replay tooling).
                try:
                    response.policy_version = f"v{policy.version}"
                except Exception:
                    pass
            except Exception as exc:
                logger.exception(
                    "RetrievalEngine failed for query=%r; refusing legacy fallback",
                    payload.query,
                )
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Retrieval engine unavailable.",
                ) from exc


        # ---- Persist structured recall run / results ----------------------
        # v0.3.0.x feature flag: STRUCTURED_FEEDBACK=off falls back to the
        # v0.2.x audit-log-style persistence (none here — the legacy
        # path didn't persist per-hit traces). When off, we skip the
        # entire structured persistence block; the response still ships.
        if not bool(getattr(settings, "structured_feedback", True)):
            return response

        # Best-effort: a write failure must not break the response,
        # but it should be loud enough for ops to notice.
        try:
            from .recall_feedback import (
                record_recall_run,
                record_recall_result,
            )
            qid = response.query_id or ""
            if qid:
                diag = response.diagnostics or {}
                # v0.3.0.x: surface diagnostics as first-class fields so
                # the offline evaluation pipeline can reason about which
                # channels were live and which collections were touched
                # without re-parsing the diagnostics dict.
                _dense_avail = diag.get("dense_available")
                _lex_avail = diag.get("lexical_available")
                _colls_searched = diag.get("collections_searched") or []
                if not isinstance(_colls_searched, (list, tuple)):
                    _colls_searched = []
                # The engine currently reports a single "collections_searched"
                # list. Until per-collection success/failure is wired into
                # the engine, treat them as a single succeeded bucket and
                # leave the failed list empty (downstream code can split
                # the list if the engine later publishes it).
                _colls_failed = diag.get("collections_failed") or []
                if not isinstance(_colls_failed, (list, tuple)):
                    _colls_failed = []
                _colls_succeeded = [
                    c for c in _colls_searched
                    if c and c not in _colls_failed
                ]
                record_recall_run(
                    query_id=qid,
                    query_text=payload.query,
                    retrieval_mode=payload.mode or "hybrid",
                    policy_version=response.policy_version or f"v{policy.version}",
                    latency_ms=float(response.took_ms or 0.0),
                    retrieval_status=str(diag.get("status") or "ok"),
                    degraded_reason=diag.get("degraded_reason") or diag.get("reason"),
                    fallback_used=bool(response.fallback and response.fallback.used),
                    corpus_snapshot_id=str(diag.get("corpus_snapshot_id") or "")
                    or None,
                    dense_available=(
                        bool(_dense_avail) if _dense_avail is not None else None
                    ),
                    lexical_available=(
                        bool(_lex_avail) if _lex_avail is not None else None
                    ),
                    collections_succeeded=_colls_succeeded or None,
                    collections_failed=_colls_failed or None,
                )
                for rank, hit in enumerate(response.hits, start=1):
                    components = hit.components or {}
                    # v0.3.0.x: split each score into its raw and
                    # calibrated form when both are available. The v0.3.0
                    # engine emits the calibrated score in
                    # ``components["vector"]`` / ``components["lexical"]``;
                    # the raw form is the dense_score / lexical_score on
                    # the underlying ScoredMemoryCandidate (we don't have
                    # it here, so we fall back to the same calibrated
                    # value and let the pipeline fill the raw form from
                    # engine diagnostics if it has them).
                    _vec = components.get("vector")
                    _lex = components.get("lexical")
                    _imp = components.get("importance")
                    # Recency component isn't surfaced on the public
                    # RecallHit in v0.3.0; record the importance
                    # contribution as the recency score proxy so the
                    # column is non-null when available, and the
                    # pipeline can recompute the true recency component
                    # from ``created_at``/``updated_at`` later.
                    _rec_proxy = (
                        components.get("recency")
                        if "recency" in components
                        else None
                    )
                    record_recall_result(
                        query_id=qid,
                        candidate_key=hit.candidate_key or "",
                        memory_id=hit.id,
                        collection=hit.collection or "",
                        rank=rank,
                        status=hit.status.value,
                        vector_score=float(_vec) if _vec is not None else 0.0,
                        lexical_score=float(_lex) if _lex is not None else 0.0,
                        rrf_score=float(components.get("rrf", 0.0)),
                        final_score=float(hit.score),
                        explanation=hit.explanation or "",
                        vector_score_raw=(
                            float(_vec) if _vec is not None else None
                        ),
                        vector_score_calibrated=(
                            float(_vec) if _vec is not None else None
                        ),
                        lexical_score_raw=(
                            float(_lex) if _lex is not None else None
                        ),
                        lexical_score_calibrated=(
                            float(_lex) if _lex is not None else None
                        ),
                        importance_score=(
                            float(_imp) if _imp is not None else float(hit.importance)
                        ),
                        recency_score=(
                            float(_rec_proxy) if _rec_proxy is not None else None
                        ),
                        feedback_score=(
                            float(components.get("feedback", 0.0))
                            if "feedback" in components
                            else None
                        ),
                        display_score=float(hit.score),
                    )
        except Exception as exc:
            logger.warning("record_recall_run/result failed: %s", exc)

        return response

    # --- Feedback API -----------------------------------------------------

    @app.post(
        "/api/feedback",
        tags=["api"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_feedback(payload: FeedbackEntry) -> dict:
        """Record useful/not-useful feedback on a recall hit.

        v0.3.0: when ``query_id`` and ``candidate_key`` are present
        AND the STRUCTURED_FEEDBACK feature flag is enabled (default),
        the feedback is stored in the new structured tables
        (``recall_runs`` / ``recall_results`` / ``feedback_events``).

        v0.3.0.x: when STRUCTURED_FEEDBACK=off the structured path is
        disabled and we route to the legacy ``feedback.record_feedback``
        helper even when ``query_id`` / ``candidate_key`` are provided.
        This preserves the v0.2.x semantics operators may rely on.
        The legacy path (``memory_id`` + ``query``) is always used
        when those structured fields are missing.
        """
        structured_on = bool(getattr(settings, "structured_feedback", True))
        if structured_on and payload.query_id and payload.candidate_key:
            from .recall_feedback import record_feedback_v030
            try:
                row_id = record_feedback_v030(
                    query_id=payload.query_id,
                    candidate_key=payload.candidate_key,
                    useful=payload.useful,
                    memory_id=payload.memory_id or "",
                )
            except ValueError as exc:
                # G5.1 strong validation: the candidate_key was not
                # actually returned for this query_id. Surface the
                # error to the client as HTTP 422 (semantic: the
                # request was well-formed but the server refuses to
                # process it because the payload references a state
                # the server does not have on file).
                raise HTTPException(
                    status_code=422,
                    detail=str(exc),
                )
        else:
            from .feedback import record_feedback
            row_id = record_feedback(
                memory_id=payload.memory_id,
                query=payload.query,
                useful=payload.useful,
                note=payload.note,
            )
        return {"status": "ok", "row_id": row_id}

    # --- Audit log API ----------------------------------------------------

    @app.get(
        "/api/feedback-summary",
        tags=["api"],
        dependencies=[Depends(require_auth)],
    )
    def api_feedback_summary() -> dict:
        """Return aggregated feedback ratios and the current weight snapshot."""
        from .ranking import _load_feedback_weights
        # Try fresh replay from the audit log
        _replay_result: Optional[dict] = None
        try:
            import importlib.util as _util
            _spec = _util.spec_from_file_location(
                "_replay_feedback_mod",
                str(Path(__file__).resolve().parents[1] / "scripts" / "replay_feedback.py"),
            )
            if _spec and _spec.loader:
                _mod = _util.module_from_spec(_spec)
                _spec.loader.exec_module(_mod)
                audit_store = get_audit_store()
                _replay_result = _mod.replay(audit_store._db_path)  # type: ignore[attr-defined]
        except Exception:
            pass
        if _replay_result is not None:
            weights = _replay_result
        else:
            weights = _load_feedback_weights() or {}
        return {
            "source": "feedback-summary",
            "weights": weights,
        }

    @app.get(
        "/api/audit-log",
        tags=["api"],
        dependencies=[Depends(require_auth)],
    )
    def api_audit_log(limit: int = 50, action: str | None = None) -> dict:
        audit = get_audit_store()
        entries = audit.list_recent(limit=min(limit, 200), action=action)
        return {
            "entries": [e.model_dump(mode="json") for e in entries],
            "count": len(entries),
        }

    # --- Consolidation API ------------------------------------------------

    @app.post(
        "/api/consolidate-duplicates",
        tags=["api"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_consolidate(payload: ConsolidationRequest) -> dict:
        """Analyze a duplicate consolidation without modifying storage."""

        backend = _backend()
        memories = backend.list_memories()
        mem_map = {m.id: m for m in memories}

        members = []
        not_found = []
        for mid in payload.cluster_ids:
            m = mem_map.get(mid)
            if m:
                members.append(m)
            else:
                not_found.append(mid)

        if not members:
            raise HTTPException(status_code=404, detail="No memory IDs found")

        result = consolidate_cluster(members, strategy=payload.strategy)

        # Log to audit store
        audit = get_audit_store()
        audit.log(
            "consolidate_analysis",
            detail=f"cluster_ids={payload.cluster_ids!r} strategy={payload.strategy} "
            f"consolidated_id={result.consolidated_id} "
            f"merged={len(result.merged_member_ids)}",
        )

        return {
            "consolidation": result.model_dump(mode="json"),
            "not_found": not_found,
        }

    @app.post(
        "/api/maintenance/reclassify",
        tags=["api", "maintenance"],
        dependencies=[Depends(require_auth), Depends(require_csrf_for_cookie_session)],
    )
    def api_reclassify(payload: ReclassifyRequest) -> dict:
        """Trigger the OpenClaw Memory OS auto-classifier (writes to Qdrant).

        Default behaviour is **apply** (writes tier / importance / type /
        topic / owner_confirmed for points whose classifier output differs).
        Pass ``dry_run=true`` in the request body to preview only.
        """
        import subprocess
        import sys
        from pathlib import Path

        from .config import get_settings
        settings = get_settings()
        project_root = Path(__file__).resolve().parents[1]
        script = project_root / "scripts" / "tier_classifier.py"
        if not script.exists():
            raise HTTPException(status_code=500, detail=f"classifier script not found: {script}")

        # Default: run against the primary + secondary collections configured
        # for this backend (auto-coverage, no operator input needed).
        backend = _backend()
        colls = payload.collections or backend.list_collections()
        if not colls:
            colls = [settings.qdrant_collection]

        cmd = [sys.executable, str(script)]
        for c in colls:
            cmd += ["--collection", c]
        if payload.dry_run:
            cmd.append("--dry-run")

        try:
            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=1800)
        except subprocess.TimeoutExpired:
            raise HTTPException(status_code=504, detail="reclassify timed out after 1800s")

        # Log to audit store
        try:
            audit = get_audit_store()
            audit.log(
                "reclassify",
                detail=f"collections={colls} dry_run={payload.dry_run} "
                       f"exit={proc.returncode}",
            )
        except Exception:
            pass

        return {
            "collections": colls,
            "dry_run": payload.dry_run,
            "exit_code": proc.returncode,
            "stdout": proc.stdout,
            "stderr": proc.stderr,
        }

    return app


# Module-level app instance for ``uvicorn openclaw_memory_os.app:app``.
app = _build_app()


def create_app() -> FastAPI:
    """Factory for tests / alternative configurations."""
    reset_settings_cache()
    return _build_app()
