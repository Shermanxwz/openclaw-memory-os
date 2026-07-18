"""Command-line entry point.

Usage examples::

    openclaw-memory-os health
    openclaw-memory-os recall --query "worker model rule"
    openclaw-memory-os serve --host 127.0.0.1 --port 7788
    openclaw-memory-os privacy-scan

The CLI is intentionally thin. The web app is the primary interface; the
CLI exists for cron jobs, smoke tests, and ``scripts/`` automation.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Optional, Sequence

from .backends import get_backend
from .config import get_settings, reset_settings_cache
from .models import RecallRequest
from .policy_store import PolicyStore
from .ranking import build_recall_response
from .retrieval_engine import RetrievalEngine, build_recall_response_v030


def _print_json(data, *, indent: int = 2) -> None:
    json.dump(data, sys.stdout, indent=indent, default=str, ensure_ascii=False)
    sys.stdout.write("\n")


def cmd_health(_args: argparse.Namespace) -> int:
    settings = get_settings()
    backend = get_backend(settings)
    from .analytics import build_health_summary
    summary = build_health_summary(backend)
    _print_json(summary.model_dump(mode="json"))
    return 0


def cmd_recall(args: argparse.Namespace) -> int:
    """Run a recall test and print the JSON response.

    v0.3.0: this command wires the unified ``RetrievalEngine`` as the
    canonical recall pipeline (dense + lexical + RRF + Active-first
    / Superseded-fallback contract). The legacy
    :func:`openclaw_memory_os.ranking.build_recall_response` is kept
    only as a defensive fallback so older cron / smoke scripts don't
    crash if the engine path raises on a pathological backend.

    Every successful run is persisted to the structured
    ``recall_runs`` / ``recall_results`` tables so the offline
    evaluation pipeline can replay the same query under different
    policy versions. Persistence is best-effort — a write failure
    prints a stderr warning but does not change the response shape.
    """
    settings = get_settings()
    backend = get_backend(settings)
    req = RecallRequest(
        query=args.query,
        mode=args.mode,
        since_days=args.since_days,
        include_superseded=args.include_superseded,
        include_expired=args.include_expired,
        limit=args.limit,
    )

    # Single PolicyStore instance per CLI invocation; reload the
    # on-disk file if an admin dropped a new policy into place.
    policy_store = PolicyStore()
    try:
        policy_store.reload_if_changed()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"policy reload_if_changed failed: {exc}", file=sys.stderr)
    policy = policy_store.get()

    # v0.3.0.x feature flag: RETRIEVAL_ENGINE_V2=off forces the legacy
    # ``ranking.build_recall_response`` scorer. When on (default) we use
    # the unified ``RetrievalEngine`` and fall back to legacy only when
    # the engine path raises (defensive fallback, see below).
    use_legacy = not bool(getattr(settings, "retrieval_engine_v2", True))

    if use_legacy:
        # Legacy path. Dense mode still wants backend.search
        # candidates (issue #2); other modes use the full corpus.
        if (req.mode or "").lower() == "dense":
            candidates = backend.search(req.query, limit=max(req.limit * 4, 40))
            response = build_recall_response(
                backend.list_memories(),
                req,
                backend_name=backend.name,
                settings=settings,
                dense_candidates=candidates,
            )
        else:
            response = build_recall_response(
                backend.list_memories(),
                req,
                backend_name=backend.name,
                settings=settings,
            )
        # Legacy path doesn't emit diagnostics; stamp the active
        # policy version so the offline evaluation pipeline can
        # replay the run under the same policy.
        try:
            response.policy_version = f"v{policy.version}"
        except Exception:
            pass
    else:
        # Translate request flags to engine status_filter. The engine
        # re-issues a Superseded pass on its own when the active pass
        # yields fewer than fallback_min_results hits.
        statuses = ["active"]
        if req.include_superseded:
            statuses.append("superseded")
        if req.include_expired:
            statuses.append("expired")

        engine = RetrievalEngine(backend, policy_store)
        try:
            result = engine.retrieve(
                req.query,
                mode=req.mode or "hybrid",
                limit=req.limit,
                status_filter=statuses,
            )
            response = build_recall_response_v030(
                req, result, policy=policy, started_ms=0.0,
            )
            try:
                response.diagnostics = result.diagnostics.model_dump(mode="json")
            except Exception:
                pass
            try:
                response.policy_version = f"v{policy.version}"
            except Exception:
                pass
        except Exception as exc:
            print(
                f"RetrievalEngine path failed for query={req.query!r}: {exc}; "
                f"falling back to legacy build_recall_response",
                file=sys.stderr,
            )
            # Legacy fallback. Dense mode still wants backend.search
            # candidates (issue #2); other modes use the full corpus.
            if (req.mode or "").lower() == "dense":
                candidates = backend.search(req.query, limit=max(req.limit * 4, 40))
                response = build_recall_response(
                    backend.list_memories(),
                    req,
                    backend_name=backend.name,
                    settings=settings,
                    dense_candidates=candidates,
                )
            else:
                response = build_recall_response(
                    backend.list_memories(),
                    req,
                    backend_name=backend.name,
                    settings=settings,
                )

    # ---- Persist structured recall run / results ----------------------
    # v0.3.0.x feature flag: STRUCTURED_FEEDBACK=off skips per-hit
    # persistence so the CLI matches the v0.2.x behaviour. The
    # response still ships as JSON.
    if not bool(getattr(settings, "structured_feedback", True)):
        _print_json(response.model_dump(mode="json"))
        return 0

    try:
        from .recall_feedback import record_recall_result, record_recall_run
        qid = response.query_id or ""
        if qid:
            diag = response.diagnostics or {}
            _dense_avail = diag.get("dense_available")
            _lex_avail = diag.get("lexical_available")
            _colls_searched = diag.get("collections_searched") or []
            if not isinstance(_colls_searched, (list, tuple)):
                _colls_searched = []
            _colls_failed = diag.get("collections_failed") or []
            if not isinstance(_colls_failed, (list, tuple)):
                _colls_failed = []
            _colls_succeeded = [
                c for c in _colls_searched
                if c and c not in _colls_failed
            ]
            record_recall_run(
                query_id=qid,
                query_text=req.query,
                retrieval_mode=req.mode or "hybrid",
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
                _vec = components.get("vector")
                _lex = components.get("lexical")
                _imp = components.get("importance")
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
        print(f"record_recall_run/result failed: {exc}", file=sys.stderr)

    _print_json(response.model_dump(mode="json"))
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    try:
        import uvicorn  # local import to keep CLI snappy
    except ImportError:
        print("uvicorn is not installed; cannot serve.", file=sys.stderr)
        return 2
    reset_settings_cache()
    uvicorn.run(
        "openclaw_memory_os.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def cmd_privacy_scan(args: argparse.Namespace) -> int:
    from .privacy import scan_paths

    findings = scan_paths(
        args.paths,
        baseline=getattr(args, "baseline", None),
    )
    _print_json({"findings": findings, "count": len(findings)})
    return 0 if not findings else 1


def cmd_audit(args: argparse.Namespace) -> int:
    from .audit import get_audit_store
    store = get_audit_store()
    entries = store.list_recent(limit=args.limit, action=args.action)
    _print_json(
        {"entries": [e.model_dump(mode="json") for e in entries], "count": len(entries)}
    )
    return 0


def cmd_feedback(args: argparse.Namespace) -> int:
    from .feedback import record_feedback
    row_id = record_feedback(args.memory_id, args.query, args.useful, note=args.note)
    _print_json({"status": "ok", "row_id": row_id})
    return 0


def cmd_consolidate(args: argparse.Namespace) -> int:
    from .consolidation import consolidate_cluster
    from .backends import get_backend
    settings = get_settings()
    backend = get_backend(settings)
    memories = backend.list_memories()
    mem_map = {m.id: m for m in memories}
    members = []
    for mid in args.ids:
        m = mem_map.get(mid)
        if m:
            members.append(m)
    if not members:
        _print_json({"error": "No valid IDs found"})
        return 1
    result = consolidate_cluster(members, strategy=args.strategy)
    _print_json(result.model_dump(mode="json"))
    return 0


def cmd_ingest(args: argparse.Namespace) -> int:
    from .ingestion import IngestionManager
    manager = IngestionManager()
    result = manager.run_ingestion(
        collection=args.collection,
        since_days=args.since_days,
        dry_run=args.dry_run,
        limit=args.limit,
        resume=not args.no_resume,
        skip_existing=not args.no_skip,
        batch_size=args.batch_size,
    )
    _print_json(result.model_dump(mode="json"))
    return 0 if result.failed == 0 else 1 if result.status == "completed" else 2


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="openclaw-memory-os")
    sub = parser.add_subparsers(dest="command", required=True)

    p_health = sub.add_parser("health", help="Print the memory health summary as JSON.")
    p_health.set_defaults(func=cmd_health)

    p_recall = sub.add_parser("recall", help="Run a recall test against the configured backend.")
    p_recall.add_argument("--query", required=True)
    p_recall.add_argument("--mode", default="hybrid", choices=["hybrid", "keyword", "dense"])
    p_recall.add_argument("--since-days", type=int, default=None)
    p_recall.add_argument("--include-superseded", action="store_true")
    p_recall.add_argument("--include-expired", action="store_true")
    p_recall.add_argument("--limit", type=int, default=10)
    p_recall.set_defaults(func=cmd_recall)

    p_serve = sub.add_parser("serve", help="Run the FastAPI app via uvicorn.")
    p_serve.add_argument("--host", default="0.0.0.0")
    p_serve.add_argument("--port", type=int, default=7788)
    p_serve.add_argument("--reload", action="store_true")
    p_serve.set_defaults(func=cmd_serve)

    p_scan = sub.add_parser("privacy-scan", help="Scan paths for forbidden patterns.")
    p_scan.add_argument("paths", nargs="*", default=["."])
    p_scan.add_argument(
        "--baseline",
        default=None,
        help="Optional JSON baseline file pinning legitimate findings.",
    )
    p_scan.set_defaults(func=cmd_privacy_scan)

    p_audit = sub.add_parser("audit", help="View the SQLite audit log.")
    p_audit.add_argument("--limit", type=int, default=50)
    p_audit.add_argument("--action", default=None, help="Filter by action type.")
    p_audit.set_defaults(func=cmd_audit)

    p_fb = sub.add_parser("feedback", help="Record recall feedback.")
    p_fb.add_argument("--memory-id", required=True)
    p_fb.add_argument("--query", required=True)
    p_fb.add_argument("--useful", action="store_true")
    p_fb.add_argument("--not-useful", dest="useful", action="store_false")
    p_fb.set_defaults(useful=True)
    p_fb.add_argument("--note", default=None)
    p_fb.set_defaults(func=cmd_feedback)

    p_con = sub.add_parser("consolidate", help="Analyze duplicate consolidation.")
    p_con.add_argument("--ids", nargs="+", required=True, help="Memory IDs to consolidate.")
    p_con.add_argument("--strategy", default="merge", choices=["merge", "keep_newest", "keep_best"])
    p_con.set_defaults(func=cmd_consolidate)

    p_ing = sub.add_parser("ingest", help="Ingest memory files with checkpoint/resume.")
    p_ing.add_argument("--collection", default="openclaw_memory_os")
    p_ing.add_argument("--since-days", type=int, default=None)
    p_ing.add_argument("--dry-run", action="store_true")
    p_ing.add_argument("--limit", type=int, default=0)
    p_ing.add_argument("--no-resume", action="store_true", help="Force fresh start (ignore checkpoint).")
    p_ing.add_argument("--no-skip", action="store_true", help="Re-embed and upsert all chunks.")
    p_ing.add_argument("--batch-size", type=int, default=32)
    p_ing.set_defaults(func=cmd_ingest)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())