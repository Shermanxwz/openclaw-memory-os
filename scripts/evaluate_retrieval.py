#!/usr/bin/env python3
"""v0.3.0.x: Offline retrieval-evaluation entry point.

Safe, side-effect-free CLI that prints the latest persisted real offline evaluation report. Designed to be runnable
without a live Qdrant (or even without any recall data) — the script
only ever reads from the local recall_feedback SQLite store and emits
a status envelope that mirrors the shape returned by
``GET /api/dashboard/evaluation``.

Usage::

    scripts/evaluate_retrieval.py                # print JSON to stdout
    scripts/evaluate_retrieval.py --out FILE     # write JSON to FILE
    scripts/evaluate_retrieval.py --pretty       # indent for humans
    scripts/evaluate_retrieval.py --limit 50     # cap cases per run

Output envelope (stable contract)::

    {
      "status": "ok" | "unavailable" | "error",
      "generated_at": "...ISO-8601 UTC...",
      "corpus_snapshot_id": "snap-..." | null,
      "metrics": { ... EvalResult.to_dict() ... },
      "feedback": { ... get_feedback_summary() ... },
      "history": [ ... recent offline runs ... ],
      "notes": [ "..." ],
      "warnings": [ "..." ]
    }

The script NEVER fabricates metrics. When there are no judged cases
it returns ``status="unavailable"`` with all metric values defaulting
to ``null`` (for the graded v0.3.0.x fields) or ``0.0`` (for legacy
fields) and explicit ``status="unavailable"`` markers per metric.
This matches the dashboard's "honest null" contract.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

# Ensure the project root is importable when the script is invoked
# directly (e.g. ``./scripts/evaluate_retrieval.py``) — without this
# the openclaw_memory_os.* imports fail in a fresh checkout.
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger("evaluate_retrieval")


def _empty_metrics_envelope() -> Dict[str, Any]:
    """Return the metric envelope with the documented zero/null defaults.

    Honest-null contract: when there are no judged cases, every graded
    metric is ``None`` (not ``0``) so the dashboard / CLI can surface
    the em-dash placeholder instead of falsely claiming a "scored 0"
    result. ``num_cases`` stays ``0`` so the dashboard layout is
    preserved.
    """
    return {
        "recall_at_1": None,
        "recall_at_5": None,
        "recall_at_10": None,
        "mrr_at_10": None,
        "ndcg_at_10": None,
        "useful_at_1": None,
        "useful_at_5": None,
        "explicit_negative_at_5": None,
        "no_result_rate": None,
        "p50_latency": None,
        "p95_latency": None,
        "degraded_rate": None,
        "fallback_rate": None,
        "num_cases": 0,
        "judged_ndcg_at_10": None,
        "useful_superseded_fallback_rate": None,
        "num_judged_cases": 0,
        "corpus_snapshot_id": None,
        "judged_ndcg_status": "unavailable",
        "fallback_rate_status": "unavailable",
    }


def _safe_call(fn, *args, default=None, label: str = "", warnings: Optional[List[str]] = None):
    """Call ``fn`` and convert any exception into a warning.

    Used so that a transient DB / IO failure never causes the script
    to abort; we still surface the failure as a ``warnings`` entry so
    operators can see what happened.
    """
    try:
        return fn(*args)
    except Exception as exc:  # pragma: no cover - defensive
        if warnings is not None:
            warnings.append(f"{label}_failed: {exc}")
        logger.warning("%s failed: %s", label or "call", exc)
        return default


def _build_envelope(limit: int = 500) -> Dict[str, Any]:
    """Read the latest persisted real report; never fabricate a scored run."""
    from openclaw_memory_os.evaluation_reports import (
        list_evaluation_reports,
        load_latest_evaluation_report,
        unavailable_envelope,
    )
    from openclaw_memory_os.recall_feedback import get_feedback_summary

    latest = load_latest_evaluation_report()
    envelope = dict(latest) if latest is not None else unavailable_envelope()
    envelope["feedback"] = _safe_call(
        get_feedback_summary,
        default={},
        label="feedback_summary",
        warnings=envelope.setdefault("warnings", []),
    )
    envelope["history"] = [
        {
            "report_id": report.get("report_id"),
            "generated_at": report.get("generated_at"),
            "status": report.get("status"),
            "corpus_snapshot_id": report.get("corpus_snapshot_id"),
            "policy": report.get("policy", {}),
            "decision": report.get("decision", {}),
        }
        for report in list_evaluation_reports(limit=max(1, min(limit, 20)))
    ]
    # ``--limit`` is a presentation cap for this read-only report command.
    # Never claim a larger displayed sample count than the requested cap.
    metrics = dict(envelope.get("metrics") or {})
    for key in ("num_cases", "num_judged_cases"):
        try:
            metrics[key] = min(max(0, int(metrics.get(key, 0))), max(0, int(limit)))
        except (TypeError, ValueError):
            metrics[key] = 0
    envelope["metrics"] = metrics
    return envelope


def _parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="evaluate_retrieval.py",
        description="Print the offline retrieval-evaluation envelope as JSON.",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Write JSON to this path instead of stdout.",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON (indent=2) for humans.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=500,
        help="Maximum number of judged cases to load (default 500).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parse_args(argv)
    envelope = _build_envelope(limit=max(1, args.limit))
    payload = json.dumps(envelope, indent=2 if args.pretty else None, ensure_ascii=False)
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(payload + "\n", encoding="utf-8")
        logger.info("wrote %s (%d bytes)", out_path, len(payload))
    else:
        print(payload)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())