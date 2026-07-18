"""Persistent, privacy-safe offline evaluation reports.

Only real evaluation runs write reports. Dashboard and CLI consumers read these
files and return ``unavailable`` when none exists; they never fabricate metrics
with an empty ranker.
"""
from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

REPORT_SCHEMA_VERSION = 1

# Stable dashboard/CLI response shape. Graded values are explicitly ``None``
# when no real report exists; counters remain zero. This preserves backwards
# compatibility without running a fake or empty ranker.
EMPTY_METRICS: Dict[str, Any] = {
    "recall_at_1": None,
    "recall_at_5": None,
    "recall_at_10": None,
    "mrr_at_10": None,
    "ndcg_at_10": None,
    "positive_hit_at_1": None,
    "positive_hit_at_5": None,
    "positive_hit_at_10": None,
    "positive_mrr_at_10": None,
    "useful_at_1": None,
    "useful_at_5": None,
    "explicit_negative_at_5": None,
    "no_result_rate": None,
    "p50_latency": None,
    "p95_latency": None,
    "degraded_rate": None,
    "fallback_rate": None,
    "judged_ndcg_at_10": None,
    "useful_superseded_fallback_rate": None,
    "fallback_useful_rate": None,
    "num_cases": 0,
    "num_judged_cases": 0,
    "corpus_snapshot_id": None,
    "judged_ndcg_status": "unavailable",
    "fallback_rate_status": "unavailable",
}


def _default_report_dir() -> Path:
    """Resolve the report directory at call time.

    ``MEMORY_OS_RECALL_STATE_DIR`` is the common isolation boundary for recall,
    feedback, evolution state, and evaluation reports. Resolving dynamically
    prevents one process/test tenant from reading another tenant's latest report.
    """
    override = os.environ.get("MEMORY_OS_EVALUATION_REPORT_DIR", "").strip()
    if override:
        return Path(override).expanduser()
    recall_state = os.environ.get("MEMORY_OS_RECALL_STATE_DIR", "").strip()
    if recall_state:
        return Path(recall_state).expanduser() / "openclaw-memory-os" / "evaluation-reports"
    state_home = os.environ.get("XDG_STATE_HOME") or os.path.expanduser("~/.local/state")
    return Path(state_home) / "openclaw-memory-os" / "evaluation-reports"


def _secure(path: Path, *, directory: bool = False) -> None:
    if os.name == "nt":  # pragma: no cover
        return
    os.chmod(path, 0o700 if directory else 0o600)


def _prepare_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    _secure(path, directory=True)


def _atomic_write(path: Path, payload: Dict[str, Any]) -> None:
    _prepare_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    _secure(tmp)
    os.replace(tmp, path)
    _secure(path)


def new_report_id() -> str:
    return uuid.uuid4().hex


def normalize_report(report: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(report, dict):
        raise TypeError("evaluation report must be a dict")
    normalized = dict(report)
    normalized.setdefault("schema_version", REPORT_SCHEMA_VERSION)
    if int(normalized["schema_version"]) != REPORT_SCHEMA_VERSION:
        raise ValueError("unsupported evaluation report schema_version")
    normalized.setdefault("report_id", new_report_id())
    normalized.setdefault("generated_at", datetime.now(timezone.utc).isoformat())
    normalized.setdefault("status", "ok")
    if normalized["status"] not in {"ok", "unavailable", "error"}:
        raise ValueError("invalid evaluation report status")
    normalized.setdefault("corpus_snapshot_id", None)
    supplied_metrics = normalized.get("metrics") or {}
    if not isinstance(supplied_metrics, dict):
        raise TypeError("evaluation report metrics must be a dict")
    metrics = dict(EMPTY_METRICS)
    metrics.update(supplied_metrics)
    if normalized.get("corpus_snapshot_id") is not None:
        metrics["corpus_snapshot_id"] = normalized["corpus_snapshot_id"]
    normalized["metrics"] = metrics
    normalized.setdefault("active_metrics", {})
    normalized.setdefault("candidate_metrics", {})
    normalized.setdefault("policy", {})
    normalized.setdefault("split", {})
    normalized.setdefault("decision", {})
    normalized.setdefault("warnings", [])
    normalized.setdefault("notes", [])
    return normalized


def save_evaluation_report(
    report: Dict[str, Any], *, report_dir: Optional[Path] = None
) -> Path:
    directory = Path(report_dir) if report_dir is not None else _default_report_dir()
    _prepare_dir(directory)
    payload = normalize_report(report)
    report_id = str(payload["report_id"])
    target = directory / f"{report_id}.json"
    _atomic_write(target, payload)
    _atomic_write(directory / "latest.json", payload)
    return target


def _load(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return normalize_report(raw)
    except Exception:
        return None


def load_latest_evaluation_report(
    *, report_dir: Optional[Path] = None
) -> Optional[Dict[str, Any]]:
    directory = Path(report_dir) if report_dir is not None else _default_report_dir()
    return _load(directory / "latest.json")


def list_evaluation_reports(
    *, report_dir: Optional[Path] = None, limit: int = 5
) -> List[Dict[str, Any]]:
    directory = Path(report_dir) if report_dir is not None else _default_report_dir()
    if not directory.exists():
        return []
    reports: List[Dict[str, Any]] = []
    for path in directory.glob("*.json"):
        if path.name == "latest.json":
            continue
        report = _load(path)
        if report is not None:
            reports.append(report)
    reports.sort(key=lambda r: str(r.get("generated_at") or ""), reverse=True)
    return reports[: max(0, int(limit))]


def unavailable_envelope() -> Dict[str, Any]:
    return {
        "status": "unavailable",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "report_id": None,
        "corpus_snapshot_id": None,
        "metrics": dict(EMPTY_METRICS),
        "active_metrics": {},
        "candidate_metrics": {},
        "policy": {},
        "split": {},
        "decision": {"status": "no_offline_report"},
        "warnings": [],
        "notes": ["No persisted offline evaluation report is available yet."],
    }


__all__ = [
    "REPORT_SCHEMA_VERSION",
    "EMPTY_METRICS",
    "new_report_id",
    "normalize_report",
    "save_evaluation_report",
    "load_latest_evaluation_report",
    "list_evaluation_reports",
    "unavailable_envelope",
]
