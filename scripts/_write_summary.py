#!/usr/bin/env python3
"""Write a maintenance summary JSON for the dashboard to read.

Usage:
    write_summary.py <log_file> <summary_file>

The summary file is written atomically via tempfile + os.replace so the dashboard
can read it without partial-state risk. The output keeps legacy top-level fields
for the dashboard while also exposing multi-collection details under
``collections`` and aggregate counters under ``totals``.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


START_RE = re.compile(r"\[maintenance\s+[\dTZ:.+-]+\]\s+(?:starting; collections:|starting maintenance)")
COLL_RE = re.compile(
    r"\[maintenance\s+[\dTZ:.+-]+\]\s+---\s+\[\d+/\d+\]\s+collection=(\S+)\s+---"
)
STEP_1_RE = re.compile(r"\[maintenance\s+[\dTZ:.+-]+\]\s+step 1/\d+:")
TIER_LOADED_RE = re.compile(r"\[tier\]\s+loaded\s+(\d+)\s+points", re.IGNORECASE)
EXPIRE_CANDIDATES_RE = re.compile(r"\[expire\]\s+candidates:\s+(\d+)", re.IGNORECASE)
SUPERSEDE_APPLIED_RE = re.compile(r"\[supersede\]\s+applied\s+(\d+)\s+supersede\s+links", re.IGNORECASE)
SNAPSHOT_NAME_RE = re.compile(r"snapshot\s+name:\s+(\S+)", re.IGNORECASE)
SIZE_RE = re.compile(r'"size":\s*(\d+)', re.IGNORECASE)

# Wave 2 (2026-07-21): new sub-step markers emitted by maintenance.sh and
# the memory-brain pipeline. They all share the
# ``[maintenance <ts> ...] [brain-...] run_id=... ...`` shape so a single
# set of regexes can pluck out run-scoped sub-step state from the log.
BRAIN_STEP_STARTED_RE = re.compile(
    r"\[maintenance\s+[\dTZ:.+-]+\]\s+\[brain-step\]\s+run_id=(\S+)\s+started=(\S+)"
)
# Wave 4 (2026-07-21): per-substep bracket markers emitted by
# ``scripts/memory_brain.py``. These carry independent started/finished
# timestamps for the ingest and consolidate leaves so the dashboard can
# render distinct substep runtimes instead of mirroring the parent
# ``[brain-step]`` window.
BRAIN_SUBSTEP_RE = re.compile(
    r"\[brain-substep\]\s+run_id=(\S+)\s+name=(\S+)\s+"
    r"started=(\S+)\s+finished=(\S+)\s+exit=(\d+)"
)
BRAIN_STEP_FINISHED_RE = re.compile(
    r"\[maintenance\s+[\dTZ:.+-]+\]\s+\[brain-step\]\s+run_id=(\S+)\s+finished=(\S+)\s+exit=(\d+)"
)
BRAIN_PIPELINE_RE = re.compile(
    r"\[brain-pipeline\]\s+run_id=(\S+)\s+status=(\S+)\s+ingest_exit=(\d+)\s+consolidate_exit=(\d+)"
)
BRAIN_INGEST_RE = re.compile(
    r"\[brain-ingest\]\s+run_id=(\S+)\s+files_processed=(\d+)\s+"
    r"total_ingested=(\d+)\s+total_skipped=(\d+)\s+error_queue=(\d+)\s+status=(\S+)"
)
BRAIN_CONSOLIDATE_OK_RE = re.compile(
    r"\[brain-consolidate\]\s+run_id=(\S+)\s+status=(\S+)\s+topics_merged=(\d+)\s+"
    r"merged_topics=(\d+)\s+threshold=(\d+)\s+"
    r"new_since_24h=(\d+)\s+total_points=(\d+)\s+dream_count=(\d+)"
)
# Reason can contain spaces (e.g. ``新增 0 < 20``), so we capture
# everything up to the next sentinel keyword. The skipped marker
# always carries ``new_since_24h=`` / ``total_points=`` /
# ``merged_topics=`` / ``threshold=`` / ``topics_merged=`` after the
# reason, so that gives us hard anchors even on the skipped path.
BRAIN_CONSOLIDATE_SKIP_RE = re.compile(
    r"\[brain-consolidate\]\s+run_id=(\S+)\s+status=(\S+)\s+reason=(.+?)\s+"
    r"new_since_24h=(\d+)\s+total_points=(\d+)\s+merged_topics=(\d+)\s+"
    r"threshold=(\d+)\s+topics_merged=(\d+)"
)
CHECKPOINT_ID_RE = re.compile(r'"checkpoint_id"\s*:\s*"([^"]+)"')

# Per-step status token set. Anything else the parser surfaces as
# ``failed`` so the dashboard never reads a fake green card.
_ALLOWED_STEP_STATUS = {"ok", "noop", "skipped", "failed", "degraded"}


def _empty_step() -> dict[str, Any]:
    """Shape of one entry in the ``steps`` sub-map.

    Mirrors the brief: status, started_at, finished_at, duration_seconds,
    exit_code, plus a free-form ``reason`` field for skipped steps. Each
    leaf module may add step-specific counters (chunks_scanned /
    expired_count / etc.) without touching this skeleton.
    """
    return {
        "status": None,
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "exit_code": None,
        "reason": None,
    }


def _empty_consolidation() -> dict[str, Any]:
    """Top-level ``consolidation`` aggregate read by the dashboard."""
    return {
        "status": None,
        "started_at": None,
        "finished_at": None,
        "duration_seconds": None,
        "new_since_24h": 0,
        "total_points": 0,
        "topics_merged": 0,
        "merged_topics": 0,
        "threshold": 20,
        "reason": "",
        "run_id": None,
    }


def parse_substeps(block: list[str]) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    """Walk the maintenance log block and build per-step status + consolidation.

    Returns ``(steps, consolidation)`` where ``steps`` is keyed by
    ``memory_brain`` / ``ingest`` / ``reclassify`` / ``supersede`` /
    ``expire`` / ``snapshot`` / ``lexical_refresh`` and ``consolidation``
    is the top-level aggregate.

    Anything we cannot prove from the log stays ``None`` (status) /
    ``0`` (counters) so a partially-populated summary is still honest
    about which steps actually ran. This is the Wave 2 contract: the
    dashboard never reads a fabricated green tick from an empty sub-step.
    """
    steps: dict[str, dict[str, Any]] = {
        "memory_brain": _empty_step(),
        "ingest": {
            "status": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "chunks_scanned": 0,
            "ingested_new": 0,
            "checkpoint_id": None,
        },
        "reclassify": _empty_step(),
        "supersede": {
            "status": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "links_applied": 0,
        },
        "expire": {
            "status": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "candidates": 0,
        },
        "snapshot": {
            "status": None,
            "started_at": None,
            "finished_at": None,
            "exit_code": None,
            "name": None,
            "size_bytes": 0,
        },
        "lexical_refresh": _empty_step(),
    }
    consolidation = _empty_consolidation()

    brain_started_at: str | None = None
    brain_finished_at: str | None = None
    brain_exit_code: int | None = None
    ingest_checkpoint_id: str | None = None

    def _normalise_status(raw: str) -> str:
        raw = (raw or "").strip().lower()
        return raw if raw in _ALLOWED_STEP_STATUS else "failed"

    for raw_line in block:
        line = raw_line.strip()

        # --- memory-brain unified pipeline bracketing ------------------
        # Note: finished= marker is matched BEFORE the brain_step_active
        # guard so the timestamp is captured even when the brain-pipeline
        # marker (which writes to ``steps.memory_brain``) appears in the
        # log between the ``started=`` and ``finished=`` markers.
        # Wave 4 (2026-07-21): per-substep markers take priority. They
        # are processed BEFORE the parent ``[brain-step]`` markers so
        # the leaf timestamps win whenever both are present (which is
        # the normal unified-pipeline case). The parent bracket still
        # feeds ``steps.memory_brain`` because the dashboard surfaces it
        # as the unified pipeline window.
        m = BRAIN_SUBSTEP_RE.search(line)
        if m:
            sub_run_id = m.group(1)
            sub_name = m.group(2)
            sub_started = m.group(3)
            sub_finished = m.group(4)
            try:
                sub_exit = int(m.group(5))
            except (TypeError, ValueError):
                sub_exit = None
            if sub_name == "ingest":
                # Independent bracket — never mirror the parent step.
                steps["ingest"]["started_at"] = sub_started
                steps["ingest"]["finished_at"] = sub_finished
                if steps["ingest"].get("exit_code") is None and sub_exit is not None:
                    steps["ingest"]["exit_code"] = sub_exit
                if not steps["ingest"].get("run_id"):
                    steps["ingest"]["run_id"] = sub_run_id
            elif sub_name == "consolidate":
                # Independent bracket — never mirror the parent step.
                consolidation["started_at"] = sub_started
                consolidation["finished_at"] = sub_finished
                consolidation["run_id"] = sub_run_id
            # Continue to the next line so we don't accidentally fire
            # the parent brain-step branch with a stale bracket.
            continue
        m = BRAIN_STEP_STARTED_RE.search(line)
        if m:
            brain_step_active = True
            brain_step_run_id = m.group(1)
            brain_step_started_at = m.group(2)
            brain_started_at = m.group(2)
            # Wave 4: only ``steps.memory_brain`` mirrors the parent
            # bracket. The leaf steps (ingest / consolidation) get
            # their independent timestamps from the BRAIN_SUBSTEP_RE
            # branch above, so we no longer fan the parent started_at
            # into the children here.
            steps["memory_brain"]["started_at"] = m.group(2)
            continue
        m = BRAIN_STEP_FINISHED_RE.search(line)
        if m:
            brain_finished_at = m.group(2)
            try:
                brain_exit_code = int(m.group(3))
            except (TypeError, ValueError):
                brain_exit_code = None
            brain_step_active = False
            # Only mirror the finished_at onto ``steps.memory_brain``
            # (the parent window). The leaf steps keep the timestamps
            # they got from BRAIN_SUBSTEP_RE.
            if steps["memory_brain"]["status"] is not None:
                steps["memory_brain"]["finished_at"] = brain_finished_at
                if steps["memory_brain"].get("exit_code") is None:
                    steps["memory_brain"]["exit_code"] = (
                        0 if brain_exit_code == 0 else (brain_exit_code or 1)
                    )
            continue
        m = BRAIN_PIPELINE_RE.search(line)
        if m:
            try:
                ingest_exit = int(m.group(3))
            except (TypeError, ValueError):
                ingest_exit = None
            try:
                consolidate_exit = int(m.group(4))
            except (TypeError, ValueError):
                consolidate_exit = None
            token = _normalise_status(m.group(2))
            steps["memory_brain"]["status"] = token
            steps["memory_brain"]["exit_code"] = (
                0 if token in {"ok", "skipped", "noop"} else (brain_exit_code or 1)
            )
            # Bracket timestamps come from the [brain-step] markers above.
            steps["memory_brain"]["started_at"] = brain_started_at
            steps["memory_brain"]["finished_at"] = brain_finished_at
            # ``sub_run_id`` is the leaf-level run correlation token: when
            # the unified pipeline fires both ingest and consolidate,
            # they share the parent run_id and the leaf modules each
            # carry it through their stdout markers. We surface the
            # pipeline-level run_id here so the dashboard can join the
            # step row back to ``run_id`` at the top level.
            steps["memory_brain"]["sub_run_id"] = m.group(1)
            steps["memory_brain"]["ingest_exit_code"] = ingest_exit
            steps["memory_brain"]["consolidate_exit_code"] = consolidate_exit
            continue

        # --- ingest sub-step -------------------------------------------
        m = BRAIN_INGEST_RE.search(line)
        if m:
            try:
                total_ingested = int(m.group(3))
                total_skipped = int(m.group(4))
                files_processed = int(m.group(2))
            except (TypeError, ValueError):
                total_ingested = total_skipped = files_processed = 0
            steps["ingest"]["status"] = _normalise_status(m.group(6))
            steps["ingest"]["exit_code"] = 0
            steps["ingest"]["ingested_new"] = total_ingested
            steps["ingest"]["skipped"] = total_skipped
            steps["ingest"]["files_processed"] = files_processed
            # Wave 4: ingest bracket comes from BRAIN_SUBSTEP_RE, not
            # from the parent ``brain_started_at`` mirror. We only fall
            # back to the parent when the substep marker is missing
            # (i.e. an ad-hoc legacy invocation).
            if not steps["ingest"].get("started_at"):
                steps["ingest"]["started_at"] = brain_started_at
            if not steps["ingest"].get("finished_at"):
                steps["ingest"]["finished_at"] = brain_finished_at
            steps["ingest"]["sub_run_id"] = m.group(1)
            continue

        # --- consolidation sub-step (skipped / ok) ----------------------
        # The ok marker shape carries topics_merged right after status;
        # the skipped marker carries reason in the same position. We
        # try the ok shape first because it has more anchors.
        m = BRAIN_CONSOLIDATE_OK_RE.search(line)
        if m:
            consolidation["run_id"] = m.group(1)
            consolidation["status"] = _normalise_status(m.group(2))
            try:
                consolidation["topics_merged"] = int(m.group(3))
            except (TypeError, ValueError):
                consolidation["topics_merged"] = 0
            try:
                consolidation["merged_topics"] = int(m.group(4))
            except (TypeError, ValueError):
                consolidation["merged_topics"] = consolidation["topics_merged"]
            try:
                consolidation["threshold"] = int(m.group(5))
            except (TypeError, ValueError):
                consolidation["threshold"] = 20
            try:
                consolidation["new_since_24h"] = int(m.group(6))
            except (TypeError, ValueError):
                consolidation["new_since_24h"] = 0
            try:
                consolidation["total_points"] = int(m.group(7))
            except (TypeError, ValueError):
                consolidation["total_points"] = 0
            # Wave 4: consolidation bracket comes from BRAIN_SUBSTEP_RE
            # (independent). Fall back to the parent ``brain_*`` values
            # only when the substep marker is missing (legacy / ad-hoc
            # invocations).
            if not consolidation.get("started_at"):
                consolidation["started_at"] = brain_started_at
            if not consolidation.get("finished_at"):
                consolidation["finished_at"] = brain_finished_at
            consolidation["reason"] = ""
            continue
        m = BRAIN_CONSOLIDATE_SKIP_RE.search(line)
        if m:
            consolidation["run_id"] = m.group(1)
            consolidation["status"] = _normalise_status(m.group(2))
            consolidation["reason"] = m.group(3)
            try:
                consolidation["new_since_24h"] = int(m.group(4))
            except (TypeError, ValueError):
                consolidation["new_since_24h"] = 0
            try:
                consolidation["total_points"] = int(m.group(5))
            except (TypeError, ValueError):
                consolidation["total_points"] = 0
            try:
                consolidation["merged_topics"] = int(m.group(6))
            except (TypeError, ValueError):
                consolidation["merged_topics"] = 0
            consolidation["topics_merged"] = consolidation["merged_topics"]
            try:
                consolidation["threshold"] = int(m.group(7))
            except (TypeError, ValueError):
                consolidation["threshold"] = 20
            # Bracket timestamps come from the [brain-step] markers; the
            # skipped path may fire before they print, so fall through to
            # the existing orient line + the bottom-of-function duration
            # fill so the dashboard always reads a complete bracket.
            consolidation["started_at"] = consolidation.get("started_at") or brain_started_at
            consolidation["finished_at"] = consolidation.get("finished_at") or brain_finished_at
            continue

        # --- orient(): total_points + new_since_24h are printed by the
        # consolidate script before the trigger check. We pick them up
        # from the log without re-running orient ourselves.
        if "Phase 1: Orient" in line or "总记忆" in line:
            # Pattern: "总记忆: 321 | 24h新增: 0 | MEMORY.md: ..."
            totals_m = re.search(r"总记忆:\s*(\d+)\s*\|\s*24h新增:\s*(\d+)", line)
            if totals_m:
                consolidation["total_points"] = int(totals_m.group(1))
                consolidation["new_since_24h"] = int(totals_m.group(2))

        # --- ingest JSON block (existing checkpoint_id extraction) ----
        if '"checkpoint_id"' in line:
            ck = CHECKPOINT_ID_RE.search(line)
            if ck:
                ingest_checkpoint_id = ck.group(1)
                steps["ingest"]["checkpoint_id"] = ck.group(1)

        # --- existing per-collection counters --------------------------
        if m := SUPERSEDE_APPLIED_RE.search(line):
            try:
                steps["supersede"]["links_applied"] += int(m.group(1))
            except (TypeError, ValueError):
                pass
        if m := EXPIRE_CANDIDATES_RE.search(line):
            try:
                steps["expire"]["candidates"] += int(m.group(1))
            except (TypeError, ValueError):
                pass
        if m := SNAPSHOT_NAME_RE.search(line):
            steps["snapshot"]["name"] = m.group(1)
        if m := SIZE_RE.search(line):
            try:
                steps["snapshot"]["size_bytes"] = int(m.group(1))
            except (TypeError, ValueError):
                pass
        if "[snapshot] ok:" in line:
            steps["snapshot"]["status"] = "ok"

        # --- legacy maintenance.sh log markers -------------------------
        # The maintenance.sh runner still emits one-line "  step N/N:
        # <name>" markers; we treat those as ok unless the run failed.
        # Sub-step exit codes stay at None unless we have a matching
        # failure line further down (the runner logs ``mark_failure``
        # separately).
        if "  step 1/" in line and "ingest" in line.lower():
            if steps["ingest"]["status"] is None:
                steps["ingest"]["status"] = "ok"
        if "  step 2/" in line and ("reclassif" in line.lower() or "tier" in line.lower()):
            if steps["reclassify"]["status"] is None:
                steps["reclassify"]["status"] = "ok"
        if "  step 3/" in line and "supersede" in line.lower():
            if steps["supersede"]["status"] is None:
                steps["supersede"]["status"] = "ok"
        if "  step 4/" in line and "expire" in line.lower():
            if steps["expire"]["status"] is None:
                steps["expire"]["status"] = "ok"
        if "  step 5/" in line and "snapshot" in line.lower():
            if steps["snapshot"]["status"] is None:
                steps["snapshot"]["status"] = "ok"
        if "  lexical refresh: ok" in line:
            if steps["lexical_refresh"]["status"] is None:
                steps["lexical_refresh"]["status"] = "ok"

        # Failure markers: when the runner logs ``completed with
        # failures=N failed_step=<name>``, mark every step the failure
        # text mentions as failed and leave the rest as ok.
        if "completed with failures=" in line:
            failed_step = None
            fm = re.search(r"failed_step=([^\s]+)", line)
            if fm:
                failed_step = fm.group(1)
            if failed_step:
                lower = failed_step.lower()
                for key, needles in (
                    ("ingest", ("ingest failed", "ingest")),
                    ("reclassify", ("reclassif", "tier")),
                    ("supersede", ("supersede",)),
                    ("expire", ("expir",)),
                    ("snapshot", ("snapshot",)),
                    ("lexical_refresh", ("lexical",)),
                    ("memory_brain", ("memory-brain", "memory brain", "brain")),
                ):
                    if any(n in lower for n in needles):
                        if steps[key]["status"] in (None, "ok"):
                            steps[key]["status"] = "failed"

    # Sub-step correlation: if the run_id we read from maintenance.sh
    # env matches the brain-pipeline marker, attach it to the parent
    # step so the dashboard can join ``steps.memory_brain.sub_run_id``
    # back to the top-level ``run_id``.
    if steps["memory_brain"].get("sub_run_id") and brain_step_run_id:
        steps["memory_brain"]["run_id"] = brain_step_run_id

    # If ingest ran without going through the unified brain pipeline,
    # still surface any checkpoint_id we discovered from the legacy
    # JSON line so the dashboard has something to correlate on.
    if ingest_checkpoint_id and not steps["ingest"].get("checkpoint_id"):
        steps["ingest"]["checkpoint_id"] = ingest_checkpoint_id

    # Compute ``duration_seconds`` for every sub-step + the consolidation
    # aggregate so the dashboard can render it directly without having
    # to subtract timestamps on the client. The contract is
    # duration_seconds = finished_at - started_at (in whole seconds,
    # 0 when both sides resolve to the same instant). Skipped steps
    # still get a duration (>= 0); ``None`` is reserved for steps that
    # never produced any bracket markers.
    _fill_duration(steps["memory_brain"])
    _fill_duration(steps["ingest"])
    _fill_duration(consolidation)
    _fill_duration(steps["reclassify"])
    _fill_duration(steps["supersede"])
    _fill_duration(steps["expire"])
    _fill_duration(steps["snapshot"])
    _fill_duration(steps["lexical_refresh"])

    return steps, consolidation


def _fill_duration(step: dict[str, Any]) -> None:
    """Populate ``duration_seconds`` on a step dict if both timestamps resolve.

    Leaves the field ``None`` when either timestamp is missing — that is
    the canonical "no run" signal and the dashboard reads it as an
    explicit error rather than as a quiet zero. Sub-second intervals are
    kept as a float (``0.4128``) rather than rounded to zero so the
    dashboard can still render a meaningful "0.4s" tick.
    """
    started = step.get("started_at")
    finished = step.get("finished_at")
    if not started or not finished:
        return
    try:
        from datetime import datetime as _dt  # local import keeps the module cheap
        s = _dt.fromisoformat(started.replace("Z", "+00:00"))
        f = _dt.fromisoformat(finished.replace("Z", "+00:00"))
        delta = (f - s).total_seconds()
        step["duration_seconds"] = max(round(delta, 3), 0.0)
    except Exception:
        # If either timestamp is unparseable, leave duration_seconds
        # at ``None`` so the dashboard knows the source data is broken
        # instead of inventing a silent zero.
        return


def find_last_ok_block(lines: list[str]) -> list[str]:
    """Find the most recent maintenance block in the log.

    ``maintenance.sh`` writes the summary *before* it prints the final
    ``[maintenance ...] ok`` line. During that moment the newest run has a
    ``starting; collections:`` marker but no ``ok`` marker yet. If that newest
    start is after the newest ok, parse the active run; otherwise parse the
    newest completed ok block.
    """
    latest_start = None
    latest_step = None
    latest_ok = None
    for idx, line in enumerate(lines):
        if START_RE.match(line):
            latest_start = idx
        if STEP_1_RE.match(line):
            latest_step = idx
        if line.startswith("[maintenance ") and line.rstrip().endswith("ok"):
            latest_ok = idx

    active_start = latest_start if latest_start is not None else latest_step
    if active_start is not None and (latest_ok is None or active_start > latest_ok):
        return lines[active_start:]

    if latest_ok is not None:
        for j in range(latest_ok - 1, -1, -1):
            if START_RE.match(lines[j]):
                return lines[j : latest_ok + 1]
        for j in range(latest_ok - 1, -1, -1):
            if STEP_1_RE.match(lines[j]):
                return lines[j : latest_ok + 1]

    return []


def extract_ingestion_json(block: list[str]) -> dict[str, Any]:
    """Extract the first multi-line JSON object containing total_chunks/written."""
    in_json = False
    json_parts: list[str] = []
    depth = 0
    for line in block:
        stripped = line.strip()
        if stripped.startswith("{") and not in_json:
            in_json = True
            json_parts = [line]
            depth = stripped.count("{") - stripped.count("}")
        elif in_json:
            json_parts.append(line)
            depth += stripped.count("{") - stripped.count("}")
        else:
            continue
        if in_json and depth > 0:
            continue
        # depth == 0 means we just closed an object; reset state.
        in_json = False
        if not json_parts:
            continue
        try:
            payload = json.loads("\n".join(json_parts))
            if "total_chunks" in payload or "written" in payload:
                return payload
        except (json.JSONDecodeError, ValueError):
            pass
        json_parts = []
    return {}


def _empty_collection() -> dict[str, Any]:
    return {
        "ingest_chunks": 0,
        "ingested_new": 0,
        "ingest_skipped": False,
        "points_scanned": 0,
        "expired_count": 0,
        "superseded_links": 0,
        "snapshot_name": None,
        "snapshot_size_bytes": 0,
        "snapshot_ok": False,
    }


def _first_json_payload(lines: list[str]) -> tuple[dict[str, Any], int]:
    """Return ``(payload, end_index)`` for the first JSON object in ``lines``.

    Used by :func:`parse_collection_summaries` so we can match an
    ingestion JSON block to the collection it belongs to. The caller
    slices off the consumed prefix so subsequent lines (snapshot /
    supersede markers) attach to the same collection.
    """
    in_json = False
    json_parts: list[str] = []
    depth = 0
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if not in_json:
            if stripped.startswith("{"):
                in_json = True
                json_parts = [line]
                depth = stripped.count("{") - stripped.count("}")
            else:
                continue
        else:
            json_parts.append(line)
            depth += stripped.count("{") - stripped.count("}")
        if in_json and depth > 0:
            continue
        # depth == 0 means we just closed an object; reset state.
        in_json = False
        if not json_parts:
            continue
        try:
            payload = json.loads("\n".join(json_parts))
            return payload, idx + 1
        except (json.JSONDecodeError, ValueError):
            json_parts = []
    return {}, -1


def _iter_collection_blocks(
    block: list[str],
) -> list[tuple[str, list[str]]]:
    """Split a maintenance log block into ``(collection, lines)`` chunks.

    The collector splits on the ``--- [i/N] collection=NAME ---`` marker
    that ``maintenance.sh`` emits. Pre-multi-collection logs (no marker)
    fall back to a single ``openclaw_memory_os`` chunk so legacy files
    still parse.
    """
    chunks: list[tuple[str, list[str]]] = []
    current_name: str | None = None
    current_lines: list[str] = []
    saw_marker = False
    for line in block:
        m = COLL_RE.match(line)
        if m:
            saw_marker = True
            if current_name is not None:
                chunks.append((current_name, current_lines))
            current_name = m.group(1)
            current_lines = []
        else:
            if current_name is None:
                current_name = "openclaw_memory_os"
                current_lines = []
            current_lines.append(line)
    if current_name is not None:
        chunks.append((current_name, current_lines))
    if not saw_marker and chunks == []:
        # Entirely marker-less block (older single-collection logs).
        chunks = [("openclaw_memory_os", list(block))]
    return chunks


def parse_collection_summaries(
    block: list[str],
) -> dict[str, dict[str, Any]]:
    """Walk a maintenance log block and return per-collection summaries.

    Each collection gets a fresh dict from :func:`_empty_collection`
    that the parser fills in. The result preserves collection order
    because Python dicts keep insertion order.
    """
    out: dict[str, dict[str, Any]] = {}
    for name, lines in _iter_collection_blocks(block):
        summary = _empty_collection()
        out[name] = summary

        # Extract the first ingestion JSON inside this collection's slice.
        payload, _ = _first_json_payload(lines)
        if "total_chunks" in payload or "written" in payload:
            summary["ingest_chunks"] = int(payload.get("total_chunks", 0) or 0)
            summary["ingested_new"] = int(payload.get("written", 0) or 0)

        for line in lines:
            lower = line.lower()
            if "ingest (skipped" in lower:
                summary["ingest_skipped"] = True
            if m := TIER_LOADED_RE.search(line):
                summary["points_scanned"] += int(m.group(1))
            if m := EXPIRE_CANDIDATES_RE.search(line):
                summary["expired_count"] += int(m.group(1))
            if m := SUPERSEDE_APPLIED_RE.search(line):
                summary["superseded_links"] += int(m.group(1))
            if m := SNAPSHOT_NAME_RE.search(line):
                summary["snapshot_name"] = m.group(1)
            if m := SIZE_RE.search(line):
                summary["snapshot_size_bytes"] = int(m.group(1))
            if "[snapshot] ok:" in line:
                summary["snapshot_ok"] = True

    return out


def build_totals(collections: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-collection summaries into top-level totals.

    Totals reflect the union across all collections for the most recent
    maintenance run, which is what the dashboard cards want to display.
    """
    totals = {
        "ingest_chunks": 0,
        "ingested_new": 0,
        "points_scanned": 0,
        "snapshots_ok": 0,
        "snapshot_count": 0,
        "superseded_links_total": 0,
        "expired_count_total": 0,
    }
    for summary in collections.values():
        totals["ingest_chunks"] += int(summary.get("ingest_chunks") or 0)
        totals["ingested_new"] += int(summary.get("ingested_new") or 0)
        totals["points_scanned"] += int(summary.get("points_scanned") or 0)
        totals["expired_count_total"] += int(summary.get("expired_count") or 0)
        totals["superseded_links_total"] += int(
            summary.get("superseded_links") or 0
        )
        if summary.get("snapshot_name"):
            totals["snapshot_count"] += 1
        if summary.get("snapshot_ok"):
            totals["snapshots_ok"] += 1
    return totals


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        sys.stderr.write("usage: write_summary.py <log_file> <summary_file>\n")
        return 2
    log_path = Path(argv[1])
    out_path = Path(argv[2])

    # Status fields populated by maintenance.sh via env vars.
    # maintenance.sh ALWAYS sets these before invoking us so we never
    # silently fall back to "ok". Defaults are only used when invoked
    # outside the maintenance.sh flow (e.g. legacy tests).
    status = os.environ.get("MAINTENANCE_STATUS") or "ok"
    exit_code_str = os.environ.get("MAINTENANCE_EXIT_CODE") or "0"
    failed_step = os.environ.get("MAINTENANCE_FAILED_STEP") or None
    started_at = os.environ.get("MAINTENANCE_STARTED_AT") or datetime.now(timezone.utc).isoformat()
    finished_at = os.environ.get("MAINTENANCE_FINISHED_AT") or datetime.now(timezone.utc).isoformat()

    # Wave 2 (2026-07-21): shared run_id + mode the runner propagates via
    # env so every sub-step status lands under one canonical key.
    # ``MAINTENANCE_RUN_ID`` / ``MAINTENANCE_MODE`` are unset when this
    # writer is called outside the maintenance.sh flow (legacy tests,
    # one-off operator runs); the defaults keep the legacy summary
    # schema intact for those callers.
    run_id = os.environ.get("MAINTENANCE_RUN_ID") or None
    mode = os.environ.get("MAINTENANCE_MODE") or "daily"

    # Read the previous summary to find last_success_at (do not regress).
    last_success_at = None
    try:
        if out_path.exists():
            prev = json.loads(out_path.read_text(encoding="utf-8"))
            last_success_at = prev.get("last_success_at")
    except (OSError, ValueError):
        pass

    out: dict[str, Any] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "exit_code": int(exit_code_str),
        "failed_step": failed_step,
        "last_success_at": last_success_at,
        # Wave 2 (2026-07-21) — additive fields. Defaulted to None so
        # legacy / non-maintenance callers do not break.
        "run_id": run_id,
        "mode": mode,
        # Per-step status populated by ``parse_substeps`` below. The
        # default skeleton matches the brief so the dashboard can read
        # the keys unconditionally even when no log was found.
        "steps": {
            "memory_brain": _empty_step(),
            "ingest": {
                "status": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "chunks_scanned": 0,
                "ingested_new": 0,
                "checkpoint_id": None,
            },
            "reclassify": _empty_step(),
            "supersede": {
                "status": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "links_applied": 0,
            },
            "expire": {
                "status": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "candidates": 0,
            },
            "snapshot": {
                "status": None,
                "started_at": None,
                "finished_at": None,
                "exit_code": None,
                "name": None,
                "size_bytes": 0,
            },
            "lexical_refresh": _empty_step(),
        },
        # Top-level consolidation aggregate the dashboard reads
        # directly (no per-collection breakdown needed).
        "consolidation": _empty_consolidation(),
        "ingested_total": 0,
        "ingested_new": 0,
        "chunks_scanned": 0,
        "expired_count": 0,
        "superseded_links": 0,
        "snapshot_name": None,
        "snapshot_size_bytes": 0,
        "collections": {},
        "totals": {
            "ingest_chunks": 0,
            "ingested_new": 0,
            "points_scanned": 0,
            "expired_count_total": 0,
            "superseded_links_total": 0,
            "snapshots_ok": 0,
            "snapshot_count": 0,
        },
    }

    try:
        lines = [ln.rstrip() for ln in log_path.open("r", errors="replace")]
    except (OSError, PermissionError):
        lines = []

    block = find_last_ok_block(lines)
    if block:
        collections = parse_collection_summaries(block)
        totals = build_totals(collections)
        out["collections"] = collections
        out["totals"] = totals

        # Wave 2 (2026-07-21): sub-step status + consolidation aggregate.
        # These are additive on top of the legacy top-level fields so
        # the dashboard keeps reading ``ingested_total`` /
        # ``chunks_scanned`` etc. unchanged.
        steps, consolidation = parse_substeps(block)
        out["steps"] = steps
        out["consolidation"] = consolidation

        # If the per-step ingest counter wasn't populated by the brain
        # markers (e.g. maintenance ran without ``ENABLE_MEMORY_BRAIN=1``
        # and the cli ingest path wrote the legacy JSON block), fall
        # back to the aggregate ingest JSON the parser already pulled.
        if steps["ingest"]["status"] is None:
            payload = extract_ingestion_json(block)
            if payload:
                try:
                    steps["ingest"]["chunks_scanned"] = int(
                        payload.get("total_chunks", 0) or 0
                    )
                except (TypeError, ValueError):
                    pass
                try:
                    steps["ingest"]["ingested_new"] = int(
                        payload.get("written", 0) or 0
                    )
                except (TypeError, ValueError):
                    pass
                ck = payload.get("checkpoint_id")
                if ck:
                    steps["ingest"]["checkpoint_id"] = ck
                steps["ingest"]["status"] = "ok"

        # Legacy field mapping for the existing dashboard cards.
        # Prefer the first ingestion JSON we found (so the "memory file
        # ingest" card still shows the right per-collection chunk
        # count for the first/main collection). If no JSON was found
        # (e.g. all collections skipped ingest), fall back to the
        # aggregate totals so the dashboard never reads zeros from a
        # real run.
        payload = extract_ingestion_json(block)
        if payload:
            scanned = int(payload.get("total_chunks", 0) or 0)
            written = int(payload.get("written", 0) or 0)
        else:
            scanned = totals["ingest_chunks"]
            written = totals["ingested_new"]
        out["ingested_total"] = scanned
        out["ingested_new"] = written
        out["chunks_scanned"] = scanned
        out["expired_count"] = totals["expired_count_total"]
        out["superseded_links"] = totals["superseded_links_total"]

        # Snapshot: legacy field always shows the LAST one that has a name.
        # ``snapshot_ok`` is set by the new ``[snapshot] ok:`` log line in
        # ``backup_snapshot.sh``; for older logs that don't emit it we
        # treat a snapshot as present whenever a name and size were
        # captured. This keeps the dashboard's "最近快照" card populated
        # for legacy single-collection runs.
        last_snapshot_name: str | None = None
        last_snapshot_size = 0
        for summary in collections.values():
            name = summary.get("snapshot_name")
            if not name:
                continue
            last_snapshot_name = name
            last_snapshot_size = int(summary.get("snapshot_size_bytes") or 0)
        if last_snapshot_name is not None:
            out["snapshot_name"] = last_snapshot_name
            out["snapshot_size_bytes"] = last_snapshot_size

    # On a successful run, capture finished_at as last_success_at.
    if out["status"] == "success" and out["exit_code"] == 0:
        out["last_success_at"] = out["finished_at"]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    try:
        with tmp.open("w", encoding="utf-8") as fh:
            json.dump(out, fh, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, out_path)
    except OSError as exc:
        sys.stderr.write(f"write_summary: failed to write {out_path}: {exc}\n")
        return 1
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass

    print(f"summary written: {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
