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


START_RE = re.compile(r"\[maintenance\s+[\dTZ:.+-]+\]\s+starting; collections:")
COLL_RE = re.compile(
    r"\[maintenance\s+[\dTZ:.+-]+\]\s+---\s+\[\d+/\d+\]\s+collection=(\S+)\s+---"
)
STEP_1_RE = re.compile(r"\[maintenance\s+[\dTZ:.+-]+\]\s+step 1/\d+:")
TIER_LOADED_RE = re.compile(r"\[tier\]\s+loaded\s+(\d+)\s+points", re.IGNORECASE)
EXPIRE_CANDIDATES_RE = re.compile(r"\[expire\]\s+candidates:\s+(\d+)", re.IGNORECASE)
SUPERSEDE_APPLIED_RE = re.compile(r"\[supersede\]\s+applied\s+(\d+)\s+supersede\s+links", re.IGNORECASE)
SNAPSHOT_NAME_RE = re.compile(r"snapshot\s+name:\s+(\S+)", re.IGNORECASE)
SIZE_RE = re.compile(r'"size":\s*(\d+)', re.IGNORECASE)


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

    out: dict[str, Any] = {
        "last_run": datetime.now(timezone.utc).isoformat(),
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
