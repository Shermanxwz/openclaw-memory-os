#!/usr/bin/env python3
"""Replay feedback audit log entries and produce weight snapshots.

Reads the SQLite audit log, aggregates all ``action="feedback"`` entries
over the 24h / 7d / 30d time windows, and writes the computed useful
ratio to a weight snapshot file consumed by the ranking module.

Usage:

    scripts/replay_feedback.py [--db-path AUDIT_DB] [--out WEIGHTS_JSON]

Defaults match the OS runtime layout (``~/.local/state/...``) so the
script can be called from ``autonomous_governance.sh`` without arguments.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

STATE_DIR = Path(
    os.environ.get(
        "XDG_STATE_HOME",
        os.path.expanduser("~/.local/state"),
    )
) / "openclaw-memory-os"

DEFAULT_DB = STATE_DIR / "audit_log.sqlite"
DEFAULT_OUT = STATE_DIR / "feedback-weights.json"


def _default_db_path() -> Path:
    """Resolve the audit DB path, respecting the env override used by AuditStore."""
    env_path = os.environ.get("OPENCLAW_MEMORY_OS_AUDIT_PATH")
    if env_path:
        return Path(env_path)
    return DEFAULT_DB


def replay(db_path: Path) -> dict:
    """Read all ``action="feedback"`` entries and aggregate useful/total ratios.

    Returns a dict with keys ``ratio_24h``, ``ratio_7d``, ``ratio_30d``,
    ``total_useful``, ``total_not_useful``, and ``computed_at``.
    """
    import sqlite3

    if not db_path.exists():
        logger.warning("Audit DB not found at %s; returning empty weights", db_path)
        return {
            "ratio_24h": None,
            "ratio_7d": None,
            "ratio_30d": None,
            "total_useful": 0,
            "total_not_useful": 0,
            "computed_at": datetime.now(timezone.utc).isoformat(),
        }

    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            "SELECT timestamp, detail FROM audit_log WHERE action = ? ORDER BY id",
            ("feedback",),
        ).fetchall()
    finally:
        conn.close()

    now = datetime.now(timezone.utc)
    cutoff_24h = now - timedelta(hours=24)
    cutoff_7d = now - timedelta(days=7)
    cutoff_30d = now - timedelta(days=30)

    useful = 0
    not_useful = 0
    useful_24h = 0
    total_24h = 0
    useful_7d = 0
    total_7d = 0
    useful_30d = 0
    total_30d = 0

    for r in rows:
        ts_str = r["timestamp"]
        detail = r["detail"] or ""
        # Parse timestamp (ISO format from utcnow().isoformat())
        try:
            ts = datetime.fromisoformat(ts_str)
        except (ValueError, TypeError):
            continue
        is_useful = "useful=True" in detail or "useful=true" in detail.lower()
        if is_useful:
            useful += 1
        else:
            not_useful += 1

        if ts >= cutoff_24h:
            total_24h += 1
            if is_useful:
                useful_24h += 1
        if ts >= cutoff_7d:
            total_7d += 1
            if is_useful:
                useful_7d += 1
        if ts >= cutoff_30d:
            total_30d += 1
            if is_useful:
                useful_30d += 1

    def _ratio(useful_count: int, total_count: int) -> Optional[float]:
        if total_count == 0:
            return None
        return round(useful_count / total_count, 4)

    return {
        "ratio_24h": _ratio(useful_24h, total_24h),
        "ratio_7d": _ratio(useful_7d, total_7d),
        "ratio_30d": _ratio(useful_30d, total_30d),
        "total_useful": useful,
        "total_not_useful": not_useful,
        "computed_at": now.isoformat(),
    }


def write_weights(weights: dict, out_path: Path) -> Path:
    """Atomically write the weight snapshot with 0600 permissions."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(
        suffix=".tmp",
        prefix="feedback-weights-",
        dir=str(out_path.parent),
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2, ensure_ascii=False)
            f.write("\n")
        os.chmod(tmp, 0o600)
        os.replace(tmp, str(out_path))
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    logger.info("Weight snapshot written to %s", out_path)
    return out_path


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="Replay feedback and write weight snapshot")
    parser.add_argument(
        "--db-path",
        default=None,
        help="Path to the audit log SQLite DB (default: auto-detect)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output path for the weight snapshot JSON (default: ~/.local/state/.../feedback-weights.json)",
    )
    args = parser.parse_args(argv[1:] if len(argv) > 1 else [])

    db_path = Path(args.db_path) if args.db_path else _default_db_path()
    out_path = Path(args.out) if args.out else DEFAULT_OUT

    weights = replay(db_path)
    write_weights(weights, out_path)
    print(json.dumps(weights, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
