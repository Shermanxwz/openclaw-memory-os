#!/usr/bin/env python3
"""Write a tiny autonomous-governance status JSON for the dashboard.

This is a thin CLI wrapper around
``openclaw_memory_os.analytics.write_autonomous_governance_status``. The bash
runner calls it once at the end of a weekly deep-audit timer run and never
on the daily maintenance path.

Usage:

    write_governance_status.py <status_file> <result_token> [summary]

Arguments:

    status_file   Destination path. If the parent directory does not exist
                  it is created with mode 0700. The file itself is written
                  0600 via tempfile + os.replace.
    result_token  One of: ok / failed / degraded / running / pending / skipped. Anything else
                  is normalised to ``failed`` so the dashboard never
                  shows a bogus green state.
    summary       Short, single-line, redacted summary. Kept under 300
                  characters; longer inputs are truncated with ``...``.
                  Empty / missing argument is fine.

The output JSON is intentionally minimal:

    {"last_run": "...", "last_result": "...", "last_summary": "..."}

No collection names, file paths, IPs, tokens, or private metadata are
ever written. The writer contract is enforced inside
``write_autonomous_governance_status``.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

# Asia/Shanghai is the operator's wall-clock zone; mirror it in the
# fallback so the on-disk timestamp matches the in-app reader.  UTC+8
# is the fixed offset (no DST) and avoids depending on tzdata.
_SHANGHAI = timezone(timedelta(hours=8))

try:
    from openclaw_memory_os.analytics import write_autonomous_governance_status
except ImportError:
    # The runner is designed to be invoked from a project venv that has
    # openclaw_memory_os importable. On hosts where the runner is
    # deliberately invoked through a system Python that does not have
    # the package on the path (e.g. CI fixtures, recovery shells, or
    # the bash runner's missing-venv fallback), degrade to a stdlib-only
    # writer that produces the same 3-key schema. The full helper from
    # the package is still preferred because it owns the redactor and
    # permission-mode guarantees; this fallback only mirrors its
    # write shape so the dashboard contract holds either way.
    def _now_iso_shanghai() -> str:
        return datetime.now(_SHANGHAI).isoformat(timespec="microseconds")

    def _sanitize_summary(text: str) -> str:
        # Mirror the in-app redactor's surface behaviour: drop control
        # characters and collapse internal whitespace.
        if not text:
            return ""
        cleaned = "".join(ch for ch in text if ch == "\n" or ord(ch) >= 0x20)
        cleaned = " ".join(cleaned.split())
        if len(cleaned) > 300:
            cleaned = cleaned[:297] + "..."
        return cleaned

    def _allowed_token(token: str) -> str:
        return token if token in {"ok", "failed", "degraded", "running", "pending", "skipped"} else "failed"

    def write_autonomous_governance_status(
        status_file_path: Path,
        result_token: str = "ok",
        summary: str = "",
        finished_at: str | None = None,
        scheduled_at: str | None = None,
        started_at: str | None = None,
        duration_seconds: int | None = None,
        next_scheduled_at: str | None = None,
        exit_code: int | None = None,
        mode: str | None = None,
    ) -> Path:
        target = Path(status_file_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        os.chmod(target.parent, 0o700)
        payload = {
            "last_run": finished_at or _now_iso_shanghai(),
            "last_result": _allowed_token(result_token),
            "last_summary": _sanitize_summary(summary),
        }
        # Mirror analytics.py: only emit the Wave 2 extended fields
        # when the runner actually supplied ``started_at``. The legacy
        # 3-key contract holds otherwise.
        if started_at:
            payload["started_at"] = started_at
        if finished_at:
            payload["finished_at"] = finished_at
        if scheduled_at:
            payload["scheduled_at"] = scheduled_at
        if duration_seconds is not None:
            try:
                payload["duration_seconds"] = max(int(duration_seconds), 0)
            except (TypeError, ValueError):
                pass
        if next_scheduled_at:
            payload["next_scheduled_at"] = next_scheduled_at
        if exit_code is not None:
            try:
                payload["exit_code"] = int(exit_code)
            except (TypeError, ValueError):
                pass
        if mode:
            payload["mode"] = str(mode).strip().lower() or None
        fd, tmp_path = tempfile.mkstemp(
            prefix=target.name + ".", suffix=".tmp", dir=target.parent
        )
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                json.dump(payload, fh, ensure_ascii=False)
                fh.write("\n")
            os.chmod(tmp_path, 0o600)
            os.replace(tmp_path, target)
        except Exception:
            try:
                os.unlink(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return target


def main(argv: list[str]) -> int:
    if len(argv) < 3 or len(argv) > 4:
        sys.stderr.write(
            "usage: write_governance_status.py <status_file> <result_token> [summary]\n"
        )
        return 2

    status_file = Path(argv[1])
    result_token = argv[2]
    summary = argv[3] if len(argv) == 4 else ""

    # Wave 2 (2026-07-21): optional extended protocol. When the bash
    # runner (``autonomous_governance.sh``) sets
    # ``GOVERNANCE_STARTED_AT`` we pass through the additional fields
    # the dashboard reads (``scheduled_at`` / ``started_at`` /
    # ``finished_at`` / ``duration_seconds`` / ``next_scheduled_at`` /
    # ``exit_code`` / ``mode``). When the env vars are absent we keep
    # the legacy 3-key schema so existing callers (and tests) keep
    # working without changes.
    scheduled_at = os.environ.get("GOVERNANCE_SCHEDULED_AT") or None
    started_at = os.environ.get("GOVERNANCE_STARTED_AT") or None
    finished_at = os.environ.get("GOVERNANCE_FINISHED_AT") or None
    next_scheduled_at = os.environ.get("GOVERNANCE_NEXT_SCHEDULED_AT") or None
    duration_str = os.environ.get("GOVERNANCE_DURATION") or ""
    duration_seconds: int | None = None
    if duration_str:
        try:
            duration_seconds = max(int(duration_str), 0)
        except (TypeError, ValueError):
            duration_seconds = None
    exit_code_str = os.environ.get("GOVERNANCE_EXIT_CODE") or ""
    exit_code: int | None = None
    if exit_code_str:
        try:
            exit_code = int(exit_code_str)
        except (TypeError, ValueError):
            exit_code = None
    mode = os.environ.get("GOVERNANCE_MODE") or None

    try:
        written = write_autonomous_governance_status(
            status_file_path=status_file,
            result_token=result_token,
            summary=summary,
            # ``finished_at`` continues to drive ``last_run`` so the
            # 3-key contract stays backwards-compatible.
            finished_at=finished_at,
            scheduled_at=scheduled_at,
            started_at=started_at,
            duration_seconds=duration_seconds,
            next_scheduled_at=next_scheduled_at,
            exit_code=exit_code,
            mode=mode,
        )
    except OSError as exc:
        sys.stderr.write(f"write_governance_status: failed: {exc}\n")
        return 1

    # Single-line print so timer logs / ``grep`` pick it up cleanly. The
    # content is the destination path only — never the payload.
    print(f"governance_status_written: {written}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))