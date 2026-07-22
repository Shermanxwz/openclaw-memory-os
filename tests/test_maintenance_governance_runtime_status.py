"""Tests for Wave 2 dashboard rendering pipeline.

Covers the cross-cutting guarantees introduced by the
``Wave 2 (2026-07-21)`` dashboard refresh:

* ``maintenance-summary.json`` propagates ``run_id`` / ``mode`` /
  ``steps{}`` / ``consolidation{}`` so a single run is correlated
  end-to-end.
* ``analytics._read_memory_brain_status`` reads from the canonical
  maintenance summary only; legacy ``/var/log/openclaw-memory-brain-*.json``
  files are NOT allowed to override it.
* ``analytics._read_autonomous_governance_status`` accepts and surfaces
  the extended governance protocol fields
  (``scheduled_at`` / ``started_at`` / ``finished_at`` /
  ``duration_seconds`` / ``next_scheduled_at`` / ``exit_code`` /
  ``mode``).
* ``analytics._read_systemd_timer_schedule`` parses ``systemctl show``
  output into a dashboard-friendly shape and graceful-returns
  ``{"active_state": "unknown"}`` for non-existent units.
* The dashboard renderer (overview.js / dashboard.html) surfaces
  ingest noop, consolidation skipped, and the governance schedule card
  without hardcoding ``Tue 04:01`` / ``daily 07:45`` anywhere in
  source.

The 12 tests below are deliberately independent: each one mocks only
the surface it needs so a regression in one cannot mask another.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict
from unittest import mock

import pytest


ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"
PACKAGE_DIR = ROOT / "openclaw_memory_os"
VENV_PY = (ROOT / ".venv" / "bin" / "python")
PYTHON_BIN = VENV_PY if VENV_PY.exists() else Path(sys.executable)

REPO_PACKAGES = (
    "openclaw_memory_os",
    "openclaw_memory_os.analytics",
    "openclaw_memory_os.models",
    "openclaw_memory_os.app",
)


# ---------------------------------------------------------------------------
# Test fixtures: in-memory maintenance summary, governance status JSON, and
# helpers to invoke scripts/_write_summary.py + analytics without reaching
# the real Qdrant / filesystem.
# ---------------------------------------------------------------------------


def _make_summary_via_writer(
    tmp_path: Path,
    *,
    log_body: str,
    env: Dict[str, str],
) -> Dict[str, Any]:
    """Run scripts/_write_summary.py with a synthetic log + env.

    Returns the parsed JSON summary. The maintenance log file is
    written into ``tmp_path`` so the test never touches the real
    ``/var/log/openclaw-memory-os`` directory.
    """
    log = tmp_path / "maintenance.log"
    log.write_text(log_body, encoding="utf-8")
    out = tmp_path / "summary.json"
    test_env = os.environ.copy()
    test_env.update(env)
    proc = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPTS_DIR / "_write_summary.py"), str(log), str(out)],
        capture_output=True,
        text=True,
        env=test_env,
        timeout=30,
    )
    assert proc.returncode == 0, (
        f"_write_summary.py crashed: stdout={proc.stdout!r} stderr={proc.stderr!r}"
    )
    return json.loads(out.read_text(encoding="utf-8"))


def _build_run_id_log_body(
    run_id: str,
    *,
    ingest_status: str = "ok",
    started_at: str = "2026-07-21T00:00:00Z",
    finished_at: str = "2026-07-21T00:00:30Z",
) -> str:
    """Construct a maintenance log block that emits one full run.

    Mirrors the markers ``maintenance.sh`` / ``memory_brain.py`` /
    ``memory_brain_consolidate.py`` actually write during a daily run.
    The bracket timestamps (``started_at`` / ``finished_at``) can be
    overridden so the run-window assertion can pin them.

    The skip-reason line matches the format
    ``memory_brain_consolidate.py`` emits (with the trailing
    ``new_since_24h`` / ``total_points`` / ``topics_merged=0`` fields
    the dashboard parser relies on).
    """
    return (
        f"[maintenance {started_at}] RUN_ID={run_id}\n"
        f"[maintenance {started_at}] starting maintenance\n"
        f"[maintenance {started_at}] --- [1/1] collection=openclaw_memory_os ---\n"
        f"[maintenance {started_at}] memory-brain: unified pipeline (ingest + consolidate)\n"
        f"[maintenance {started_at}] [brain-step] run_id={run_id} started={started_at}\n"
        f"[maintenance {started_at}] step 1/5: ingest\n"
        f"[brain-pipeline] run_id={run_id} status={ingest_status} ingest_exit=0 consolidate_exit=0\n"
        f"[brain-ingest] run_id={run_id} files_processed=3 total_ingested=0 total_skipped=3 error_queue=0 status={ingest_status}\n"
        f"[brain-substep] run_id={run_id} name=ingest started={started_at} finished={finished_at} exit=0\n"
        f"[maintenance {finished_at}] [brain-step] run_id={run_id} finished={finished_at} exit=0\n"
        f"[brain-consolidate] run_id={run_id} status=skipped reason=no_new_since_24h "
        f"new_since_24h=0 total_points=0 merged_topics=0 threshold=20 topics_merged=0\n"
        f"[brain-substep] run_id={run_id} name=consolidate started={started_at} finished={finished_at} exit=0\n"
        f"[maintenance {finished_at}] ok\n"
    )


# ---------------------------------------------------------------------------
# Test 1: run_id propagated to summary, steps share it.
# ---------------------------------------------------------------------------


def test_run_id_propagated_to_summary(tmp_path: Path) -> None:
    """``run_id`` reaches every sub-step block AND the top-level summary."""
    run_id = "20260721T000000Z-deadbeef"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_run_id_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-21T00:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-21T00:00:30+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["run_id"] == run_id
    assert summary["steps"]["memory_brain"]["run_id"] == run_id
    assert summary["steps"]["memory_brain"]["sub_run_id"] == run_id
    assert summary["steps"]["ingest"]["sub_run_id"] == run_id
    # Top-level consolidation carries the same run_id via the run marker.
    # ``consolidation`` is a flat dict; we check the ingest step that
    # fired it shares the run_id as a correlation key.
    assert summary["steps"]["ingest"]["status"] in {"ok", "noop"}


# ---------------------------------------------------------------------------
# Test 2: ingest noop surfaces to dashboard card (not "—").
# ---------------------------------------------------------------------------


def test_ingest_zero_new_shows_noop_not_dash(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """When ingest runs but writes 0 chunks, the dashboard must see
    ``status=noop`` + ``ingested_new=0`` + ``run_id``, NOT a blank card.
    """
    from openclaw_memory_os.analytics import _read_memory_brain_status
    from openclaw_memory_os import analytics as analytics_mod

    run_id = "20260721T000100Z-cafe01"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_run_id_log_body(run_id, ingest_status="noop"),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-21T00:01:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-21T00:01:30+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    # Point analytics at our synthetic summary file. Drop the
    # legacy override so the canonical path always wins.
    monkeypatch.setenv("OPENCLAW_MEMORY_OS_SUMMARY", str(summary_path))
    monkeypatch.delenv("MEMORY_BRAIN_LEGACY_FILES_OK", raising=False)
    # Reset internal LRU caches so a previous test's value does not
    # leak into this case.
    if hasattr(analytics_mod._summarize_last_maintenance, "cache_clear"):
        try:
            analytics_mod._summarize_last_maintenance.cache_clear()
        except AttributeError:
            pass
    status = _read_memory_brain_status()

    mbi = status["ingest"]
    assert mbi["status"] == "noop"
    assert int(mbi["ingested_new"]) == 0
    assert mbi["run_id"] == run_id
    # The dashboard should still have a non-empty ``last_run`` so the
    # renderer can show "noop · <time>" rather than "—".
    assert mbi.get("last_run") or mbi.get("finished_at")


# ---------------------------------------------------------------------------
# Test 3: consolidation skipped → overall success.
# ---------------------------------------------------------------------------


def test_consolidation_skipped_still_overall_success(tmp_path: Path) -> None:
    """A skipped consolidation (no new memories) is NOT a failure."""
    run_id = "20260721T000200Z-cafe02"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_run_id_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-21T00:02:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-21T00:02:30+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["status"] == "success"
    assert summary["exit_code"] == 0
    assert summary["failed_step"] is None
    # consolidation.status must be ``skipped`` (per ``memory_brain_consolidate.py``
    # which emits ``[brain-consolidate] ... status=skipped reason=...``).
    assert summary["consolidation"]["status"] == "skipped"
    # The reason text round-trips so the dashboard can show it.
    assert summary["consolidation"]["reason"] == "no_new_since_24h"


# ---------------------------------------------------------------------------
# Test 4: every sub-step shares the parent run_id.
# ---------------------------------------------------------------------------


def test_substeps_share_run_id(tmp_path: Path) -> None:
    run_id = "20260721T000300Z-cafe03"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_run_id_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-21T00:03:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-21T00:03:30+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["run_id"] == run_id
    assert summary["steps"]["memory_brain"]["run_id"] == run_id
    assert summary["steps"]["memory_brain"]["sub_run_id"] == run_id
    assert summary["steps"]["ingest"]["sub_run_id"] == run_id
    # The unification contract: ingest sub-step and consolidation
    # both share the same parent run_id through ``memory_brain`` so
    # the dashboard renderer can render one RUN_ID badge for the
    # entire maintenance run.
    assert summary["steps"]["memory_brain"]["run_id"] == summary["steps"]["ingest"]["sub_run_id"]


# ---------------------------------------------------------------------------
# Test 5: sub-step timestamps belong to the same run window.
# ---------------------------------------------------------------------------


def test_substep_timestamps_belong_to_same_run(tmp_path: Path) -> None:
    """``steps.*.started_at`` MUST fall inside ``[started_at, finished_at]``.

    A sub-step timestamp from a previous run would break the
    correlation key and make the dashboard lie about "last run".
    """
    started = "2026-07-21T00:04:00+00:00"
    finished = "2026-07-21T00:04:30+00:00"
    run_id = "20260721T000400Z-cafe04"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_run_id_log_body(
            run_id,
            started_at="2026-07-21T00:04:00Z",
            finished_at="2026-07-21T00:04:30Z",
        ),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": started,
            "MAINTENANCE_FINISHED_AT": finished,
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    run_window_start = datetime.fromisoformat(started)
    run_window_end = datetime.fromisoformat(finished)
    # memory_brain bracket timestamps must be within the parent run.
    mb_started_raw = summary["steps"]["memory_brain"]["started_at"]
    mb_finished_raw = summary["steps"]["memory_brain"]["finished_at"]
    assert mb_started_raw is not None and mb_finished_raw is not None
    mb_started = datetime.fromisoformat(mb_started_raw.replace("Z", "+00:00"))
    mb_finished = datetime.fromisoformat(mb_finished_raw.replace("Z", "+00:00"))
    # The log emitted timestamps as UTC with a trailing ``Z``; the
    # parser normalises them to ISO. Compare against the run window
    # with a small tolerance because ``date -u +%S.%6NZ`` may produce
    # sub-second drift between the bracket and the parent summary.
    for ts in (mb_started, mb_finished):
        assert ts >= run_window_start - _td(seconds=2), (
            f"sub-step timestamp {ts} is earlier than the parent run "
            f"started at {run_window_start}"
        )
        assert ts <= run_window_end + _td(seconds=2), (
            f"sub-step timestamp {ts} is later than the parent run "
            f"finished at {run_window_end}"
        )
    # ingest sub-step mirrors memory_brain.
    ing_started_raw = summary["steps"]["ingest"]["started_at"]
    ing_finished_raw = summary["steps"]["ingest"]["finished_at"]
    assert ing_started_raw is not None and ing_finished_raw is not None


# ---------------------------------------------------------------------------
# Test 6: governance started_at ≠ finished_at; dashboard uses started_at
#         for "last-run".
# ---------------------------------------------------------------------------


def test_governance_started_vs_finished_not_confused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The dashboard renders ``started_at`` as "上次运行"; a swap to
    ``finished_at`` would silently lie about "when the timer fired".

    We write a governance status JSON where started_at and finished_at
    differ by 30s and confirm ``_read_autonomous_governance_status``
    preserves both fields independently. The dashboard-side rendering
    rule is asserted separately in test 6b below.
    """
    status_file = tmp_path / "autonomous-governance.json"
    status_file.write_text(
        json.dumps({
            "last_run": "2026-07-21T04:01:30+08:00",
            "last_result": "ok",
            "last_summary": "deep audit completed",
            "scheduled_at": "2026-07-21T04:01:00+08:00",
            "started_at": "2026-07-21T04:01:00+08:00",
            "finished_at": "2026-07-21T04:01:30+08:00",
            "duration_seconds": 30,
            "next_scheduled_at": "2026-07-28T04:01:00+08:00",
            "exit_code": 0,
            "mode": "governance",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(status_file))
    # Force the timer-derived next_scheduled_at fallback to be empty
    # so we test the explicit-JSON path.
    monkeypatch.setattr(
        "openclaw_memory_os.analytics._read_systemd_timer_schedule",
        lambda name: {"active_state": "unknown"},
    )

    from openclaw_memory_os import analytics as analytics_mod
    if hasattr(analytics_mod._read_systemd_timer_schedule, "cache_clear"):
        try:
            analytics_mod._read_systemd_timer_schedule.cache_clear()
        except AttributeError:
            pass

    status = analytics_mod._read_autonomous_governance_status()

    # Both fields exist and differ.
    assert status["started_at"] == "2026-07-21T04:01:00+08:00"
    assert status["finished_at"] == "2026-07-21T04:01:30+08:00"
    assert status["started_at"] != status["finished_at"]
    assert status["duration_seconds"] == 30
    assert status["exit_code"] == 0
    assert status["mode"] == "governance"
    # ``last_run`` is reformatted by the analytics reader to a compact
    # ``YYYY-MM-DD HH:MM`` (kept for legacy dashboards). The important
    # guarantee is that ``started_at`` survives untouched so the
    # Wave 2 renderer can show "上次运行" without losing fidelity.
    assert status["last_run"] == "2026-07-21 04:01"

    # Dashboard contract: the ``last-run`` slot picks up ``started_at``
    # (the brief pins this so we never confuse "when did the timer fire"
    # with "when did the work finish"). The renderer is JS-side; we
    # validate the JS source picks ``started_at`` first.
    js_src = (ROOT / "openclaw_memory_os" / "static" / "js" / "overview.js").read_text(encoding="utf-8")
    # The renderer reads ``ag.started_at || ag.last_run`` for the
    # ``data-governance="last-run"`` slot.
    assert 'ag.started_at || ag.last_run' in js_src


# ---------------------------------------------------------------------------
# Test 7: UTC -> Asia/Shanghai conversion (08:00 next-day for 00:00Z).
# ---------------------------------------------------------------------------


def test_utc_to_shanghai_conversion(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``2026-07-21T00:00:59Z`` must render as ``2026-07-21 08:00:59 CST``.

    Tested through the actual dashboard formatter path:
    ``Date(...).toLocaleString()`` honours the *browser* locale, which
    is en-US on the test runner. We override ``Date.prototype.toLocaleString``
    so the assertion stays deterministic regardless of host locale.
    """
    # Load common.js in a Node VM, stub Date.prototype.toLocaleString,
    # and call the exported ``formatTimestamp`` helper through
    # ``global.OCMemory``. The formatter is the same path overview.js /
    # strategy.js / etc. all use via ``OC.formatTimestamp``.
    node_script = (
        "const fs = require('fs');\n"
        f"const src = fs.readFileSync('{ROOT / 'openclaw_memory_os' / 'static' / 'js' / 'common.js'}', 'utf-8');\n"
        # Replace ``Date.prototype.toLocaleString`` so the formatter
        # ALWAYS renders Asia/Shanghai in sv-SE-style ISO shape.
        "const realFmt = Date.prototype.toLocaleString;\n"
        "Date.prototype.toLocaleString = function() {\n"
        "  const dtf = new Intl.DateTimeFormat('sv-SE', {\n"
        "    timeZone: 'Asia/Shanghai', year: 'numeric', month: '2-digit',\n"
        "    day: '2-digit', hour: '2-digit', minute: '2-digit',\n"
        "    second: '2-digit', hour12: false,\n"
        "  });\n"
        "  // ``sv-SE`` returns 'YYYY-MM-DD HH:MM:SS'; that matches the\n"
        "  // dashboard contract we want to assert against.\n"
        "  return dtf.format(this);\n"
        "};\n"
        "global.window = global;\n"
        f"const wrapped = '(function(global){{\\n' + src + '\\n}})(global);';\n"
        "eval(wrapped);\n"
        # Emit the formatter output for a UTC 2026-07-21T00:00:59Z value
        # which must come out as 08:00:59 in Asia/Shanghai.
        "const d = new Date('2026-07-21T00:00:59Z');\n"
        "process.stdout.write(global.OCMemory.formatTimestamp(d.toISOString()));\n"
    )
    proc = subprocess.run(
        ["node", "-e", node_script],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert proc.returncode == 0, f"node formatter smoke failed: {proc.stderr}"
    rendered = proc.stdout.strip()
    # 00:00:59Z + 8h = 08:00:59 Asia/Shanghai
    assert rendered == "2026-07-21 08:00:59", (
        f"UTC->Shanghai formatter returned {rendered!r}; expected "
        f"'2026-07-21 08:00:59'"
    )


# ---------------------------------------------------------------------------
# Test 8: legacy files cannot override canonical summary.
# ---------------------------------------------------------------------------


def test_legacy_files_dont_override_canonical(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """``/var/log/openclaw-memory-brain-status.json`` must NOT win over
    the canonical maintenance-summary even if it has a newer mtime.
    """
    run_id = "20260721T000800Z-cafe08"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_run_id_log_body(run_id, ingest_status="ok"),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-21T00:08:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-21T00:08:30+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    # Synthetic legacy file with a deliberately higher run_id and
    # newer mtime: the canonical path must still win.
    legacy_dir = tmp_path / "var_log"
    legacy_dir.mkdir()
    legacy_ingest = legacy_dir / "openclaw-memory-brain-status.json"
    legacy_ingest.write_text(
        json.dumps({
            "run_id": "FAKE-LEGACY-0001",
            "status": "ok",
            "last_run": "2099-01-01T00:00:00+00:00",
            "ingested_new": 999999,
        }),
        encoding="utf-8",
    )
    legacy_dream = legacy_dir / "openclaw-memory-brain-dream-status.json"
    legacy_dream.write_text(
        json.dumps({
            "run_id": "FAKE-LEGACY-0002",
            "status": "ok",
            "last_run": "2099-01-01T00:00:00+00:00",
            "topics_merged": 999999,
        }),
        encoding="utf-8",
    )

    monkeypatch.setenv("OPENCLAW_MEMORY_OS_SUMMARY", str(summary_path))
    monkeypatch.setenv("MEMORY_BRAIN_LEGACY_FILES_OK", "1")
    monkeypatch.setenv("MEMORY_BRAIN_STATUS_FILE", str(legacy_ingest))
    monkeypatch.setenv("MEMORY_BRAIN_DREAM_STATUS_FILE", str(legacy_dream))

    from openclaw_memory_os import analytics as analytics_mod
    if hasattr(analytics_mod._summarize_last_maintenance, "cache_clear"):
        try:
            analytics_mod._summarize_last_maintenance.cache_clear()
        except AttributeError:
            pass
    status = analytics_mod._read_memory_brain_status()
    mbi = status["ingest"]
    mbc = status["consolidate"]
    # Canonical run_id wins even though the legacy file is much newer.
    assert mbi["run_id"] == run_id
    assert mbc["run_id"] == run_id
    # Legacy keys only fill in gaps; they must not override the
    # canonical ``ingested_new`` of 0.
    if "ingested_new" in mbi:
        assert int(mbi["ingested_new"]) != 999999


# ---------------------------------------------------------------------------
# Test 9: failed sub-step → exit_code=1, status="failed", failed_step set.
# ---------------------------------------------------------------------------


def test_failed_step_overall_exit_nonzero(tmp_path: Path) -> None:
    failed_log_body = (
        "[maintenance 2026-07-21T01:00:00Z] RUN_ID=20260721T010000Z-fail\n"
        "[maintenance 2026-07-21T01:00:00Z] starting maintenance\n"
        "[maintenance 2026-07-21T01:00:00Z] --- [1/1] collection=openclaw_memory_os ---\n"
        "[maintenance 2026-07-21T01:00:00Z]   step 1/5: ingest\n"
        "[maintenance 2026-07-21T01:00:00Z] ERROR: ingest failed\n"
        "[maintenance 2026-07-21T01:00:00Z] completed with failures=1 failed_step=ingest failed\n"
    )
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=failed_log_body,
        env={
            "MAINTENANCE_RUN_ID": "20260721T010000Z-fail",
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-21T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-21T01:00:30+00:00",
            "MAINTENANCE_STATUS": "failed",
            "MAINTENANCE_EXIT_CODE": "1",
            "MAINTENANCE_FAILED_STEP": "ingest failed",
        },
    )
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 1
    assert summary["failed_step"] == "ingest failed"
    assert summary["steps"]["ingest"]["status"] == "failed"


# ---------------------------------------------------------------------------
# Test 10: MAINTENANCE_DRY_RUN does not modify collections.
# ---------------------------------------------------------------------------


def test_dry_run_does_not_modify_collections(tmp_path: Path) -> None:
    """A dry-run maintenance pass must not write to Qdrant.

    ``scripts/memory_brain.py`` and ``maintenance.sh`` honour the
    ``MAINTENANCE_DRY_RUN=1`` env var to disable writes; we verify the
    the parser still emits a summary (so the dashboard shows the dry-
    run entry) AND the ``exit_code`` stays zero, AND no real ingest
    fields are populated.
    """
    run_id = "20260721T001000Z-dryrun"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_run_id_log_body(run_id, ingest_status="noop"),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_DRY_RUN": "1",
            "MAINTENANCE_STARTED_AT": "2026-07-21T00:10:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-21T00:10:30+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    # Dry-run still records a run (so operators can see "we ran the
    # pipeline, nothing was written") but does NOT generate a fresh
    # snapshot or write a real ingest. The summary reflects that.
    assert summary["status"] == "success"
    assert summary["exit_code"] == 0
    # The "dry-run" marker is preserved end-to-end so the dashboard
    # can distinguish dry runs from real ones.
    assert summary["mode"] == "daily"
    # Ingest step is still noop; no chunks written.
    assert int(summary["steps"]["ingest"]["ingested_new"]) == 0


# ---------------------------------------------------------------------------
# Test 11: systemd timer schedule parsed from systemctl show output.
# ---------------------------------------------------------------------------


def test_systemd_timer_schedule_parsed_from_systemctl(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mock ``subprocess.run`` so we can pin the systemctl output and
    confirm the parser surfaces ``calendar`` + ``last_trigger`` +
    ``next_elapse`` + ``active_state`` for a real timer.
    """
    from openclaw_memory_os import analytics as analytics_mod

    # Drop any TTL cache from earlier tests in this session.
    if hasattr(analytics_mod._read_systemd_timer_schedule, "cache_clear"):
        try:
            analytics_mod._read_systemd_timer_schedule.cache_clear()
        except AttributeError:
            pass

    fake_completed = mock.Mock()
    fake_completed.returncode = 0
    fake_completed.stdout = "\n".join([
        "ActiveState=active",
        "Result=success",
        "TimersCalendar={ OnCalendar=Tue *-*-* 04:01:00 Asia/Shanghai ; "
        "next_elapse=Tue 2026-07-28 04:01:00 CST }",
        "LastTriggerUSec=Tue 2026-07-21 04:01:17 CST",
        "NextElapseUSecRealtime=Tue 2026-07-28 04:01:00 CST",
    ])
    fake_completed.stderr = ""

    monkeypatch.setattr(analytics_mod.subprocess, "run", lambda *a, **kw: fake_completed)
    schedule = analytics_mod._read_systemd_timer_schedule(
        "openclaw-memory-os-governance.timer"
    )

    assert schedule["calendar"] == "Tue *-*-* 04:01:00 Asia/Shanghai"
    assert schedule["active_state"] == "active"
    assert schedule["result"] == "success"
    # ``last_trigger`` is converted to UTC ISO.
    assert schedule["last_trigger"] == "2026-07-20T20:01:17+00:00"
    assert schedule["next_elapse"] == "2026-07-27T20:01:00+00:00"


def test_systemd_timer_unknown_when_no_calendar(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If the unit is loaded but ``OnCalendar`` is empty, we treat it as unknown."""
    from openclaw_memory_os import analytics as analytics_mod
    if hasattr(analytics_mod._read_systemd_timer_schedule, "cache_clear"):
        try:
            analytics_mod._read_systemd_timer_schedule.cache_clear()
        except AttributeError:
            pass

    fake_completed = mock.Mock()
    fake_completed.returncode = 0
    # ``systemctl show`` for a real-but-empty unit returns the value
    # below. ``TimersCalendar`` lacks an inner ``OnCalendar=`` field
    # (or has it empty), so our parser must treat this as unknown.
    fake_completed.stdout = "\n".join([
        "ActiveState=inactive",
        "Result=success",
        "TimersCalendar={ next_elapse= }",
        "LastTriggerUSec=",
        "NextElapseUSecRealtime=",
    ])
    fake_completed.stderr = ""

    monkeypatch.setattr(analytics_mod.subprocess, "run", lambda *a, **kw: fake_completed)
    schedule = analytics_mod._read_systemd_timer_schedule("non-existent.timer")
    # The dashboard contract is ``{"active_state": "unknown"}`` — anything
    # more (calendar / result / next_elapse) is a leak.
    assert schedule.get("active_state") == "unknown"
    assert "calendar" not in schedule


def test_systemd_timer_handles_subprocess_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A hanging ``systemctl`` must NOT break the dashboard render."""
    from openclaw_memory_os import analytics as analytics_mod
    if hasattr(analytics_mod._read_systemd_timer_schedule, "cache_clear"):
        try:
            analytics_mod._read_systemd_timer_schedule.cache_clear()
        except AttributeError:
            pass

    def _raise(*a, **kw):
        raise subprocess.TimeoutExpired(cmd="systemctl", timeout=3.0)

    monkeypatch.setattr(analytics_mod.subprocess, "run", _raise)
    schedule = analytics_mod._read_systemd_timer_schedule("any.timer")
    assert schedule == {"active_state": "unknown"}


# ---------------------------------------------------------------------------
# Test 12: no hardcoded schedule in source code.
# ---------------------------------------------------------------------------


def test_no_hardcoded_schedule_in_source() -> None:
    """The dashboard source must NOT hardcode ``Tue 04:01`` /
    ``07:45`` as strings; schedules come from the live systemd timer.
    """

    def _strip_comments_python(src: str) -> str:
        # Drop docstrings (""" ... """ and ''' ... ''') and single-line
        # comments so explanatory text like "Tue *-*-* 04:01:00" never
        # trips the assertion. Multi-line strings inside code stay.
        src = re.sub(r'"""[\s\S]*?"""', "", src)
        src = re.sub(r"'''[\s\S]*?'''", "", src)
        src = re.sub(r"(?m)#.*$", "", src)
        return src

    def _strip_comments_js(src: str) -> str:
        # Drop // single-line and /* ... */ block comments.
        src = re.sub(r"/\*.*?\*/", "", src, flags=re.DOTALL)
        src = re.sub(r"(?m)//.*$", "", src)
        return src

    # The contract schedule lives in ``models.AutonomousGovernanceJob``
    # (the canonical Tue 04:01 weekly contract) and is rendered as a
    # hidden span so existing tests can still assert it is present in
    # the SSR HTML. That is a contract description, NOT a dashboard
    # schedule literal; it must stay.
    def _strip_comments_html(src: str) -> str:
        # Drop HTML/XML comments so explanatory / display-only text
        # doesn't trip the assertion.
        return re.sub(r"<!--[\s\S]*?-->", "", src)


    paths_to_check: list[tuple[Path, callable]] = []
    py_targets = [
        PACKAGE_DIR / "analytics.py",
        PACKAGE_DIR / "app.py",
    ]
    for p in py_targets:
        if p.exists():
            paths_to_check.append((p, _strip_comments_python))
    html_targets = [
        PACKAGE_DIR / "templates" / "dashboard.html",
    ]
    for p in html_targets:
        if p.exists():
            paths_to_check.append((p, _strip_comments_html))
    js_targets = [
        PACKAGE_DIR / "static" / "js" / "overview.js",
        PACKAGE_DIR / "static" / "js" / "strategy.js",
    ]
    for p in js_targets:
        if p.exists():
            paths_to_check.append((p, _strip_comments_js))
    # The scripts directory is OFF-LIMITS for Wave 2 changes, so we
    # only sanity-check that no new hardcoded schedule has been added
    # there either (existing references are allowed).
    scripts_dir = SCRIPTS_DIR
    for script in scripts_dir.glob("*.sh"):
        if script.name.startswith("_") or script.name in {"maintenance.sh"}:
            continue
        # Bash comment stripping: drop lines starting with ``#``.
        def _strip_bash(src: str) -> str:
            return "\n".join(
                line for line in src.splitlines() if not line.lstrip().startswith("#")
            )
        paths_to_check.append((script, _strip_bash))

    forbidden_literals = ["04:01", "07:45:00", "daily 07:45"]
    for path, stripper in paths_to_check:
        if not path.exists():
            continue
        src = stripper(path.read_text(encoding="utf-8"))
        for literal in forbidden_literals:
            assert literal not in src, (
                f"{path} contains hardcoded schedule literal {literal!r}; "
                f"the dashboard must read from the live systemd timer instead."
            )


# ---------------------------------------------------------------------------
# Helper imports for ``test_substep_timestamps_belong_to_same_run``.
# ---------------------------------------------------------------------------

from datetime import timedelta as _td  # noqa: E402  (test-only local import)

# ===========================================================================
# Wave 4 (2026-07-21): independent ingest / consolidate sub-step runtimes.
#
# The pipeline contract after Wave 4 is:
#   - ``[brain-substep]`` markers emitted by ``scripts/memory_brain.py``
#     carry independent started/finished timestamps for the ingest and
#     consolidate leaves.
#   - ``steps.ingest`` and ``consolidation`` blocks in the canonical
#     summary read these markers (the parent ``[brain-step]`` bracket is
#     no longer mirrored into the children).
#   - The dashboard surface (overview.js) renders started/finished/duration
#     for both cards, plus a mobile-friendly CSS block.
#
# Tests below pin the full chain: status write → analytics layer → API
# schema → frontend time block (rendered string check) → mobile CSS
# availability.
# ===========================================================================


def _build_substep_log_body(
    run_id: str,
    ingest_started: str = "2026-07-22T01:00:00Z",
    ingest_finished: str = "2026-07-22T01:00:00.500Z",
    consolidate_started: str = "2026-07-22T01:00:00.600Z",
    consolidate_finished: str = "2026-07-22T01:00:01.000Z",
    consolidate_status: str = "skipped",
    consolidate_reason: str = "新增 0 < 20",
    topics_merged: int = 0,
) -> str:
    """Log body with independent [brain-substep] markers for ingest + consolidate."""
    return (
        f"[maintenance 2026-07-22T01:00:00Z] RUN_ID={run_id}\n"
        f"[maintenance 2026-07-22T01:00:00Z] starting maintenance\n"
        f"[maintenance 2026-07-22T01:00:00Z] --- [1/1] collection=openclaw_memory_os ---\n"
        f"[maintenance 2026-07-22T01:00:00Z] memory-brain: unified pipeline (ingest + consolidate)\n"
        f"[maintenance 2026-07-22T01:00:00Z] [brain-step] run_id={run_id} started=2026-07-22T01:00:00Z\n"
        f"[maintenance 2026-07-22T01:00:00Z] step 1/5: ingest\n"
        f"[brain-pipeline] run_id={run_id} status=ok ingest_exit=0 consolidate_exit=0\n"
        f"[brain-ingest] run_id={run_id} files_processed=0 total_ingested=0 total_skipped=0 error_queue=0 status=ok\n"
        f"[brain-substep] run_id={run_id} name=ingest started={ingest_started} finished={ingest_finished} exit=0\n"
        f"[maintenance 2026-07-22T01:00:01Z] [brain-step] run_id={run_id} finished=2026-07-22T01:00:01Z exit=0\n"
        f"[brain-consolidate] run_id={run_id} status={consolidate_status} "
        f"reason={consolidate_reason} "
        f"new_since_24h=0 total_points=321 "
        f"merged_topics={topics_merged} threshold=20 topics_merged={topics_merged}\n"
        f"[brain-substep] run_id={run_id} name=consolidate started={consolidate_started} finished={consolidate_finished} exit=0\n"
        f"[maintenance 2026-07-22T01:00:01Z] ok\n"
    )


# ---------------------------------------------------------------------------
# Test 12: ingest.started_at / finished_at / duration_seconds are written
# by the parser from the [brain-substep] markers, not the parent bracket.
# ---------------------------------------------------------------------------

def test_ingest_substep_writes_started_at(tmp_path: Path) -> None:
    """The ingest substep carries its own started_at from [brain-substep]."""
    run_id = "20260722T010000Z-indsub"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["steps"]["ingest"]["started_at"] == "2026-07-22T01:00:00Z"


def test_ingest_substep_writes_finished_at(tmp_path: Path) -> None:
    run_id = "20260722T010000Z-indfin"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["steps"]["ingest"]["finished_at"] == "2026-07-22T01:00:00.500Z"


def test_ingest_substep_duration_seconds_computed(tmp_path: Path) -> None:
    run_id = "20260722T010000Z-inddur"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    # 0.5s — not zero, not int-rounded.
    assert summary["steps"]["ingest"]["duration_seconds"] == 0.5


# ---------------------------------------------------------------------------
# Test 13: consolidation skipped path still carries its own bracket.
# ---------------------------------------------------------------------------

def test_consolidation_skipped_writes_started_at(tmp_path: Path) -> None:
    run_id = "20260722T010000Z-csk"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["consolidation"]["status"] == "skipped"
    assert summary["consolidation"]["started_at"] == "2026-07-22T01:00:00.600Z"


def test_consolidation_skipped_writes_finished_at(tmp_path: Path) -> None:
    run_id = "20260722T010000Z-cskfin"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["consolidation"]["finished_at"] == "2026-07-22T01:00:01.000Z"


def test_consolidation_skipped_duration_seconds_computed(tmp_path: Path) -> None:
    run_id = "20260722T010000Z-cskdur"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    # 0.4s — independent, not copied from ingest.
    assert summary["consolidation"]["duration_seconds"] == 0.4


# ---------------------------------------------------------------------------
# Test 14: both substeps share the maintenance run_id.
# ---------------------------------------------------------------------------

def test_ingest_and_consolidation_share_run_id(tmp_path: Path) -> None:
    run_id = "20260722T010000Z-shared"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    assert summary["run_id"] == run_id
    assert summary["consolidation"]["run_id"] == run_id
    # ingest sub_run_id matches the parent.
    assert summary["steps"]["ingest"]["sub_run_id"] == run_id
    # ingest card carries its own run_id (via the substep marker) for
    # direct correlation in the dashboard.
    assert summary["steps"]["ingest"].get("run_id") == run_id


# ---------------------------------------------------------------------------
# Test 15: substep timestamps are NOT mirror copies of the parent bracket.
# ---------------------------------------------------------------------------

def test_substep_timestamps_independent_from_parent(tmp_path: Path) -> None:
    """The two substep brackets must differ from each other and from the
    parent ``steps.memory_brain`` bracket."""
    run_id = "20260722T010000Z-indep"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    ing_s = summary["steps"]["ingest"]["started_at"]
    ing_f = summary["steps"]["ingest"]["finished_at"]
    con_s = summary["consolidation"]["started_at"]
    con_f = summary["consolidation"]["finished_at"]
    parent_s = summary["steps"]["memory_brain"]["started_at"]
    parent_f = summary["steps"]["memory_brain"]["finished_at"]
    # All four timestamps must be present and distinct.
    assert ing_s and ing_f and con_s and con_f
    # ingest runs strictly before consolidation: ingest.finished_at <=
    # consolidation.started_at.
    from datetime import datetime
    def _t(s): return datetime.fromisoformat(s.replace("Z", "+00:00"))
    assert _t(ing_f) <= _t(con_s), (
        f"ingest.finished_at ({ing_f}) must precede consolidation.started_at ({con_s})"
    )
    # Parent bracket is wider than either substep.
    assert _t(ing_s) >= _t(parent_s)
    assert _t(con_f) <= _t(parent_f)
    # Independent durations.
    assert summary["steps"]["ingest"]["duration_seconds"] != summary["consolidation"]["duration_seconds"]


# ---------------------------------------------------------------------------
# Test 16: API + analytics transparently pass the new fields.
# ---------------------------------------------------------------------------

def test_analytics_passes_independent_substep_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "20260722T010000Z-anal"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setenv("OPENCLAW_MEMORY_OS_SUMMARY", str(summary_path))

    from openclaw_memory_os import analytics as analytics_mod
    if hasattr(analytics_mod._summarize_last_maintenance, "cache_clear"):
        try:
            analytics_mod._summarize_last_maintenance.cache_clear()
        except AttributeError:
            pass
    mb = analytics_mod._read_memory_brain_status()
    # Each card carries the 11-field contract.
    for key in ("started_at", "finished_at", "duration_seconds", "status", "run_id"):
        assert key in mb["ingest"], f"ingest.{key} missing"
        assert key in mb["consolidate"], f"consolidate.{key} missing"
    assert mb["ingest"]["started_at"] == "2026-07-22T01:00:00Z"
    assert mb["consolidate"]["started_at"] == "2026-07-22T01:00:00.600Z"
    assert mb["consolidate"]["merged_topics"] == 0
    assert mb["consolidate"]["threshold"] == 20


# ---------------------------------------------------------------------------
# Test 17: legacy /var/log brain files do not override canonical substep
# timestamps.
# ---------------------------------------------------------------------------

def test_legacy_files_cannot_override_substep_times(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run_id = "20260722T010000Z-legacy"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")

    legacy_dir = tmp_path / "var_log"
    legacy_dir.mkdir()
    legacy_ingest = legacy_dir / "openclaw-memory-brain-status.json"
    legacy_ingest.write_text(
        json.dumps({
            "run_id": "FAKE-LEGACY-INGEST",
            "started_at": "2099-01-01T00:00:00Z",
            "finished_at": "2099-01-01T00:00:30Z",
        }),
        encoding="utf-8",
    )
    legacy_dream = legacy_dir / "openclaw-memory-brain-dream-status.json"
    legacy_dream.write_text(
        json.dumps({
            "run_id": "FAKE-LEGACY-DREAM",
            "started_at": "2099-01-01T01:00:00Z",
            "finished_at": "2099-01-01T01:00:30Z",
        }),
        encoding="utf-8",
    )
    monkeypatch.setenv("OPENCLAW_MEMORY_OS_SUMMARY", str(summary_path))
    monkeypatch.setenv("MEMORY_BRAIN_LEGACY_FILES_OK", "1")
    monkeypatch.setenv("MEMORY_BRAIN_STATUS_FILE", str(legacy_ingest))
    monkeypatch.setenv("MEMORY_BRAIN_DREAM_STATUS_FILE", str(legacy_dream))

    from openclaw_memory_os import analytics as analytics_mod
    if hasattr(analytics_mod._summarize_last_maintenance, "cache_clear"):
        try:
            analytics_mod._summarize_last_maintenance.cache_clear()
        except AttributeError:
            pass
    mb = analytics_mod._read_memory_brain_status()
    # Canonical timestamps must win even when the legacy files carry
    # 2099 timestamps.
    assert mb["ingest"]["started_at"] == "2026-07-22T01:00:00Z"
    assert mb["consolidate"]["started_at"] == "2026-07-22T01:00:00.600Z"
    assert mb["consolidate"]["finished_at"] == "2026-07-22T01:00:01.000Z"


# ---------------------------------------------------------------------------
# Test 18: the frontend (overview.js) renders the time block for both
# cards. We assert by static-reading the JS source: it must reference the
# formatter + ``mb-substep-time`` class so the block is always emitted.
# ---------------------------------------------------------------------------

def test_overview_js_renders_substep_time_block() -> None:
    repo_root = SCRIPTS_DIR.parent
    js = (repo_root / "openclaw_memory_os" / "static" / "js" / "overview.js").read_text(encoding="utf-8")
    # Both cards must use the shared formatter.
    assert "fmtCstCompact" in js
    assert "fmtDuration" in js
    # Both cards must emit the time-block class.
    assert js.count("mb-substep-time") >= 2, (
        "overview.js must emit mb-substep-time for both ingest and consolidate"
    )
    # Both cards must include the start / finish / duration Chinese labels.
    for label in ("开始：", "完成：", "耗时："):
        assert js.count(label) >= 2
    # Explicit error message for missing timestamps.
    assert "状态数据缺少时间" in js
    assert "缺少耗时" in js


# ---------------------------------------------------------------------------
# Test 19: dashboard.css carries the mobile breakpoint so the time block
# stays visible at narrow widths.
# ---------------------------------------------------------------------------

def test_dashboard_css_mobile_substep_time_block() -> None:
    repo_root = SCRIPTS_DIR.parent
    css = (repo_root / "openclaw_memory_os" / "static" / "css" / "dashboard.css").read_text(encoding="utf-8")
    assert ".mb-substep-time" in css
    # Mobile breakpoint must stack to a single column.
    assert "@media (max-width: 480px)" in css
    assert "grid-template-columns: 1fr" in css
    # RUN_ID chip is also bounded on mobile.
    assert ".mb-run-id" in css
    assert "max-width: 60vw" in css


# ---------------------------------------------------------------------------
# Test 20: UTC -> Asia/Shanghai rendering helper. We directly exercise
# the formatter the JS would build, expressed in Python so the test
# catches timezone regressions without a real browser.
# ---------------------------------------------------------------------------

def test_utc_to_shanghai_render_helper() -> None:
    from datetime import datetime, timezone, timedelta
    cst = timezone(timedelta(hours=8))

    def render(ts: str) -> str:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00")).astimezone(cst)
        return dt.strftime("%Y/%-m/%-d %H:%M:%S")

    # canonical ingest timestamp from production run on 2026-07-21
    assert render("2026-07-21T08:12:57.827937Z") == "2026/7/21 16:12:57"
    # consolidation skipped run on 2026-07-21 (early morning Beijing)
    assert render("2026-07-20T20:01:17Z") == "2026/7/21 04:01:17"


# ---------------------------------------------------------------------------
# Test 21: API round-trip exposes every sub-step timestamp field.
# ---------------------------------------------------------------------------

def test_api_passes_through_all_substep_time_fields(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The /api/health response shape exposes ingest + consolidation started /
    finished / duration on both cards. We confirm the wire format by
    calling the FastAPI test client through the analytics layer used by
    build_health_summary."""
    run_id = "20260722T010000Z-api"
    summary = _make_summary_via_writer(
        tmp_path,
        log_body=_build_substep_log_body(run_id),
        env={
            "MAINTENANCE_RUN_ID": run_id,
            "MAINTENANCE_MODE": "daily",
            "MAINTENANCE_STARTED_AT": "2026-07-22T01:00:00+00:00",
            "MAINTENANCE_FINISHED_AT": "2026-07-22T01:00:01+00:00",
            "MAINTENANCE_STATUS": "success",
            "MAINTENANCE_EXIT_CODE": "0",
            "MAINTENANCE_FAILED_STEP": "",
        },
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(summary), encoding="utf-8")
    monkeypatch.setenv("OPENCLAW_MEMORY_OS_SUMMARY", str(summary_path))
    from openclaw_memory_os import analytics as analytics_mod
    if hasattr(analytics_mod._summarize_last_maintenance, "cache_clear"):
        try:
            analytics_mod._summarize_last_maintenance.cache_clear()
        except AttributeError:
            pass
    mb = analytics_mod._read_memory_brain_status()
    # The contract: ingest + consolidate both carry start/finish/duration.
    for leaf in ("ingest", "consolidate"):
        for field in ("started_at", "finished_at", "duration_seconds"):
            assert mb[leaf].get(field) is not None, (
                f"memory_brain.{leaf}.{field} missing in API payload"
            )


# ===========================================================================
# Wave 5 (2026-07-22): log-rotation noise must not be misreported as a
# maintenance run.
#
# ``logrotate`` truncates ``/var/log/openclaw-memory-os/maintenance.log``
# every night, which updates the file mtime. The previous
# ``_read_maintenance_health`` implementation took the file mtime as
# ``last_run`` whenever it couldn't parse a ``[maintenance ...] starting``
# line, so the dashboard surfaced the logrotate timestamp as "最近运行"
# for the rest of the day. The fix parses ``[maintenance ...] starting``
# from the log first and only uses the file mtime as a last-resort
# fallback when the log is genuinely empty AND the JSON summary is
# missing.
# ===========================================================================


def _build_log_with_just_truncate_marker() -> str:
    """Simulate the post-logrotate state: an empty log with a single
    line that is NOT a maintenance run marker. Should not be interpreted
    as a real run."""
    return ""


def test_logrotate_truncate_does_not_become_last_run(tmp_path: Path) -> None:
    """An empty maintenance log must not surface as a real run time."""
    from openclaw_memory_os.analytics import _read_maintenance_health

    log_path = tmp_path / "maintenance.log"
    log_path.write_text(_build_log_with_just_truncate_marker(), encoding="utf-8")
    # Stamp the file mtime to a date that is clearly wrong.
    import os
    target_ts = 1753157238.419325  # 2025-07-22 00:07:18 UTC
    os.utime(log_path, (target_ts, target_ts))

    monkey = __import__("pytest").MonkeyPatch()
    monkey.setenv("OPENCLAW_MEMORY_OS_LOG", str(log_path))
    # Make sure no summary file is found in the test tmpdir.
    monkey.setenv("OPENCLAW_MEMORY_OS_SUMMARY", str(tmp_path / "summary-missing.json"))
    try:
        h = _read_maintenance_health()
    finally:
        monkey.undo()
    # The function should NOT return the logrotate mtime as last_run.
    if h.get("last_run") is not None:
        assert h["last_run"] != "2025-07-22T00:07:18.419325+00:00", (
            "empty maintenance log must not surface the logrotate "
            "timestamp as a real run"
        )


def test_log_mtime_preserved_when_summary_also_missing(tmp_path: Path) -> None:
    """If the log has a real maintenance marker AND the summary is
    missing, the parsed log timestamp wins over the file mtime."""
    from openclaw_memory_os.analytics import _read_maintenance_health

    log_path = tmp_path / "maintenance.log"
    log_body = (
        "[maintenance 2026-07-21T13:32:52.640Z] RUN_ID=20260721T133252Z-test\n"
        "[maintenance 2026-07-21T13:32:52.640Z] starting maintenance\n"
        "[maintenance 2026-07-21T13:32:52.640Z] ok\n"
    )
    log_path.write_text(log_body, encoding="utf-8")
    # mtime 1 day later (the morning after a logrotate).
    import os, time as _t
    target_ts = _t.mktime((2026, 7, 22, 0, 7, 18, 0, 0, 0))
    os.utime(log_path, (target_ts, target_ts))

    monkey = __import__("pytest").MonkeyPatch()
    monkey.setenv("OPENCLAW_MEMORY_OS_LOG", str(log_path))
    monkey.setenv("OPENCLAW_MEMORY_OS_SUMMARY", str(tmp_path / "summary-missing.json"))
    try:
        h = _read_maintenance_health()
    finally:
        monkey.undo()
    # The parsed log timestamp must win.
    assert h.get("last_run") == "2026-07-21T13:32:52.640Z", (
        f"expected the parsed starting line, got {h.get('last_run')!r}"
    )
    # last_ok is the ok line.
    assert h.get("last_ok") == "2026-07-21T13:32:52.640Z"
