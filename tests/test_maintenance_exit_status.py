"""Tests for maintenance.sh exit semantics + summary status field mapping.

The dashboard MUST NOT report a failed maintenance run as "success".

These tests exercise maintenance.sh directly via subprocess with stubbed
python step invocations so we can deterministically trigger per-step
failures. The same env-var pattern that production uses
(MAINTENANCE_STATUS, MAINTENANCE_EXIT_CODE, MAINTENANCE_FAILED_STEP) is
injected into _write_summary.py for the summary assertions.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = REPO_ROOT / "scripts"
MAINTENANCE_SH = SCRIPTS_DIR / "maintenance.sh"
VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
PYTHON_BIN = VENV_PY if VENV_PY.exists() else Path(sys.executable)


def _build_patched_maintenance(tmp_path: Path):
    """Build a patched maintenance.sh + stub scripts directory."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    stub_dir = tmp_path / "stubs"
    stub_dir.mkdir()

    # Python wrapper that execs the real python
    (bin_dir / "python").write_text(
        "#!/usr/bin/env bash\nexec \"" + str(PYTHON_BIN) + "\" \"$@\"\n"
    )
    (bin_dir / "python").chmod(0o755)

    # Stub scripts — each exits with the corresponding *_RC env var (default 0)
    for name in (
        "ingest", "reclassify", "supersede", "expire",
        "snapshot", "lexical", "summary", "governance_status",
    ):
        uc = name.upper()
        (stub_dir / f"{name}.sh").write_text(
            "#!/usr/bin/env bash\n"
            f"echo \"stub: {name}\"\n"
            f"exit ${{{uc}_RC:-0}}\n"
        )
        (stub_dir / f"{name}.sh").chmod(0o755)

    log_path = tmp_path / "maintenance.log"
    summary_path = tmp_path / "summary.json"

    # Patch the original maintenance.sh
    src = MAINTENANCE_SH.read_text(encoding="utf-8")
    src = src.replace(
        'SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"',
        f'SCRIPT_DIR="{stub_dir}"'
    )
    src = src.replace(
        'VENV_PY="$PROJECT_DIR/.venv/bin/python"',
        f'VENV_PY="{bin_dir / "python"}"'
    )
    # Ingest step
    src = src.replace(
        '''    WORKSPACE_ROOT="${WORKSPACE_ROOT:-$PROJECT_DIR/..}" \\
      "$VENV_PY" -m openclaw_memory_os.cli ingest --collection "$COLLECTION" >> "$LOG_FILE" 2>&1 \\
      || mark_failure "ingest failed"''',
        f'''    "{stub_dir}/ingest.sh" >> "$LOG_FILE" 2>&1 \\
      || mark_failure "ingest failed"'''
    )
    src = src.replace(
        '"$VENV_PY" "$SCRIPT_DIR/tier_classifier.py" --collection "$COLLECTION"',
        f'"{stub_dir}/reclassify.sh"'
    )
    src = src.replace(
        '"$VENV_PY" "$SCRIPT_DIR/supersede_detect.py" --collection "$COLLECTION" --recency-gap-days 7',
        f'"{stub_dir}/supersede.sh"'
    )
    src = src.replace(
        '"$VENV_PY" "$SCRIPT_DIR/expire_cron.py" --collection "$COLLECTION"',
        f'"{stub_dir}/expire.sh"'
    )
    src = src.replace(
        '"$SCRIPT_DIR/backup_snapshot.sh" "$COLLECTION"',
        f'"{stub_dir}/snapshot.sh"'
    )
    src = src.replace(
        '"$VENV_PY" "$SCRIPT_DIR/refresh_lexical.py"',
        f'"{stub_dir}/lexical.sh"'
    )
    src = src.replace(
        '"$VENV_PY" "$SCRIPT_DIR/_write_summary.py"',
        f'"{stub_dir}/summary.sh"'
    )
    src = src.replace(
        '"$VENV_PY" "$SCRIPT_DIR/_write_governance_status.py"',
        f'"{stub_dir}/governance_status.sh"'
    )

    patched = tmp_path / "maintenance.sh"
    patched.write_text(src, encoding="utf-8")
    patched.chmod(0o755)
    return patched, log_path, summary_path, state_dir


def _run_maintenance(script, log, summary, state_dir, stub_rcs):
    env = os.environ.copy()
    env.update({
        "MAINTAIN_COLLECTIONS": "test_coll",
        "LOG_FILE": str(log),
        "SUMMARY_FILE": str(summary),
        "XDG_STATE_HOME": str(state_dir),
        "BACKUP_DIR": str(state_dir / "backups"),
        "WRITE_GOVERNANCE_STATUS": "0",
        "ENABLE_MEMORY_BRAIN": "0",
        "SKIP_INGEST_COLLECTIONS": "",
    })
    for name, rc in stub_rcs.items():
        env[f"{name.upper()}_RC"] = str(rc)
    return subprocess.run(
        ["bash", str(script)],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
    )


@pytest.fixture
def patched(tmp_path):
    script, log, summary, state = _build_patched_maintenance(tmp_path)
    return script, log, summary, state


def test_maintenance_failure_exits_nonzero_with_failed_step(patched):
    script, log, summary, state = patched
    proc = _run_maintenance(script, log, summary, state, stub_rcs={"ingest": 1})
    assert proc.returncode == 1, (
        f"Expected exit 1, got {proc.returncode}\n"
        f"stdout: {proc.stdout[-500:]}\nstderr: {proc.stderr[-500:]}"
    )
    log_text = log.read_text(encoding="utf-8")
    assert "failed_step=ingest failed" in log_text
    assert "completed with failures=1" in log_text


def test_maintenance_failure_marks_correct_failed_step(patched):
    script, log, summary, state = patched
    proc = _run_maintenance(
        script, log, summary, state,
        stub_rcs={"ingest": 0, "reclassify": 1},
    )
    assert proc.returncode == 1
    log_text = log.read_text(encoding="utf-8")
    assert "failed_step=reclassification failed" in log_text


def test_maintenance_success_exits_zero(patched):
    script, log, summary, state = patched
    proc = _run_maintenance(script, log, summary, state, stub_rcs={})
    assert proc.returncode == 0, (
        f"Expected exit 0, got {proc.returncode}\n"
        f"stderr: {proc.stderr[-500:]}"
    )
    log_text = log.read_text(encoding="utf-8")
    assert log_text.rstrip().endswith("ok")


def test_write_summary_records_failure_status(tmp_path, monkeypatch):
    log = tmp_path / "maintenance.log"
    log.write_text(
        "[maintenance 2026-07-20T10:00:00Z] starting maintenance\n"
        "[maintenance 2026-07-20T10:00:00Z]   step 1/5: ingest\n"
        "[maintenance 2026-07-20T10:00:00Z] completed with failures=1 failed_step=ingest failed\n"
    )
    out = tmp_path / "summary.json"
    monkeypatch.setenv("MAINTENANCE_STARTED_AT", "2026-07-20T02:00:00+00:00")
    monkeypatch.setenv("MAINTENANCE_FINISHED_AT", "2026-07-20T02:30:23+00:00")
    monkeypatch.setenv("MAINTENANCE_STATUS", "failed")
    monkeypatch.setenv("MAINTENANCE_EXIT_CODE", "1")
    monkeypatch.setenv("MAINTENANCE_FAILED_STEP", "ingest failed")

    proc = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPTS_DIR / "_write_summary.py"), str(log), str(out)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=30,
    )
    assert proc.returncode == 0, f"_write_summary failed: {proc.stderr}"
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["status"] == "failed"
    assert summary["exit_code"] == 1
    assert summary["failed_step"] == "ingest failed"
    assert summary["started_at"] == "2026-07-20T02:00:00+00:00"
    assert summary["finished_at"] == "2026-07-20T02:30:23+00:00"


def test_write_summary_records_success_status_and_last_success_at(tmp_path, monkeypatch):
    log = tmp_path / "maintenance.log"
    log.write_text(
        "[maintenance 2026-07-20T10:00:00Z] starting maintenance\n"
        "[maintenance 2026-07-20T10:00:00Z] ok\n"
    )
    out = tmp_path / "summary.json"
    monkeypatch.setenv("MAINTENANCE_STARTED_AT", "2026-07-20T02:00:00+00:00")
    monkeypatch.setenv("MAINTENANCE_FINISHED_AT", "2026-07-20T02:30:23+00:00")
    monkeypatch.setenv("MAINTENANCE_STATUS", "success")
    monkeypatch.setenv("MAINTENANCE_EXIT_CODE", "0")
    monkeypatch.setenv("MAINTENANCE_FAILED_STEP", "")

    proc = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPTS_DIR / "_write_summary.py"), str(log), str(out)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=30,
    )
    assert proc.returncode == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["status"] == "success"
    assert summary["exit_code"] == 0
    assert summary["failed_step"] is None
    assert summary["last_success_at"] == "2026-07-20T02:30:23+00:00"


def test_last_success_at_does_not_regress_on_failure(tmp_path, monkeypatch):
    out = tmp_path / "summary.json"
    out.write_text(json.dumps({
        "last_success_at": "2026-07-19T07:45:00+00:00",
        "status": "success",
        "exit_code": 0,
        "failed_step": None,
    }))
    log = tmp_path / "maintenance.log"
    log.write_text("[maintenance 2026-07-20T10:00:00Z] starting maintenance\n")
    monkeypatch.setenv("MAINTENANCE_STARTED_AT", "2026-07-20T02:00:00+00:00")
    monkeypatch.setenv("MAINTENANCE_FINISHED_AT", "2026-07-20T02:30:23+00:00")
    monkeypatch.setenv("MAINTENANCE_STATUS", "failed")
    monkeypatch.setenv("MAINTENANCE_EXIT_CODE", "1")
    monkeypatch.setenv("MAINTENANCE_FAILED_STEP", "ingest failed")

    proc = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPTS_DIR / "_write_summary.py"), str(log), str(out)],
        capture_output=True, text=True, env=os.environ.copy(), timeout=30,
    )
    assert proc.returncode == 0
    summary = json.loads(out.read_text(encoding="utf-8"))
    assert summary["last_success_at"] == "2026-07-19T07:45:00+00:00"
    assert summary["status"] == "failed"


def test_maintenance_bash_syntax_ok():
    r = subprocess.run(
        ["bash", "-n", str(MAINTENANCE_SH)],
        capture_output=True, text=True, timeout=10,
    )
    assert r.returncode == 0, f"maintenance.sh syntax error: {r.stderr}"


# ---------------------------------------------------------------------------
# Wave 2 (2026-07-21): maintenance.sh now propagates RUN_ID + steps{}
# + consolidation{} so the dashboard can correlate sub-step state with
# a single canonical run. These tests guard that contract.
# ---------------------------------------------------------------------------


def test_summary_writes_run_id_and_steps_block(tmp_path, monkeypatch):
    """The maintenance-summary JSON carries ``run_id`` at the top
    level AND a populated ``steps{}`` block (memory_brain / ingest /
    reclassify / supersede / expire / snapshot / lexical_refresh) so
    the dashboard can render ingest noop / consolidation skipped
    without ambiguity.
    """
    log_body = (
        "[maintenance 2026-07-21T03:00:00Z] RUN_ID=20260721T030000Z-ridprop\n"
        "[maintenance 2026-07-21T03:00:00Z] starting maintenance\n"
        "[maintenance 2026-07-21T03:00:00Z] --- [1/1] collection=openclaw_memory_os ---\n"
        "[maintenance 2026-07-21T03:00:00Z] memory-brain: unified pipeline (ingest + consolidate)\n"
        "[maintenance 2026-07-21T03:00:00Z] [brain-step] run_id=20260721T030000Z-ridprop started=2026-07-21T03:00:00Z\n"
        "[maintenance 2026-07-21T03:00:00Z] step 1/5: ingest\n"
        "[brain-pipeline] run_id=20260721T030000Z-ridprop status=noop ingest_exit=0 consolidate_exit=0\n"
        "[brain-ingest] run_id=20260721T030000Z-ridprop files_processed=3 total_ingested=0 total_skipped=3 error_queue=0 status=noop\n"
        "[brain-substep] run_id=20260721T030000Z-ridprop name=ingest started=2026-07-21T03:00:00Z finished=2026-07-21T03:00:10Z exit=0\n"
        "[maintenance 2026-07-21T03:00:30Z] [brain-step] run_id=20260721T030000Z-ridprop finished=2026-07-21T03:00:30Z exit=0\n"
        "[brain-consolidate] run_id=20260721T030000Z-ridprop status=ok topics_merged=1 merged_topics=1 threshold=20 new_since_24h=3 total_points=250 dream_count=0\n"
        "[brain-substep] run_id=20260721T030000Z-ridprop name=consolidate started=2026-07-21T03:00:15Z finished=2026-07-21T03:00:30Z exit=0\n"
        "[maintenance 2026-07-21T03:01:00Z] ok\n"
    )
    log = tmp_path / "maintenance.log"
    log.write_text(log_body, encoding="utf-8")
    summary = tmp_path / "summary.json"
    test_env = os.environ.copy()
    test_env.update({
        "MAINTENANCE_RUN_ID": "20260721T030000Z-ridprop",
        "MAINTENANCE_MODE": "daily",
        "MAINTENANCE_STARTED_AT": "2026-07-21T03:00:00+00:00",
        "MAINTENANCE_FINISHED_AT": "2026-07-21T03:01:00+00:00",
        "MAINTENANCE_STATUS": "success",
        "MAINTENANCE_EXIT_CODE": "0",
        "MAINTENANCE_FAILED_STEP": "",
    })
    proc = subprocess.run(
        [str(PYTHON_BIN), str(SCRIPTS_DIR / "_write_summary.py"), str(log), str(summary)],
        capture_output=True,
        text=True,
        env=test_env,
        timeout=30,
    )
    assert proc.returncode == 0, f"_write_summary failed: {proc.stderr}"
    payload = json.loads(summary.read_text(encoding="utf-8"))
    # Top-level ``run_id`` round-trips.
    assert payload["run_id"] == "20260721T030000Z-ridprop"
    # ``mode`` propagates so the dashboard can label "daily" / "governance".
    assert payload["mode"] == "daily"
    # ``steps`` block is fully populated.
    assert "steps" in payload
    assert "memory_brain" in payload["steps"]
    assert "ingest" in payload["steps"]
    assert payload["steps"]["ingest"]["status"] == "noop"
    assert int(payload["steps"]["ingest"]["ingested_new"]) == 0
    assert payload["steps"]["memory_brain"]["run_id"] == "20260721T030000Z-ridprop"
    # ``consolidation`` block carries status + topics_merged.
    assert payload["consolidation"]["status"] == "ok"
    assert int(payload["consolidation"]["topics_merged"]) == 1
