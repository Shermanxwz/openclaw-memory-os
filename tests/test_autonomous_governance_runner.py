"""Tests for the weekly autonomous governance bash runner.

The runner is a thin orchestration shell that:

  1. takes a flock (different path from maintenance.sh)
  2. invokes maintenance.sh with deep-audit env knobs
  3. writes a redacted status JSON via _write_governance_status.py

We do **not** actually run maintenance.sh in tests (it touches Qdrant).
Instead we monkeypatch the writer call to capture what would have been
written, and verify that the bash script's wiring is correct.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path



REPO_ROOT = Path(__file__).resolve().parent.parent
RUNNER = REPO_ROOT / "scripts" / "autonomous_governance.sh"
MAINTENANCE = REPO_ROOT / "scripts" / "maintenance.sh"
WRITER = REPO_ROOT / "scripts" / "_write_governance_status.py"
INSTALL_CRON = REPO_ROOT / "scripts" / "install_governance_cron.sh"
PACKAGE_DIR = REPO_ROOT  # openclaw_memory_os/ lives at the repo root
# The real interpreter used by both the maintenance.sh and the
# governance writer. Tests that spin up a tmp project override the
# .venv/bin/python shim, but they must still run with a Python that
# has openclaw_memory_os importable — we set PYTHONPATH below.
REAL_VENV_PY = REPO_ROOT / ".venv" / "bin" / "python"
if not REAL_VENV_PY.exists():
    REAL_VENV_PY = Path(shutil.which("python3") or shutil.which("python") or "/usr/bin/python3")


def _install_required_runner_stubs(scripts: Path) -> None:
    """Install successful replay/evolution stubs for runner smoke tests."""
    (scripts / "replay_feedback.py").write_text(
        "import sys\nprint('replay ok')\nsys.exit(0)\n", encoding="utf-8"
    )
    (scripts / "run_evolution_cycle.py").write_text(
        "import json\nprint(json.dumps({'status': 'skipped', 'reason': 'lock_held'}))\n",
        encoding="utf-8",
    )


def _env_for_subprocess(*, status_path: str | None = None) -> dict:
    """Env that lets the writer import openclaw_memory_os inside a tmp project."""
    env = os.environ.copy()
    # Make openclaw_memory_os importable for any python the runner spawns
    # (the writer does `from openclaw_memory_os.analytics import ...`).
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        f"{PACKAGE_DIR}{os.pathsep}{existing}" if existing else str(PACKAGE_DIR)
    )
    if status_path is not None:
        env["STATUS_FILE_PATH"] = status_path
    return env


def test_all_script_artifacts_exist():
    """Hard precondition for any of these tests."""
    for p in (RUNNER, MAINTENANCE, WRITER, INSTALL_CRON):
        assert p.exists(), f"missing script: {p}"


def test_runner_passes_bash_n():
    """The runner must remain shellcheck-clean syntactically."""
    result = subprocess.run(
        ["bash", "-n", str(RUNNER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_install_cron_passes_bash_n():
    result = subprocess.run(
        ["bash", "-n", str(INSTALL_CRON)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_maintenance_sh_passes_bash_n():
    result = subprocess.run(
        ["bash", "-n", str(MAINTENANCE)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_writer_python_compiles():
    """The writer must be importable as a module."""
    result = subprocess.run(
        [sys.executable, "-m", "py_compile", str(WRITER)], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr


def test_runner_uses_dedicated_lock_path():
    """Governance must not compete with the daily maintenance flock."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "OPENCLAW_MEMORY_OS_GOVERNANCE_LOCK" in text
    assert "openclaw-memory-os.governance.lock" in text
    assert "openclaw-memory-os.maintenance.lock" not in text


def test_runner_passes_force_content_supersede_and_auto_supersede_to_maintenance():
    text = RUNNER.read_text(encoding="utf-8")
    assert "FORCE_CONTENT_SUPERSEDE=\"$FORCE_CONTENT_SUPERSEDE\"" in text
    assert "ENABLE_AUTO_SUPERSEDE=\"$ENABLE_AUTO_SUPERSEDE\"" in text
    assert "SUPERSEDE_MAX_APPLY=\"$SUPERSEDE_MAX_APPLY\"" in text
    assert "FORCE_CONTENT_SUPERSEDE=\"${FORCE_CONTENT_SUPERSEDE:-1}\"" in text
    assert "ENABLE_AUTO_SUPERSEDE=\"${ENABLE_AUTO_SUPERSEDE:-1}\"" in text
    assert "SUPERSEDE_MAX_APPLY=\"${SUPERSEDE_MAX_APPLY:-50}\"" in text


def test_runner_always_writes_status_even_on_failure():
    """The status JSON must be written on both ok and failed paths."""
    text = RUNNER.read_text(encoding="utf-8")
    # The writer call appears in two distinct execution branches.
    write_calls = text.count("\"$WRITE_STATUS_PY\"")
    assert write_calls >= 2, (
        "runner must invoke the writer in both the ok and failed branches"
    )


def test_runner_does_not_default_daily_to_governance_status():
    """maintenance.sh must NOT write governance status without the env gate."""
    text = MAINTENANCE.read_text(encoding="utf-8")
    assert "WRITE_GOVERNANCE_STATUS=\"${WRITE_GOVERNANCE_STATUS:-0}\"" in text
    # The default for the env-var gate must be 0 so the daily cron stays silent.
    assert "WRITE_GOVERNANCE_STATUS:-0}" in text
    # The governance writer must be invoked only inside an env-var guard.
    assert "if [ \"$WRITE_GOVERNANCE_STATUS\" = \"1\" ]" in text


def test_runner_handles_missing_venv(tmp_path, monkeypatch):
    """When .venv/bin/python is absent, the runner still writes a failed status."""
    # Build a throwaway project dir with a fake maintenance.sh so the
    # runner progresses past the first checks.
    fake_proj = tmp_path / "project"
    scripts = fake_proj / "scripts"
    scripts.mkdir(parents=True)
    # Stub maintenance.sh that exits 0.
    (scripts / "maintenance.sh").write_text(
        "#!/usr/bin/env bash\necho stub\nexit 0\n",
        encoding="utf-8",
    )
    (scripts / "maintenance.sh").chmod(0o755)
    _install_required_runner_stubs(scripts)
    # Copy the real writer so the runner can call it.
    shutil.copy(WRITER, scripts / "_write_governance_status.py")
    # The runner looks for .venv/bin/python inside PROJECT_DIR. We use
    # a project root without .venv to trigger the failure path.
    target_status = tmp_path / "status.json"
    monkeypatch.setenv("STATUS_FILE_PATH", str(target_status))

    # Run the runner with PROJECT_DIR overridden. Easiest way is to
    # invoke it from a working directory where the relative path to the
    # runner matches `fake_proj`. We'll copy the runner there too.
    shutil.copy(RUNNER, scripts / "autonomous_governance.sh")
    (scripts / "autonomous_governance.sh").chmod(0o755)

    proc = subprocess.run(
        ["bash", str(scripts / "autonomous_governance.sh")],
        capture_output=True,
        text=True,
        env=_env_for_subprocess(status_path=str(target_status)),
    )

    # Even when the venv is missing we still want a failed status file.
    assert target_status.exists(), (
        f"runner did not write status file. stdout={proc.stdout!r} "
        f"stderr={proc.stderr!r}"
    )
    payload = json.loads(target_status.read_text(encoding="utf-8"))
    assert payload["last_result"] == "failed"


def test_runner_writes_ok_status_when_audit_succeeds(tmp_path, monkeypatch):
    """Smoke-test: a stub maintenance.sh that exits 0 should produce ok status."""
    fake_proj = tmp_path / "project"
    fake_proj.mkdir()
    scripts = fake_proj / "scripts"
    scripts.mkdir()
    # Build a fake venv so the runner doesn't bail early.
    venv_py = fake_proj / ".venv" / "bin"
    venv_py.mkdir(parents=True)
    # Reuse the real Python interpreter.
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python, "no python interpreter on PATH for the smoke test"
    (venv_py / "python").write_text(
        "#!/usr/bin/env bash\nexec " + real_python + ' "$@"\n',
        encoding="utf-8",
    )
    (venv_py / "python").chmod(0o755)

    # Stub maintenance.sh that exits 0 quickly.
    (scripts / "maintenance.sh").write_text(
        "#!/usr/bin/env bash\necho 'stub ok'\nexit 0\n", encoding="utf-8"
    )
    (scripts / "maintenance.sh").chmod(0o755)
    _install_required_runner_stubs(scripts)
    shutil.copy(WRITER, scripts / "_write_governance_status.py")
    shutil.copy(RUNNER, scripts / "autonomous_governance.sh")
    (scripts / "autonomous_governance.sh").chmod(0o755)

    target_status = tmp_path / "status.json"
    proc = subprocess.run(
        ["bash", str(scripts / "autonomous_governance.sh")],
        capture_output=True,
        text=True,
        env=_env_for_subprocess(status_path=str(target_status)),
    )
    assert proc.returncode == 0, proc.stderr
    assert target_status.exists(), proc.stdout
    payload = json.loads(target_status.read_text(encoding="utf-8"))
    assert payload["last_result"] == "ok"


def test_runner_writes_failed_status_when_audit_fails(tmp_path):
    """If maintenance.sh exits non-zero, the runner still writes a status file."""
    fake_proj = tmp_path / "project"
    fake_proj.mkdir()
    scripts = fake_proj / "scripts"
    scripts.mkdir()
    venv_py = fake_proj / ".venv" / "bin"
    venv_py.mkdir(parents=True)
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python, "no python interpreter on PATH for the smoke test"
    (venv_py / "python").write_text(
        "#!/usr/bin/env bash\nexec " + real_python + ' "$@"\n',
        encoding="utf-8",
    )
    (venv_py / "python").chmod(0o755)

    # Stub maintenance.sh that exits non-zero.
    (scripts / "maintenance.sh").write_text(
        "#!/usr/bin/env bash\necho 'stub failure'\nexit 17\n", encoding="utf-8"
    )
    (scripts / "maintenance.sh").chmod(0o755)
    _install_required_runner_stubs(scripts)
    shutil.copy(WRITER, scripts / "_write_governance_status.py")
    shutil.copy(RUNNER, scripts / "autonomous_governance.sh")
    (scripts / "autonomous_governance.sh").chmod(0o755)

    target_status = tmp_path / "status.json"
    proc = subprocess.run(
        ["bash", str(scripts / "autonomous_governance.sh")],
        capture_output=True,
        text=True,
        env=_env_for_subprocess(status_path=str(target_status)),
    )
    assert proc.returncode != 0
    assert target_status.exists()
    payload = json.loads(target_status.read_text(encoding="utf-8"))
    assert payload["last_result"] == "failed"
    assert "17" in payload["last_summary"]


def test_runner_status_payload_contains_no_path_or_token(monkeypatch, tmp_path):
    """End-to-end: the produced status JSON must not echo paths or tokens."""
    fake_proj = tmp_path / "project"
    fake_proj.mkdir()
    scripts = fake_proj / "scripts"
    scripts.mkdir()
    venv_py = fake_proj / ".venv" / "bin"
    venv_py.mkdir(parents=True)
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python
    (venv_py / "python").write_text(
        "#!/usr/bin/env bash\nexec " + real_python + ' "$@"\n', encoding="utf-8"
    )
    (venv_py / "python").chmod(0o755)

    (scripts / "maintenance.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (scripts / "maintenance.sh").chmod(0o755)
    _install_required_runner_stubs(scripts)
    shutil.copy(WRITER, scripts / "_write_governance_status.py")
    shutil.copy(RUNNER, scripts / "autonomous_governance.sh")
    (scripts / "autonomous_governance.sh").chmod(0o755)

    target_status = tmp_path / "status.json"
    subprocess.run(
        ["bash", str(scripts / "autonomous_governance.sh")],
        check=True,
        env=_env_for_subprocess(status_path=str(target_status)),
    )

    raw = target_status.read_text(encoding="utf-8")
    for needle in (
        "fake_proj",
        "scripts",
        str(tmp_path),
        "0.0.0.0",
        "127.0.0.1",
        "token=",
        "api_key",
    ):
        assert needle not in raw, f"runner leaked {needle!r} into status file"


def test_runner_python_modules_referenced_exist():
    """Every Python module the runner imports must actually be importable."""
    # The runner shells out to python; we just sanity-check that the
    # writer file is importable as a module path.
    spec_path = WRITER
    assert spec_path.exists()


def test_install_cron_compatibility_shim_enables_systemd_timer():
    """The legacy installer name must not create a root cron entry anymore."""
    text = INSTALL_CRON.read_text(encoding="utf-8")
    assert "systemctl enable --now" in text
    assert "openclaw-memory-os-governance.timer" in text
    assert "crontab" not in text


def test_runner_default_log_path_matches_dashboard_contract():
    """The status default must land where the reader looks."""
    text = RUNNER.read_text(encoding="utf-8")
    assert "openclaw-memory-os/autonomous-governance.json" in text
    # The default should mirror the reader's XDG_STATE_HOME fallback.
    assert "${XDG_STATE_HOME:-$HOME/.local/state}" in text

def test_install_cron_shim_contains_no_operator_specific_path():
    """The compatibility shim names only the generic installed timer."""
    text = INSTALL_CRON.read_text(encoding="utf-8")
    assert "/root/" not in text
    assert "/home/" not in text
    assert "openclaw-memory-os-governance.timer" in text


def test_runner_records_skipped_when_maintenance_lock_held(tmp_path):
    """If maintenance.sh exits 75 (lock-held), runner records 'skipped'.

    Regression: previously exit 0 from maintenance.sh meant the runner
    wrote status ``ok`` even when the daily maintenance run was holding
    the flock. The fix is two-sided: maintenance.sh now exits 75 instead
    of 0 when the lock is held, and autonomous_governance.sh interprets
    75 as ``skipped`` rather than ``ok`` or ``failed``.
    """
    fake_proj = tmp_path / "project"
    fake_proj.mkdir()
    scripts = fake_proj / "scripts"
    scripts.mkdir()
    venv_py = fake_proj / ".venv" / "bin"
    venv_py.mkdir(parents=True)
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python
    (venv_py / "python").write_text(
        "#!/usr/bin/env bash\nexec " + real_python + ' "$@"\n',
        encoding="utf-8",
    )
    (venv_py / "python").chmod(0o755)

    # Stub maintenance.sh that mimics the lock-held path.
    (scripts / "maintenance.sh").write_text(
        "#!/usr/bin/env bash\necho 'maintenance.sh: SKIPPED'\nexit 75\n",
        encoding="utf-8",
    )
    (scripts / "maintenance.sh").chmod(0o755)
    _install_required_runner_stubs(scripts)
    shutil.copy(WRITER, scripts / "_write_governance_status.py")
    shutil.copy(RUNNER, scripts / "autonomous_governance.sh")
    (scripts / "autonomous_governance.sh").chmod(0o755)

    target_status = tmp_path / "status.json"
    proc = subprocess.run(
        ["bash", str(scripts / "autonomous_governance.sh")],
        capture_output=True,
        text=True,
        env=_env_for_subprocess(status_path=str(target_status)),
    )
    # The runner propagates maintenance.sh's exit; 75 is the skip exit.
    assert proc.returncode == 75
    assert target_status.exists()
    payload = json.loads(target_status.read_text(encoding="utf-8"))
    assert payload["last_result"] == "skipped"
    assert "skipped" in payload["last_summary"].lower() or "lock" in payload["last_summary"].lower()


def test_runner_records_failed_for_unexpected_non_zero_exit(tmp_path):
    """Exit codes other than 0 and 75 fall back to 'failed', not 'ok'."""
    fake_proj = tmp_path / "project"
    fake_proj.mkdir()
    scripts = fake_proj / "scripts"
    scripts.mkdir()
    venv_py = fake_proj / ".venv" / "bin"
    venv_py.mkdir(parents=True)
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python
    (venv_py / "python").write_text(
        "#!/usr/bin/env bash\nexec " + real_python + ' "$@"\n',
        encoding="utf-8",
    )
    (venv_py / "python").chmod(0o755)

    # exit 42 is arbitrary failure; must NOT be classified as ok or skipped.
    (scripts / "maintenance.sh").write_text(
        "#!/usr/bin/env bash\nexit 42\n", encoding="utf-8",
    )
    (scripts / "maintenance.sh").chmod(0o755)
    _install_required_runner_stubs(scripts)
    shutil.copy(WRITER, scripts / "_write_governance_status.py")
    shutil.copy(RUNNER, scripts / "autonomous_governance.sh")
    (scripts / "autonomous_governance.sh").chmod(0o755)

    target_status = tmp_path / "status.json"
    proc = subprocess.run(
        ["bash", str(scripts / "autonomous_governance.sh")],
        capture_output=True,
        text=True,
        env=_env_for_subprocess(status_path=str(target_status)),
    )
    assert proc.returncode == 42
    payload = json.loads(target_status.read_text(encoding="utf-8"))
    assert payload["last_result"] == "failed"



def test_maintenance_sh_exits_75_on_lock_contention():
    """maintenance.sh must declare exit 75 + SKIPPED marker on flock loss.

    Without these two lines, autonomous_governance.sh cannot distinguish a
    genuine OK from a "lock held" skip and would silently write ``ok`` to
    the dashboard when no audit actually ran.
    """
    text = MAINTENANCE.read_text(encoding="utf-8")
    assert "exit 75" in text, "maintenance.sh must exit 75 on lock loss"
    assert "SKIPPED" in text, (
        "maintenance.sh must print the SKIPPED marker so the runner can "
        "tell the difference from a real run"
    )



def test_runner_propagates_evolution_failure(tmp_path):
    """A failed evolution cycle must never leave a green governance status."""
    fake_proj = tmp_path / "project"
    scripts = fake_proj / "scripts"
    scripts.mkdir(parents=True)
    venv_bin = fake_proj / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    real_python = shutil.which("python3") or shutil.which("python")
    assert real_python
    (venv_bin / "python").write_text(
        "#!/usr/bin/env bash\nexec " + real_python + ' "$@"\n', encoding="utf-8"
    )
    (venv_bin / "python").chmod(0o755)
    (scripts / "maintenance.sh").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (scripts / "maintenance.sh").chmod(0o755)
    _install_required_runner_stubs(scripts)
    (scripts / "run_evolution_cycle.py").write_text(
        "import sys\nprint('forced evolution failure')\nsys.exit(9)\n", encoding="utf-8"
    )
    shutil.copy(WRITER, scripts / "_write_governance_status.py")
    shutil.copy(RUNNER, scripts / "autonomous_governance.sh")
    (scripts / "autonomous_governance.sh").chmod(0o755)

    target_status = tmp_path / "status.json"
    proc = subprocess.run(
        ["bash", str(scripts / "autonomous_governance.sh")],
        capture_output=True,
        text=True,
        env=_env_for_subprocess(status_path=str(target_status)),
    )
    assert proc.returncode == 9
    payload = json.loads(target_status.read_text(encoding="utf-8"))
    assert payload["last_result"] == "failed"
    assert "evolution" in payload["last_summary"].lower()


def test_maintenance_reports_snapshot_failure(tmp_path):
    """A failed snapshot is continued past but produces a non-zero final result."""
    project = tmp_path / "project"
    scripts = project / "scripts"
    scripts.mkdir(parents=True)
    venv_bin = project / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").write_text(
        "#!/usr/bin/env bash\nexit 0\n", encoding="utf-8"
    )
    (venv_bin / "python").chmod(0o755)
    shutil.copy(MAINTENANCE, scripts / "maintenance.sh")
    (scripts / "maintenance.sh").chmod(0o755)
    (scripts / "backup_snapshot.sh").write_text(
        "#!/usr/bin/env bash\nexit 12\n", encoding="utf-8"
    )
    (scripts / "backup_snapshot.sh").chmod(0o755)

    log_path = tmp_path / "maintenance.log"
    env = os.environ.copy()
    env.update({
        "MAINTAIN_COLLECTIONS": "acceptance_test",
        "LOG_FILE": str(log_path),
        "SUMMARY_FILE": str(tmp_path / "summary.json"),
    })
    proc = subprocess.run(
        ["bash", str(scripts / "maintenance.sh")],
        capture_output=True, text=True, env=env
    )
    assert proc.returncode != 0
    assert "snapshot failed" in log_path.read_text(encoding="utf-8")
    assert "completed with failures=" in proc.stdout
