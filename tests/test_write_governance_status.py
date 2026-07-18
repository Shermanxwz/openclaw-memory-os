"""Unit tests for the autonomous governance status writer.

Covers:

* happy-path write of a small redacted JSON
* schema discipline (only the three contract keys are written)
* permission hardening (file mode 0600, dir mode 0700 on Unix)
* auto-mkdir of the parent directory
* overwrite semantics (second write replaces, never accumulates)
* result_token fallback (empty / unknown token → ``ok``/``failed``)
* summary redaction: whitespace collapsing, control char stripping,
  300-char truncation with ellipsis
* path resolution: explicit arg > MEMORY_OS_GOVERNANCE_STATUS env > default
"""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path

import pytest

from openclaw_memory_os.analytics import (
    _ALLOWED_RESULT_TOKENS,
    _sanitize_summary,
    write_autonomous_governance_status,
)


@pytest.fixture
def isolated_env(monkeypatch, tmp_path):
    """Make sure the writer never reads a real user's XDG state dir."""
    monkeypatch.delenv("MEMORY_OS_GOVERNANCE_STATUS", raising=False)
    return tmp_path


def test_happy_path_writes_three_keys(isolated_env):
    target = isolated_env / "governance.json"
    written = write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="deep audit completed",
    )
    assert written == target
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"last_run", "last_result", "last_summary"}
    assert payload["last_result"] == "ok"
    assert payload["last_summary"] == "deep audit completed"
    assert "+08:00" in payload["last_run"]


def test_schema_contains_no_extra_fields(isolated_env):
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="failed",
        summary="manual test",
    )
    raw = json.loads(target.read_text(encoding="utf-8"))
    # The contract is exactly three keys. Anything else is a leak vector.
    forbidden = {
        "collections",
        "collection",
        "paths",
        "path",
        "tokens",
        "token",
        "ip",
        "host",
        "file",
        "reason",
        "counts",
    }
    for key in forbidden:
        assert key not in raw, f"unexpected key {key!r} in status JSON"


def test_summary_sanitisation_does_not_rewrite_callers(monkeypatch, isolated_env):
    """The writer is not a magic redaction engine.

    It only normalises whitespace, strips control characters, and truncates
    at 300 characters. The hard contract against embedding collection
    names / paths / IPs / tokens is an obligation on the *caller* (the
    bash runner). This test pins that boundary so a future refactor
    does not silently turn the writer into a redaction filter — that
    would mask caller mistakes instead of surfacing them.
    """
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="plain human summary",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_summary"] == "plain human summary"


@pytest.mark.skipif(os.name == "nt", reason="Unix permission semantics only")
def test_file_mode_0600_on_unix(isolated_env):
    target = isolated_env / "perm" / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="permission check",
    )
    mode = stat.S_IMODE(target.stat().st_mode)
    assert mode == 0o600, f"file mode was {oct(mode)}, expected 0o600"


@pytest.mark.skipif(os.name == "nt", reason="Unix permission semantics only")
def test_directory_mode_0700_on_unix(isolated_env):
    target = isolated_env / "perm2" / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="dir check",
    )
    dir_mode = stat.S_IMODE(target.parent.stat().st_mode)
    assert dir_mode == 0o700, f"dir mode was {oct(dir_mode)}, expected 0o700"


def test_missing_parent_dir_is_created(isolated_env):
    target = isolated_env / "deep" / "nested" / "gov.json"
    assert not target.parent.exists()
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="mkdir check",
    )
    assert target.exists()
    assert target.parent.exists()


def test_second_write_overwrites_cleanly(isolated_env):
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="first run",
    )
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="failed",
        summary="second run",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_result"] == "failed"
    assert payload["last_summary"] == "second run"
    # Still exactly three keys — overwrite must not accumulate stale fields.
    assert set(payload.keys()) == {"last_run", "last_result", "last_summary"}


def test_empty_result_token_falls_back_to_ok(isolated_env):
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="",
        summary="empty token",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_result"] == "ok"


def test_unknown_result_token_falls_back_to_failed(isolated_env):
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="not-a-real-state",
        summary="unknown token",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_result"] == "failed"


@pytest.mark.parametrize("token", ["ok", "failed", "running", "pending"])
def test_all_allowed_tokens_round_trip(isolated_env, token):
    target = isolated_env / f"gov-{token}.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token=token,
        summary=f"state={token}",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_result"] == token


def test_summary_truncated_at_300_chars(isolated_env):
    target = isolated_env / "gov.json"
    long_summary = "x" * 1000
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary=long_summary,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    # Truncation uses 300-char cap with "..." suffix.
    assert len(payload["last_summary"]) == 300
    assert payload["last_summary"].endswith("...")


def test_summary_collapses_whitespace(isolated_env):
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="multi\n\n\t  line   summary",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    # Internal newlines and tabs collapse to single spaces.
    assert payload["last_summary"] == "multi line summary"


def test_summary_drops_control_characters(isolated_env):
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="hello\x00\x07world",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert "\x00" not in payload["last_summary"]
    assert "\x07" not in payload["last_summary"]
    assert "helloworld" in payload["last_summary"]


def test_summary_empty_string_is_kept(isolated_env):
    target = isolated_env / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="",
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_summary"] == ""


def test_path_resolution_prefers_explicit_arg(monkeypatch, isolated_env):
    sentinel = isolated_env / "explicit.json"
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(isolated_env / "env.json"))
    written = write_autonomous_governance_status(
        status_file_path=sentinel,
        result_token="ok",
        summary="explicit",
    )
    assert written == sentinel
    assert sentinel.exists()
    assert not (isolated_env / "env.json").exists()


def test_path_resolution_falls_back_to_env(monkeypatch, isolated_env):
    env_target = isolated_env / "from-env.json"
    monkeypatch.setenv("MEMORY_OS_GOVERNANCE_STATUS", str(env_target))
    written = write_autonomous_governance_status(
        status_file_path=None,
        result_token="ok",
        summary="env",
    )
    assert written == env_target
    assert env_target.exists()


def test_finished_at_overrides_now(isolated_env):
    target = isolated_env / "gov.json"
    fixed = "2026-07-14T04:01:00+08:00"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="fixed",
        finished_at=fixed,
    )
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_run"] == fixed


def test_sanitize_summary_does_not_crash_on_non_string():
    # Defensive: callers should always pass str, but if they don't we
    # should not blow up the writer.
    assert _sanitize_summary("") == ""
    assert _sanitize_summary("plain") == "plain"


def test_allowed_result_tokens_set_includes_skipped():
    """The dashboard contract is the source of truth; keep these aligned.

    ``skipped`` was added on 2026-07-14 to distinguish a true deep-audit
    completion (``ok``) from a no-op run where ``scripts/maintenance.sh``
    exited 75 because another process held the maintenance flock.
    """
    assert _ALLOWED_RESULT_TOKENS == {"ok", "failed", "degraded", "running", "pending", "skipped"}


def test_writer_does_not_leak_via_tempfile(tmp_path, monkeypatch):
    """After a successful write the tempfile must not linger."""
    target = tmp_path / "gov.json"
    write_autonomous_governance_status(
        status_file_path=target,
        result_token="ok",
        summary="cleanup check",
    )
    leftovers = [
        p for p in tmp_path.iterdir() if p.name.startswith(".autonomous-governance-")
    ]
    assert leftovers == [], f"tempfile leftovers: {leftovers}"



def test_writer_stdlib_fallback_runs_without_package_on_path(tmp_path, monkeypatch):
    """The CLI wrapper's import guard lets the runner work even if the
    project package is not importable (e.g. the bash runner's
    missing-venv fallback on CI).

    Regression: the runner used to crash with
    ``ModuleNotFoundError: No module named 'openclaw_memory_os'`` when
    the project venv was missing and the system Python did not have the
    package installed, leaving the dashboard with no status write.
    The CLI now embeds a stdlib-only mirror of
    ``write_autonomous_governance_status`` so the same 3-key JSON is
    produced either way.
    """
    import json as _json
    import subprocess

    writer_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "_write_governance_status.py"
    )
    target = tmp_path / "fallback.json"

    # Force ``openclaw_memory_os`` to be unimportable: scrub PYTHONPATH
    # and run with ``-S`` so the caller site-packages is also masked.
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(tmp_path),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    # Pick a Python that can run the writer but where the package is
    # genuinely absent. Using /usr/bin/python3 mirrors what the bash
    # runner does on a missing venv.
    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-S",
            str(writer_path),
            str(target),
            "failed",
            "missing venv",
        ],
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0, (
        f"writer crashed under stdlib fallback: stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
    assert target.exists()
    payload = _json.loads(target.read_text(encoding="utf-8"))
    assert set(payload.keys()) == {"last_run", "last_result", "last_summary"}
    assert payload["last_result"] == "failed"
    assert payload["last_summary"] == "missing venv"
    # Permission discipline still applies in the fallback path.
    assert (target.stat().st_mode & 0o777) == 0o600


def test_writer_stdlib_fallback_rejects_unknown_token(tmp_path):
    """Even the fallback must coerce unknown tokens to ``failed``."""
    import subprocess

    writer_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "_write_governance_status.py"
    )
    target = tmp_path / "fallback-bad.json"

    result = subprocess.run(
        [
            "/usr/bin/python3",
            "-S",
            str(writer_path),
            str(target),
            "totally-not-a-real-token",
            "noise",
        ],
        capture_output=True,
        text=True,
        env={"PATH": os.environ.get("PATH", ""), "HOME": str(tmp_path)},
    )
    assert result.returncode == 0
    payload = json.loads(target.read_text(encoding="utf-8"))
    assert payload["last_result"] == "failed"

