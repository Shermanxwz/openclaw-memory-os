"""Regression test for scripts/maintenance.sh memory_brain invocation.

External review (2026-07-14) flagged that ``run_memory_brain()`` was calling
``/usr/bin/python3`` directly, bypassing the project venv. That meant the
memory_brain pipeline silently depended on whatever happens to be installed
in the system Python — e.g. if a future base image upgrade drops
``requests``, the brain pipeline breaks but the surrounding maintenance
loop swallows the error and prints "ok".

This test reads ``scripts/maintenance.sh`` as text and asserts that any
memory_brain subprocess invocation goes through ``$VENV_PY`` (or
equivalent venv path), never a hard-coded system interpreter.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
MAINTENANCE_SH = REPO_ROOT / "scripts" / "maintenance.sh"


def _extract_run_memory_brain(source: str) -> str:
    """Return the text of the ``run_memory_brain()`` function body."""
    match = re.search(
        r"^run_memory_brain\(\)\s*\{(.*?)^\}",
        source,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "run_memory_brain() function not found in maintenance.sh"
    return match.group(1)


def test_maintenance_sh_exists() -> None:
    assert MAINTENANCE_SH.is_file(), f"missing {MAINTENANCE_SH}"


def test_maintenance_sh_does_not_hardcode_system_python_for_brain() -> None:
    """run_memory_brain() must not invoke /usr/bin/python3."""
    source = MAINTENANCE_SH.read_text(encoding="utf-8")
    body = _extract_run_memory_brain(source)

    forbidden = [
        "/usr/bin/python3",
        "/usr/local/bin/python3",
        "$(command -v python3)",  # falls back to system when venv not on PATH
    ]
    for needle in forbidden:
        assert needle not in body, (
            f"maintenance.sh run_memory_brain() still calls {needle!r}; "
            "must use $VENV_PY (project venv) instead."
        )


def test_maintenance_sh_brain_uses_venv_python() -> None:
    """run_memory_brain() must invoke $VENV_PY for both ingest + consolidate."""
    source = MAINTENANCE_SH.read_text(encoding="utf-8")
    body = _extract_run_memory_brain(source)

    # The script stores the brain script paths in MEMORY_BRAIN_INGEST /
    # MEMORY_BRAIN_CONSOLIDATE variables. What we really care about is
    # that the call site uses $VENV_PY, not /usr/bin/python3.
    ingest_call = re.search(
        r'"\$\{?VENV_PY\}?"\s+["\']?\$\{?MEMORY_BRAIN_INGEST\}?',
        body,
    )
    consolidate_call = re.search(
        r'"\$\{?VENV_PY\}?"\s+["\']?\$\{?MEMORY_BRAIN_CONSOLIDATE\}?',
        body,
    )

    assert ingest_call, "memory_brain_ingest.py must be invoked via $VENV_PY"
    assert consolidate_call, "memory_brain_consolidate.py must be invoked via $VENV_PY"


def test_pyproject_declares_requests_dependency() -> None:
    """``requests`` must be pinned in pyproject dependencies, not assumed."""
    pyproject = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    # Be lenient on the version specifier but require some form of "requests>=..."
    assert re.search(r'"requests\s*>=\s*\d', pyproject), (
        "pyproject.toml should declare `requests>=...` in dependencies so "
        "the venv install pulls it in. Without this, the memory_brain "
        "scripts would silently rely on the system interpreter."
    )


def test_venv_has_requests_installed() -> None:
    """The project venv must actually have requests importable.

    This catches a stale ``.venv`` that was set up before ``requests`` was
    added to ``pyproject.toml``.
    """
    import subprocess

    venv_python = REPO_ROOT / ".venv" / "bin" / "python"
    if not venv_python.is_file():
        # No venv at all — nothing to assert, but also nothing to fix here.
        return
    result = subprocess.run(
        [str(venv_python), "-c", "import requests; print(requests.__version__)"],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, (
        f".venv/bin/python cannot import requests; reinstall the venv.\n"
        f"stderr: {result.stderr}"
    )
    assert result.stdout.strip(), "requests.__version__ came back empty"