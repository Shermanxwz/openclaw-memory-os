"""Regression test for the package version metadata.

This test guards against drift between the Python package's ``__version__``
constant and the version declared in ``pyproject.toml`` (the source of truth
that ``importlib.metadata`` exposes).

History: an external review (2026-07-14) flagged that ``__version__`` had
fallen behind ``pyproject.toml`` (0.2.14 vs 0.2.15). This test ensures that
specific drift cannot recur silently.
"""

from __future__ import annotations

import importlib.metadata
import re

import openclaw_memory_os


def test_package_version_matches_metadata() -> None:
    """``__version__`` must equal the version reported by importlib.metadata."""
    metadata_version = importlib.metadata.version("openclaw_memory_os")
    assert openclaw_memory_os.__version__ == metadata_version, (
        f"__version__={openclaw_memory_os.__version__!r} does not match "
        f"importlib.metadata.version('openclaw_memory_os')={metadata_version!r}"
    )


def test_version_is_semver() -> None:
    """Package version must look like a normal SemVer (or PEP 440 prerelease) string.

    Accepts both ``X.Y.Z`` and PEP-440-style alphanumeric prerelease
    suffixes (e.g. ``0.3.0a0``, ``0.3.0b1``, ``0.3.0-a0``) which
    setuptools normalises ``0.3.0-a0`` to at build time.
    """
    pattern = re.compile(r"^\d+\.\d+\.\d+(?:[ab][0-9]*|rc[0-9]*|[\-+][A-Za-z0-9.]+)?$")
    assert pattern.match(openclaw_memory_os.__version__), (
        f"__version__={openclaw_memory_os.__version__!r} is not SemVer-shaped"
    )


def test_version_is_at_least_0_2_15() -> None:
    """Belt-and-braces: this test exists to make sure 0.2.15 is shipped.

    The external review cut v0.2.15; any future regression to a lower value
    should fail loudly. Tolerates SemVer / PEP 440 prerelease suffixes
    such as ``0.3.0a0`` by stripping everything from the first non-digit
    suffix separator.
    """

    version = openclaw_memory_os.__version__
    for sep in ("a", "b", "rc", "-", "+"):
        if sep in version:
            version = version.split(sep, 1)[0]
    parts = tuple(int(piece) for piece in version.split(".")[:3])
    assert parts >= (0, 2, 15), (
        f"__version__={openclaw_memory_os.__version__!r} is older than 0.2.15"
    )