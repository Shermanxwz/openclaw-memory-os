"""Test that auth_smoke.sh does not destructively modify production state.

These tests verify the contract that auth_smoke.sh:
1. Uses the readonly helper instead of /usr/bin/sqlite3 for queries.
2. Opens the sessions DB in read-only mode for verification queries.
3. Does not contain any direct /usr/bin/sqlite3 calls on the production DB.
4. The readonly helper script exists and is executable.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
AUTH_SMOKE = REPO_ROOT / "scripts" / "auth_smoke.sh"
READONLY_HELPER = REPO_ROOT / "scripts" / "session_readonly_helper.py"


class TestAuthSmokeNondestructive:
    """Verify auth_smoke.sh does not destructively access the sessions DB."""

    @pytest.fixture()
    def smoke_content(self) -> str:
        return AUTH_SMOKE.read_text()

    def test_no_sqlite3_cli_calls(self, smoke_content: str) -> None:
        """auth_smoke.sh must not call /usr/bin/sqlite3 on the sessions DB.

        The standalone sqlite3 binary opens databases in read-write mode
        by default, which modifies WAL/SHM files even for SELECT queries.
        This was the root cause of the WAL corruption bug.
        """
        # Look for patterns like: sqlite3 "$SESSIONS_DB_PATH" or sqlite3 "$DB"
        # but NOT: import sqlite3 (Python) or comments mentioning sqlite3.
        lines = smoke_content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Skip comments.
            if stripped.startswith("#"):
                continue
            # Skip Python heredoc content (lines between PY markers).
            # Check for direct sqlite3 CLI invocation.
            # Pattern: sqlite3 followed by a variable or path (not "import sqlite3").
            if re.search(r'\bsqlite3\s+"?\$', stripped):
                # This is a CLI call like: sqlite3 "$SESSIONS_DB_PATH"
                pytest.fail(
                    f"Line {i}: Found /usr/bin/sqlite3 CLI call: {stripped}\n"
                    f"Use session_readonly_helper.py instead."
                )

    def test_uses_readonly_helper(self, smoke_content: str) -> None:
        """auth_smoke.sh should use session_readonly_helper.py for DB queries."""
        assert "session_readonly_helper.py" in smoke_content, (
            "auth_smoke.sh should reference session_readonly_helper.py "
            "for read-only DB queries"
        )

    def test_python_heredoc_uses_readonly_uri(self, smoke_content: str) -> None:
        """Python heredocs in auth_smoke.sh should use ?mode=ro URI mode."""
        # Find all Python heredoc blocks and check they use readonly mode.
        # The pattern is: sqlite3.connect(f"file:{path}?mode=ro", uri=True)
        # or similar readonly URI.
        lines = smoke_content.split("\n")
        in_python_block = False
        found_readonly_connect = False
        found_writable_connect = False

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            # Detect start of Python heredoc: lines ending with <<'PY' or similar.
            if "<<'PY'" in stripped or '<<"PY"' in stripped:
                in_python_block = True
                continue
            if in_python_block and stripped == "PY":
                in_python_block = False
                continue
            if in_python_block:
                if "sqlite3.connect" in stripped:
                    if "?mode=ro" in stripped and "uri=True" in stripped:
                        found_readonly_connect = True
                    elif "SessionStore" not in stripped:
                        # SessionStore is allowed for creating test sessions.
                        found_writable_connect = True

        assert found_readonly_connect, (
            "Python heredocs should use sqlite3.connect with ?mode=ro URI"
        )
        assert not found_writable_connect, (
            "Python heredocs should NOT use sqlite3.connect without ?mode=ro "
            "(except for SessionStore which is used for creating test data)"
        )

    def test_readonly_helper_exists(self) -> None:
        """The session_readonly_helper.py script must exist."""
        assert READONLY_HELPER.is_file(), (
            f"session_readonly_helper.py not found at {READONLY_HELPER}"
        )

    def test_readonly_helper_uses_readonly_uri(self) -> None:
        """The helper script must use ?mode=ro URI mode."""
        content = READONLY_HELPER.read_text()
        assert "?mode=ro" in content, (
            "session_readonly_helper.py must use ?mode=ro URI mode"
        )
        assert "query_only=ON" in content, (
            "session_readonly_helper.py must enable PRAGMA query_only=ON"
        )

    def test_readonly_helper_no_write_operations(self) -> None:
        """The helper script must not contain any write operations."""
        content = READONLY_HELPER.read_text()
        # Check for common write patterns.
        write_patterns = [
            r"\bINSERT\b",
            r"\bUPDATE\b",
            r"\bDELETE\b",
            r"\bDROP\b",
            r"\bALTER\b",
            r"\.execute\([^)]*INSERT",
            r"\.execute\([^)]*UPDATE",
            r"\.execute\([^)]*DELETE",
        ]
        for pattern in write_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                pytest.fail(
                    f"session_readonly_helper.py contains write operation: {pattern}"
                )

    def test_smoke_no_need_command_sqlite3(self, smoke_content: str) -> None:
        """auth_smoke.sh should not require the sqlite3 CLI."""
        lines = smoke_content.split("\n")
        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            if "need_command" in stripped and "sqlite3" in stripped:
                pytest.fail(
                    f"Line {i}: auth_smawke.sh should not require sqlite3 CLI: {stripped}"
                )
