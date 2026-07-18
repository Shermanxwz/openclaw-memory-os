from __future__ import annotations

import subprocess
from pathlib import Path

SCRIPT = Path("scripts/auth_smoke.sh")


def test_auth_smoke_has_valid_bash_syntax():
    proc = subprocess.run(["bash", "-n", str(SCRIPT)], capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr


def test_auth_smoke_covers_graduation_negative_cases():
    source = SCRIPT.read_text(encoding="utf-8")
    required = (
        "password cannot be bearer",
        "random ${length}-character bearer",
        "correct password missing TOTP",
        "correct password wrong TOTP",
        "wrong password correct TOTP",
        "login without CSRF cookie/form",
        "login with mismatched CSRF",
        "logout missing CSRF",
        "logout wrong CSRF",
        "revoked cookie immediately rejected",
        "revoked cookie rejected after restart",
        "active session after restart",
        "expired SessionStore cookie",
        "session cookie equals MEMORY_OS_TOKEN",
        "session cookie equals MEMORY_OS_PASSWORD",
        "token_hash",
        "raw secret",
    )
    for marker in required:
        assert marker in source, marker


def test_auth_smoke_never_prints_secret_prefixes():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "${TOKEN:0:" not in source
    assert "${PASSWORD:0:" not in source
    assert "${TOTP_SECRET:0:" not in source
    assert "totp_secret:" not in source


def test_auth_smoke_is_explicitly_host_side():
    source = SCRIPT.read_text(encoding="utf-8")
    assert "REQUIRE_SYSTEMD" in source
    assert "systemctl restart" in source
    assert "GitHub-hosted CI" in source
