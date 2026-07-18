"""Final freeze contracts for unattended production deployment."""
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent


def read(path: str) -> str:
    return (ROOT / path).read_text(encoding="utf-8")


def test_web_service_is_unprivileged_and_state_isolated() -> None:
    unit = read("deploy/systemd/openclaw-memory-os.service")
    assert "User=openclaw-memory-os" in unit
    assert "Group=openclaw-memory-os" in unit
    assert "User=root" not in unit
    assert "ProtectHome=yes" in unit
    assert "StateDirectory=openclaw-memory-os" in unit
    assert "/root/" not in unit
    assert "OPENCLAW_MEMORY_OS_LOG=/var/log/openclaw-memory-os/maintenance.log" in unit
    assert "MemoryMax=2G" in unit
    assert "UMask=0077" in unit


def test_persistent_maintenance_and_governance_timers_ship() -> None:
    maintenance = read("deploy/systemd/openclaw-memory-os-maintenance.timer")
    governance = read("deploy/systemd/openclaw-memory-os-governance.timer")
    assert "Persistent=true" in maintenance
    assert "07:45:00 Asia/Shanghai" in maintenance
    assert "Persistent=true" in governance
    assert "Tue *-*-* 04:01:00 Asia/Shanghai" in governance


def test_nginx_never_logs_or_propagates_query_tokens() -> None:
    nginx = read("deploy/nginx/memory-os.example.com.conf")
    assert "memory_os_no_args" in nginx
    assert "access_log /var/log/nginx/memory-os.example.com.access.log combined" not in nginx
    assert nginx.count('if ($arg_token != "") { return 400; }') == 2
    log_format = nginx.split("log_format memory_os_no_args", 1)[1].split(";", 1)[0]
    assert "$request_uri" not in log_format
    assert "$args" not in log_format


def test_operator_docs_do_not_teach_query_token_login() -> None:
    for path in (
        "README.md",
        "docs/deployment.md",
        "examples/cli-and-api.md",
        "scripts/run_demo.sh",
    ):
        assert "/dashboard?token=" not in read(path), path


def test_audited_python312_runtime_lock_is_complete_for_direct_dependencies() -> None:
    lock = read("requirements/runtime-py312.lock")
    for package in (
        "fastapi",
        "uvicorn",
        "Jinja2",
        "pydantic",
        "python-dotenv",
        "python-multipart",
        "argon2-cffi",
        "requests",
        "qdrant-client",
        "httpx",
        "hyperframe",
    ):
        assert any(
            line.startswith(package + "==")
            for line in lock.splitlines()
        ), package
    assert ">=" not in lock


def test_build_backend_is_frozen_and_uses_modern_license_metadata() -> None:
    pyproject = read("pyproject.toml")
    assert 'setuptools==83.0.0' in pyproject
    assert 'wheel==0.47.0' in pyproject
    assert 'license = "MIT"' in pyproject
    assert 'license-files = ["LICENSE"]' in pyproject
    assert 'license = { text = "MIT" }' not in pyproject
    assert 'License :: OSI Approved :: MIT License' not in pyproject


def test_logrotate_matches_private_service_logs() -> None:
    policy = read("deploy/logrotate/openclaw-memory-os")
    assert "/var/log/openclaw-memory-os/maintenance.log" in policy
    assert "/var/log/openclaw-memory-os/governance.log" in policy
    assert "create 0600 openclaw-memory-os openclaw-memory-os" in policy
    assert "su openclaw-memory-os openclaw-memory-os" in policy


def test_deployer_installs_locked_runtime_and_service_timers() -> None:
    deploy = read("deploy/deploy.sh")
    assert 'PYTHON_BIN="${PYTHON_BIN:-python3.12}"' in deploy
    assert "requirements/runtime-py312.lock" in deploy
    assert "--no-deps" in deploy
    assert "useradd --system" in deploy
    assert "openclaw-memory-os-maintenance.timer" in deploy
    assert "openclaw-memory-os-governance.timer" in deploy
    assert "/etc/logrotate.d/openclaw-memory-os" in deploy
    assert "ProtectHome=yes blocks home directories" in deploy
    assert "MEMORY_OS_DOMAIN" in deploy
    assert 'ACME_DOMAIN="$DOMAIN"' in deploy
    assert "Issue the certificate before installing the TLS vhost" in deploy
    assert "previous configuration restored" in deploy
    assert "previous policy restored" in deploy


def test_host_acceptance_requires_real_scale_and_fixed_models() -> None:
    acceptance = read("scripts/final_host_acceptance.sh")
    assert "FINAL_ACCEPTANCE_ACK" in acceptance
    assert "ACCEPTANCE_MIN_POINTS" in acceptance
    assert "ACCEPTANCE_QUERY" in acceptance
    assert "RESTORE_PROOF_FILE" in acceptance
    assert "EVOLUTION_PROOF_FILE" in acceptance
    assert "gitleaks is required" in acceptance
    assert "ACCEPTANCE_GOVERNANCE_GAP_SECONDS" in acceptance
    assert "nomic-embed-text" in acceptance
    assert "qwen2.5:1.5b" in acceptance
    assert "auth_smoke.sh" in acceptance
    assert "varied_perf_gate.sh" in acceptance
    assert acceptance.count("governance-window-") == 2
    assert 'sleep "$ACCEPTANCE_GOVERNANCE_GAP_SECONDS"' in acceptance
    assert "build-wheel" in acceptance
    assert "wheel-smoke" in acceptance
    assert 'git -C "$PROJECT_ROOT" archive' in acceptance
    assert "source-archive.sha256" in acceptance
    assert "download-runtime-wheelhouse" in acceptance
    assert "wheelhouse.sha256" in acceptance


def test_maintenance_and_governance_cannot_return_fake_green() -> None:
    maintenance = read("scripts/maintenance.sh")
    governance = read("scripts/autonomous_governance.sh")
    assert "failure_count" in maintenance
    assert 'if [ "$failure_count" -gt 0 ]' in maintenance
    assert "snapshot failed" in maintenance
    assert "evolution_rc" in governance
    assert 'elif [ "$evolution_rc" -ne 0 ]' in governance
    assert "final status write failed" in governance


def test_snapshot_backup_authenticates_and_names_fallback_archive_correctly() -> None:
    backup = read("scripts/backup_snapshot.sh")
    assert 'curl_auth=(-H "api-key: $QDRANT_API_KEY")' in backup
    assert 'QDRANT_API_KEY="$QDRANT_API_KEY"' in backup
    assert '.tar.gz"' in backup
    assert 'archive="$BACKUP_DIR/${COLLECTION}-${stamp}.tar.zst"' in backup


def test_production_environment_requires_real_domain() -> None:
    env_example = read(".env.example")
    deploy = read("deploy/deploy.sh")
    assert "MEMORY_OS_DOMAIN=memory-os.example.com" in env_example
    assert 'DOMAIN="${MEMORY_OS_DOMAIN:-${ACME_DOMAIN:-}}"' in deploy
    assert 'DOMAIN" != "memory-os.example.com"' in deploy
