from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from openclaw_memory_os import sessions
from openclaw_memory_os.sessions import SessionStore


pytestmark = pytest.mark.skipif(os.name == "nt", reason="POSIX mode contract")


def _mode(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_session_store_creates_owner_only_db_and_sidecars(tmp_path: Path) -> None:
    db_path = tmp_path / "state" / "sessions.db"
    old_umask = os.umask(0)
    try:
        store = SessionStore(db_path)
    finally:
        os.umask(old_umask)
    try:
        store.create("opaque-cookie", 3600)
        assert _mode(db_path) == 0o600
        for suffix in ("-wal", "-shm"):
            sidecar = db_path.with_name(db_path.name + suffix)
            if sidecar.exists():
                assert _mode(sidecar) == 0o600
    finally:
        store.close()


def test_session_store_repairs_existing_permissive_db(tmp_path: Path) -> None:
    db_path = tmp_path / "sessions.db"
    db_path.touch(mode=0o644)
    os.chmod(db_path, 0o644)
    store = SessionStore(db_path)
    try:
        assert _mode(db_path) == 0o600
    finally:
        store.close()


def test_session_store_does_not_chmod_existing_shared_parent(tmp_path: Path) -> None:
    shared = tmp_path / "shared-state"
    shared.mkdir(mode=0o755)
    os.chmod(shared, 0o755)
    store = SessionStore(shared / "sessions.db")
    try:
        assert _mode(shared) == 0o755
    finally:
        store.close()


def test_session_store_permission_failure_is_fatal(tmp_path: Path, monkeypatch) -> None:
    def deny_fchmod(fd: int, mode: int) -> None:
        raise OSError("simulated permission denial")

    monkeypatch.setattr(sessions.os, "fchmod", deny_fchmod)
    with pytest.raises(PermissionError, match="cannot secure session database"):
        SessionStore(tmp_path / "sessions.db")


def test_session_store_rejects_symlink_db(tmp_path: Path) -> None:
    target = tmp_path / "target.db"
    target.touch(mode=0o600)
    link = tmp_path / "sessions.db"
    link.symlink_to(target)
    with pytest.raises(PermissionError, match="must not be a symlink"):
        SessionStore(link)
