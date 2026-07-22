"""Tests for scripts/_prune_helpers.py (the Qdrant cache prune helper).

The cache lives in ``/opt/qdrant/backup`` on production. Here we
exercise the prune logic against a temporary directory so the test
is hermetic: no real Qdrant, no root, no flakes.
"""
from __future__ import annotations

import importlib.util
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def _load_module():
    spec = importlib.util.spec_from_file_location(
        "prune_helpers", SCRIPTS_DIR / "_prune_helpers.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_snapshot(path: Path, *, mtime_offset: float = 0.0) -> None:
    """Create a zero-byte snapshot file and force its mtime."""
    path.write_bytes(b"")
    # We need a deterministic, strictly-increasing mtime sequence so
    # the prune logic (which sorts by mtime) is unambiguous.
    base = 1_700_000_000.0  # arbitrary, just before "now"
    os.utime(path, (base + mtime_offset, base + mtime_offset))


def test_list_cache_files_returns_newest_first(tmp_path: Path):
    mod = _load_module()
    cache = tmp_path / "qcache"
    cache.mkdir()
    files = [
        cache / "openclaw_memory_os-20260101.snapshot",
        cache / "openclaw_memory_os-20260102.snapshot",
        cache / "openclaw_memory_os-20260103.snapshot",
    ]
    for i, p in enumerate(files):
        _make_snapshot(p, mtime_offset=float(i))

    listed = mod.list_cache_files(cache, "openclaw_memory_os")
    assert [Path(p).name for _, p in listed] == [
        "openclaw_memory_os-20260103.snapshot",
        "openclaw_memory_os-20260102.snapshot",
        "openclaw_memory_os-20260101.snapshot",
    ]


def test_list_cache_files_ignores_other_collections(tmp_path: Path):
    mod = _load_module()
    cache = tmp_path / "qcache"
    cache.mkdir()
    _make_snapshot(cache / "openclaw_memory_os-20260101.snapshot")
    _make_snapshot(cache / "user_memory-20260101.snapshot")
    listed = mod.list_cache_files(cache, "openclaw_memory_os")
    assert len(listed) == 1
    assert listed[0][1].endswith("openclaw_memory_os-20260101.snapshot")


def test_list_cache_files_skips_directories(tmp_path: Path):
    """A directory matching the pattern must not be treated as a file."""
    mod = _load_module()
    cache = tmp_path / "qcache"
    cache.mkdir()
    (cache / "openclaw_memory_os-fake.snapshot").mkdir()  # directory, not file
    _make_snapshot(cache / "openclaw_memory_os-real.snapshot")
    listed = mod.list_cache_files(cache, "openclaw_memory_os")
    assert len(listed) == 1
    assert listed[0][1].endswith("real.snapshot")


def test_list_cache_files_recurses_into_legacy_subdirectories(tmp_path: Path):
    mod = _load_module()
    cache = tmp_path / "qcache"
    legacy = cache / "legacy"
    legacy.mkdir(parents=True)
    _make_snapshot(legacy / "archive_memory-legacy.snapshot")
    listed = mod.list_cache_files(cache, "archive_memory")
    assert len(listed) == 1
    assert listed[0][1].endswith("archive_memory-legacy.snapshot")


def test_prune_cache_keeps_newest_n(tmp_path: Path):
    mod = _load_module()
    cache = tmp_path / "qcache"
    cache.mkdir()
    for i in range(8):
        _make_snapshot(cache / f"openclaw_memory_os-2026010{i}.snapshot", mtime_offset=float(i))

    summary = mod.prune_cache(cache, "openclaw_memory_os", keep=5)
    assert summary["total_seen"] == 8
    assert summary["kept"] == 5
    assert summary["deleted"] == 3
    assert len(summary["deleted_paths"]) == 3

    survivors = sorted(p.name for p in cache.iterdir())
    # The three OLDEST files (offsets 0,1,2) are gone; the rest survive.
    assert survivors == [
        "openclaw_memory_os-20260103.snapshot",
        "openclaw_memory_os-20260104.snapshot",
        "openclaw_memory_os-20260105.snapshot",
        "openclaw_memory_os-20260106.snapshot",
        "openclaw_memory_os-20260107.snapshot",
    ]


def test_prune_cache_handles_missing_dir(tmp_path: Path):
    mod = _load_module()
    summary = mod.prune_cache(tmp_path / "does_not_exist", "openclaw_memory_os", keep=5)
    assert summary["skipped"] is True
    assert summary["deleted"] == 0


def test_prune_cache_under_keep_does_nothing(tmp_path: Path):
    mod = _load_module()
    cache = tmp_path / "qcache"
    cache.mkdir()
    for i in range(3):
        _make_snapshot(cache / f"openclaw_memory_os-2026010{i}.snapshot", mtime_offset=float(i))
    summary = mod.prune_cache(cache, "openclaw_memory_os", keep=5)
    assert summary["total_seen"] == 3
    assert summary["kept"] == 3
    assert summary["deleted"] == 0
    assert len(list(cache.iterdir())) == 3


def test_prune_cache_keep_zero_deletes_everything(tmp_path: Path):
    """Defensive: keep=0 should drop all matching files."""
    mod = _load_module()
    cache = tmp_path / "qcache"
    cache.mkdir()
    for i in range(3):
        _make_snapshot(cache / f"openclaw_memory_os-2026010{i}.snapshot", mtime_offset=float(i))
    summary = mod.prune_cache(cache, "openclaw_memory_os", keep=0)
    assert summary["deleted"] == 3
    assert summary["kept"] == 0
    assert list(cache.iterdir()) == []


def test_format_cache_log_skipped():
    mod = _load_module()
    assert mod.format_cache_log({"skipped": True}) == "no cache dir; nothing to prune"


def test_format_cache_log_normal():
    mod = _load_module()
    msg = mod.format_cache_log({"deleted": 3, "kept": 5, "total_seen": 8})
    assert "pruned 3" in msg
    assert "kept 5/8" in msg


def test_cli_cache_prune_writes_summary(tmp_path: Path, capsys):
    mod = _load_module()
    cache = tmp_path / "qcache"
    cache.mkdir()
    for i in range(3):
        _make_snapshot(cache / f"openclaw_memory_os-2026010{i}.snapshot", mtime_offset=float(i))
    rc = mod.main(
        [
            "_prune_helpers.py",
            "cache-prune",
            str(cache),
            "openclaw_memory_os",
            "2",
        ]
    )
    assert rc == 0
    captured = capsys.readouterr()
    assert "pruned 1" in captured.out
    assert list(cache.iterdir())  # 2 files remain


def test_cli_rejects_bad_args(capsys):
    mod = _load_module()
    rc = mod.main(["_prune_helpers.py"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "usage" in err.lower()


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-v"]))
