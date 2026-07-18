"""Tests for scripts/_write_summary.py — the maintenance log parser.

The dashboard reads the maintenance summary JSON written by this script.
Tests build fake log content that mirrors the real ``maintenance.sh``
output to confirm the per-collection totals and the new
``chunks_scanned`` / ``ingested_new`` semantics are captured correctly.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def _load_write_summary():
    """Load scripts/_write_summary.py as a module (it has no package)."""
    spec = importlib.util.spec_from_file_location(
        "write_summary", SCRIPTS_DIR / "_write_summary.py"
    )
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _make_log(tmp_path: Path, body: str):
    log = tmp_path / "maintenance.log"
    log.write_text(body, encoding="utf-8")
    out = tmp_path / "summary.json"
    return log, out


def test_writes_atomic_summary_with_chunks_scanned(tmp_path: Path):
    mod = _load_write_summary()
    log_body = (
        "[maintenance 2026-07-11T18:58:17Z] step 1/5: ingest new memories\n"
        "{\n"
        '  "checkpoint_id": "ingest_20260711_185817",\n'
        '  "total_chunks": 662,\n'
        '  "written": 0,\n'
        '  "failed": 0,\n'
        '  "status": "completed"\n'
        "}\n"
        "[maintenance 2026-07-11T18:58:17Z] step 4/5: expire old working-tier\n"
        "[expire] candidates: 3\n"
        "[maintenance 2026-07-11T18:58:17Z] step 5/5: snapshot\n"
        "[snapshot] snapshot name: openclaw_memory_os-foo.snapshot\n"
        '[snapshot] {"name":"openclaw_memory_os-foo.snapshot","size":12345}\n'
        "[maintenance 2026-07-11T18:58:18Z] ok\n"
    )
    log, out = _make_log(tmp_path, log_body)
    rc = mod.main(["_write_summary.py", str(log), str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    # The dashboard 'memory file ingest' card reads these fields.
    assert data["ingested_new"] == 0
    assert data["chunks_scanned"] == 662
    # Backwards compat alias (legacy) also stays in sync.
    assert data["ingested_total"] == 662
    assert data["expired_count"] == 3
    assert "openclaw_memory_os-foo.snapshot" in (data["snapshot_name"] or "")
    assert data["snapshot_size_bytes"] == 12345


def test_empty_log_returns_zero_summary(tmp_path: Path):
    mod = _load_write_summary()
    log, out = _make_log(tmp_path, "")
    rc = mod.main(["_write_summary.py", str(log), str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["chunks_scanned"] == 0
    assert data["ingested_new"] == 0
    assert data["ingested_total"] == 0
    assert data["expired_count"] == 0
    assert data["superseded_links"] == 0
    assert data["snapshot_name"] is None


def test_multi_collection_summary_has_totals_and_per_collection_details(tmp_path: Path):
    mod = _load_write_summary()
    log_body = "\n".join([
        "[maintenance 2026-07-12T23:45:01Z] starting; collections: openclaw_memory_os user_memory archive_memory",
        "[maintenance 2026-07-12T23:45:01Z] --- [1/3] collection=openclaw_memory_os ---",
        "[maintenance 2026-07-12T23:45:01Z]   step 1/5: ingest all memory files",
        "{",
        '  "total_chunks": 247,',
        '  "written": 1,',
        '  "status": "completed"',
        "}",
        "[tier] loaded 249 points",
        "[supersede] applied 1 supersede links",
        "[expire] candidates: 0",
        "[snapshot] snapshot name: openclaw_memory_os-foo.snapshot",
        '{"result":{"name":"openclaw_memory_os-foo.snapshot","size":100},"status":"ok"}',
        "[snapshot] ok: /backup/openclaw_memory_os.tar.zst",
        "[maintenance 2026-07-12T23:45:01Z] --- [2/3] collection=user_memory ---",
        "[maintenance 2026-07-12T23:45:01Z]   step 1/5: ingest (skipped; collection uses external ingest)",
        "[tier] loaded 25431 points",
        "[supersede] applied 23 supersede links",
        "[expire] candidates: 2",
        "[snapshot] snapshot name: user_memory-bar.snapshot",
        '{"result":{"name":"user_memory-bar.snapshot","size":200},"status":"ok"}',
        "[snapshot] ok: /backup/user_memory.tar.zst",
        "[maintenance 2026-07-12T23:45:01Z] step 6/6: writing summary → /tmp/summary.json",
        "summary written: /tmp/summary.json",
        "[maintenance 2026-07-12T23:45:01Z] ok",
    ])
    log, out = _make_log(tmp_path, log_body)
    rc = mod.main(["_write_summary.py", str(log), str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["chunks_scanned"] == 247
    assert data["ingested_new"] == 1
    assert data["superseded_links"] == 24
    assert data["expired_count"] == 2
    assert data["totals"]["points_scanned"] == 25680
    assert data["totals"]["snapshots_ok"] == 2
    assert data["collections"]["user_memory"]["ingest_skipped"] is True
    assert data["collections"]["user_memory"]["superseded_links"] == 23
    assert data["snapshot_name"] == "user_memory-bar.snapshot"


def test_missing_log_returns_zero_summary_without_throwing(tmp_path: Path):
    mod = _load_write_summary()
    log = tmp_path / "does_not_exist.log"
    out = tmp_path / "summary.json"
    rc = mod.main(["_write_summary.py", str(log), str(out)])
    assert rc == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["chunks_scanned"] == 0
    assert data["ingested_new"] == 0


def test_main_rejects_wrong_argc(capsys):
    mod = _load_write_summary()
    # Wrong length (not 2 or 3) → usage + exit 2.
    rc = mod.main(["_write_summary.py"])
    assert rc == 2
    err = capsys.readouterr().err
    assert "usage" in err.lower()
