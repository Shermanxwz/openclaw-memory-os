#!/usr/bin/env python3
"""
Memory Brain — Unified ingest + consolidate pipeline

合并了原 memory_brain_ingest.py 和 memory_brain_consolidate.py
调用顺序: 先 ingest（摄取新文件）→ 再 consolidate（整合话题）

用法: python memory_brain.py
环境变量: 参考原两个脚本 (MEMORY_DIR, QDRANT_URL, COLLECTION, LLM_PROVIDER, EMBED_PROVIDER 等)

退出码:
  0 — 两个 phase 都成功
  1 — ingest 失败但 consolidate 已尝试
  2 — consolidate 失败
  3 — 两个都失败

Wave 2 (2026-07-21): when invoked from ``scripts/maintenance.sh`` the
shared ``MAINTENANCE_RUN_ID`` env var is read here so the sub-step
status markers printed to stdout can be correlated by
``scripts/_write_summary.py``. The unified pipeline no longer writes
any ``/var/log/openclaw-memory-brain-*.json`` files — those legacy
status paths are gated on ``MAINTENANCE_RUN_ID`` in the leaf modules,
and this wrapper just prints a top-level marker the summary parser
can pick up.
"""

import os
import sys
import importlib.util
from pathlib import Path

# Shared run identifier propagated by ``scripts/maintenance.sh``. When
# unset (e.g. an operator runs ``memory_brain.py`` directly outside
# cron), we fall back to the empty string so the leaf modules still
# behave as a manual invocation (legacy /var/log writes still happen
# for compatibility).
RUN_ID = os.environ.get("MAINTENANCE_RUN_ID", "")

SCRIPT_DIR = Path(__file__).resolve().parent

def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, SCRIPT_DIR / filename)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _emit_substatus(*, ingest_rc: int, consolidate_rc: int) -> None:
    """Print a structured sub-step marker for the unified pipeline.

    The marker is captured by ``maintenance.sh`` via the
    ``>> "$LOG_FILE" 2>&1`` redirection, and ``_write_summary.py`` parses
    it to populate the ``steps.memory_brain`` block of
    ``maintenance-summary.json``. We deliberately do **not** write any
    ``/var/log/openclaw-memory-brain-*.json`` file here — that path is
    reserved for the leaf modules when invoked manually (no
    ``MAINTENANCE_RUN_ID``).
    """
    if ingest_rc == 0 and consolidate_rc == 0:
        status = "ok"
    elif ingest_rc == 0:
        status = "ok"
    elif consolidate_rc == 0:
        status = "degraded"
    else:
        status = "failed"
    run_id = RUN_ID or "manual"
    print(
        f"[brain-pipeline] run_id={run_id} status={status} "
        f"ingest_exit={ingest_rc} consolidate_exit={consolidate_rc}"
    )

def main():
    # When the maintenance.sh caller has set MAINTENANCE_RUN_ID we
    # forward it to the leaf modules via env so the consolidated
    # ``[brain-ingest]`` / ``brain-substep]`` / ``[brain-consolidate]``
    # markers carry the same run_id and can be correlated by the
    # summary parser.
    if RUN_ID:
        os.environ["MAINTENANCE_RUN_ID"] = RUN_ID
    print("=" * 60)
    print("Memory Brain — unified pipeline (ingest + consolidate)")
    print(f"  run_id: {RUN_ID or '(manual)'}")
    print("=" * 60)
    print()

    ingest_rc = 0
    consolidate_rc = 0

    # Phase 1: ingest — bracket the real call with monotonic UTC
    # timestamps so the canonical summary records independent
    # started/finished/duration for the ingest substep (the legacy
    # contract used the same [brain-step] timestamp for both ingest and
    # consolidate, which made the two cards indistinguishable).
    print("▶ Phase 1/2: ingest")
    print("-" * 60)
    import datetime as _dt
    ingest_started_at = _dt.datetime.now(_dt.timezone.utc)
    try:
        ingest_mod = load_module("memory_brain_ingest", "memory_brain_ingest.py")
        ingest_mod.main()
        print("\n✓ ingest: ok")
    except SystemExit as e:
        ingest_rc = e.code if isinstance(e.code, int) else 1
        if ingest_rc == 0:
            print("\n✓ ingest: ok (system exit 0)")
        else:
            print(f"\n✗ ingest: failed (exit {ingest_rc})")
    except Exception as e:
        ingest_rc = 1
        print(f"\n✗ ingest: exception: {e}")
        import traceback
        traceback.print_exc()
    ingest_finished_at = _dt.datetime.now(_dt.timezone.utc)
    print(
        f"[brain-substep] run_id={RUN_ID or 'manual'} name=ingest "
        f"started={ingest_started_at.isoformat().replace('+00:00', 'Z')} "
        f"finished={ingest_finished_at.isoformat().replace('+00:00', 'Z')} "
        f"exit={ingest_rc}"
    )

    print()
    print("▶ Phase 2/2: consolidate")
    print("-" * 60)
    consolidate_started_at = _dt.datetime.now(_dt.timezone.utc)
    try:
        consolidate_mod = load_module("memory_brain_consolidate", "memory_brain_consolidate.py")
        consolidate_mod.main()
        print("\n✓ consolidate: ok")
    except SystemExit as e:
        consolidate_rc = e.code if isinstance(e.code, int) else 1
        if consolidate_rc == 0:
            print("\n✓ consolidate: ok (system exit 0)")
        else:
            print(f"\n✗ consolidate: failed (exit {consolidate_rc})")
    except Exception as e:
        consolidate_rc = 1
        print(f"\n✗ consolidate: exception: {e}")
        import traceback
        traceback.print_exc()
    consolidate_finished_at = _dt.datetime.now(_dt.timezone.utc)
    print(
        f"[brain-substep] run_id={RUN_ID or 'manual'} name=consolidate "
        f"started={consolidate_started_at.isoformat().replace('+00:00', 'Z')} "
        f"finished={consolidate_finished_at.isoformat().replace('+00:00', 'Z')} "
        f"exit={consolidate_rc}"
    )

    # Emit a single consolidated sub-step status marker for the
    # summary writer. This is the only top-level status output the
    # unified pipeline produces — the legacy /var/log writes were
    # dropped (see ``memory_brain_ingest`` / ``memory_brain_consolidate``
    # for the leaf-module equivalents).
    _emit_substatus(ingest_rc=ingest_rc, consolidate_rc=consolidate_rc)

    print()
    print("=" * 60)
    if ingest_rc == 0 and consolidate_rc == 0:
        print("Memory Brain pipeline: ok")
        return 0
    elif ingest_rc == 0:
        print(f"Memory Brain pipeline: ingest ok, consolidate failed (rc={consolidate_rc})")
        return 2
    elif consolidate_rc == 0:
        print(f"Memory Brain pipeline: ingest failed (rc={ingest_rc}), consolidate ok")
        return 1
    else:
        print(f"Memory Brain pipeline: both phases failed (ingest={ingest_rc}, consolidate={consolidate_rc})")
        return 3

if __name__ == "__main__":
    sys.exit(main())
