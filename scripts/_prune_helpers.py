#!/usr/bin/env python3
"""Prune helpers for the Memory OS maintenance scripts.

The Qdrant backup cache (``/opt/qdrant/backup`` by default) accumulates
``.snapshot`` files that have already been archived by
``backup_snapshot.sh``. This module provides a small, testable
function that keeps the most-recent N files for a given collection
and deletes the rest. The logic is intentionally narrow:

  * Only files matching ``{collection}-*.snapshot`` under the cache tree are candidates.
  * We sort by mtime (newest first) and keep the first ``keep`` files.
  * We never touch anything else in the cache directory.
  * Errors are logged to stderr and never raised back to the caller;
    the caller is the backup script, and we'd rather finish a backup
    than abort it because the cache couldn't be pruned.

The function is exercised by ``tests/test_backup_snapshot_cache.py``;
that test must be safe to run as any user (no real Qdrant, no root
needed) and only writes inside a tmp directory.
"""
from __future__ import annotations

import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple


def list_cache_files(
    cache_dir: str | os.PathLike,
    collection: str,
) -> List[Tuple[float, str]]:
    """Return ``[(mtime, path), ...]`` for matching snapshot files.

    Newest first. Files are filtered to ``{collection}-*.snapshot`` and
    must be regular files. Missing cache directory is treated as
    empty; non-existent collection prefix yields an empty list.
    """
    cache = Path(cache_dir)
    if not cache.is_dir():
        return []
    out: List[Tuple[float, str]] = []
    pattern = f"{collection}-*.snapshot"
    for path in cache.rglob(pattern):
        if not path.is_file():
            continue
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        out.append((mtime, str(path)))
    out.sort(key=lambda pair: pair[0], reverse=True)
    return out


def prune_cache(
    cache_dir: str | os.PathLike,
    collection: str,
    keep: int = 5,
) -> dict:
    """Delete old cache files for ``collection``, keeping the newest ``keep``.

    Returns a summary dict with the following keys (handy for both the
    bash script and the tests):

      * ``kept``: int, files that survived this run.
      * ``deleted``: int, files removed by this run.
      * ``total_seen``: int, files matching the prefix before pruning.
      * ``deleted_paths``: list[str], absolute paths that were deleted.
      * ``skipped``: bool, True if the cache directory didn't exist.

    The function is best-effort: an OSError on a single file is logged
    but does not stop the loop. We never raise.
    """
    summary = {
        "kept": 0,
        "deleted": 0,
        "total_seen": 0,
        "deleted_paths": [],
        "skipped": False,
    }
    cache = Path(cache_dir)
    if not cache.is_dir():
        summary["skipped"] = True
        return summary
    pairs = list_cache_files(cache, collection)
    summary["total_seen"] = len(pairs)
    summary["kept"] = min(len(pairs), max(keep, 0))
    for _mtime, path in pairs[max(keep, 0):]:
        try:
            os.unlink(path)
        except OSError as exc:
            print(
                f"[snapshot] cache prune error for {path}: {exc}",
                file=sys.stderr,
            )
            continue
        summary["deleted"] += 1
        summary["deleted_paths"].append(path)
    return summary


def format_cache_log(summary: dict) -> str:
    """Render a one-line summary suitable for the backup script log."""
    if summary.get("skipped"):
        return "no cache dir; nothing to prune"
    return (
        f"pruned {summary['deleted']} cache file(s); "
        f"kept {summary['kept']}/{summary['total_seen']}"
    )


def main(argv: List[str]) -> int:
    """CLI entry point for shell integration.

    Usage::

        python _prune_helpers.py cache-prune CACHE_DIR COLLECTION [KEEP]

    Writes a one-line summary to stdout and returns 0 on success.
    """
    if len(argv) < 4 or argv[1] != "cache-prune":
        sys.stderr.write(
            "usage: _prune_helpers.py cache-prune CACHE_DIR COLLECTION [KEEP]\n"
        )
        return 2
    cache_dir = argv[2]
    collection = argv[3]
    keep = int(argv[4]) if len(argv) > 4 else 5
    summary = prune_cache(cache_dir, collection, keep=keep)
    ts = datetime.now(timezone.utc).isoformat()
    print(f"[snapshot] {ts} {format_cache_log(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
