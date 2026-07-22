#!/usr/bin/env python3
"""v0.3.0: Refresh the BM25 lexical index after a maintenance run.

This script is called by ``maintenance.sh`` after Qdrant snapshots
are complete. It re-reads all memories from the backend and
incrementally refreshes the lexical index cache.

If the lexical index is not initialised (no BM25Index instance
is running in this process), the script is a no-op — the index
will be lazily built on the first hybrid query. Running this on
a cold cache still does useful work: it seeds the index so the
first user query does not block on a synchronous BM25 build.

B2-3 fix: the script now respects per-collection identity so the
refresh does not silently merge every collection under the
literal collection name ``"memory"``. It supports three resolution
modes for the target collection:

1. ``--collection NAME`` CLI flag (highest priority).
2. ``QDRANT_COLLECTION`` env var, which ``maintenance.sh``
   exports for each collection it iterates over.
3. The repo's configured primary collection
   (``openclaw_memories`` by default).
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Point to the project venv
PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR))

try:
    from openclaw_memory_os.backends import QdrantBackend
    from openclaw_memory_os.lexical import BM25Index, incremental_refresh
    from openclaw_memory_os.contracts import (
        CandidateStatus, CandidateTier, MemoryRecord,
    )

    def _memory_to_record(memory, collection):
        return MemoryRecord(
            collection=collection,
            memory_id=str(memory.id),
            candidate_key=f"{collection}:{memory.id}",
            text=memory.text or "",
            summary=memory.summary,
            source=memory.source,
            tags=list(memory.tags or []),
            status=CandidateStatus(memory.status.value),
            tier=CandidateTier(memory.tier.value),
            importance=float(memory.importance or 0.0),
            created_at=memory.created_at,
            updated_at=memory.updated_at,
        )

    # Parse CLI args before reading env so the explicit flag wins.
    parser = argparse.ArgumentParser(
        description="Refresh the BM25 lexical index from the configured backend."
    )
    parser.add_argument(
        "--collection",
        default=None,
        help=(
            "Target Qdrant collection to index. Falls back to the "
            "QDRANT_COLLECTION env var (which maintenance.sh exports), "
            "then to the configured primary collection."
        ),
    )
    args, _unknown = parser.parse_known_args()

    # Warm up: load collection metadata from env / CLI / config
    qdrant_url = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
    # B2-3: resolve the target collection from CLI > env > default.
    # ``maintenance.sh`` exports ``QDRANT_COLLECTION`` per-collection
    # before invoking this script, so we honour that without forcing
    # operators to change the cron config.
    default_primary = os.environ.get("MEMORY_OS_PRIMARY_COLLECTION", "openclaw_memories")
    qdrant_collection = (
        args.collection
        or os.environ.get("QDRANT_COLLECTION")
        or default_primary
    )
    secondary = os.environ.get("QDRANT_SECONDARY_COLLECTIONS", "")
    secondary_list = [c.strip() for c in secondary.split(",") if c.strip()]

    # Build a fresh index from the backend's full corpus. This script
    # is called after maintenance snapshots and iterates every memory;
    # loading the previous cache and replacing each document turns the
    # run into an expensive remove+add cycle over posting lists. A fresh
    # build is both simpler and faster, while preserving the same saved
    # cache contract for request-time loading.
    state_dir = Path(
        os.environ.get(
            "XDG_STATE_HOME",
            os.path.expanduser("~/.local/state"),
        )
    ) / "openclaw-memory-os"
    cache_dir = state_dir / "lexical-index"
    index = BM25Index()

    # Iterate over memories and rebuild. B2-3: each memory
    # records its actual Qdrant collection, not the hardcoded
    # ``"memory"`` literal that the previous version stamped on
    # every record. We use the new ``iter_memories_by_collection``
    # helper so per-collection identity is preserved end-to-end.
    backend = QdrantBackend(
        qdrant_url, qdrant_collection, secondary_collections=secondary_list
    )
    records = [
        _memory_to_record(memory, coll)
        for coll, memory in backend.iter_memories_by_collection()
    ]
    added = incremental_refresh(index, records)
    index.save(cache_dir)
    print(
        f"refresh_lexical: collection={qdrant_collection} "
        f"added/refreshed {added} documents"
    )
except ImportError as exc:
    print(f"refresh_lexical: skipped ({exc})")
except Exception as exc:
    import traceback
    traceback.print_exc()
    print(f"refresh_lexical: failed ({exc})")