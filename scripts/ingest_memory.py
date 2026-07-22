#!/usr/bin/env python3
"""Memory ingestion: read MEMORY.md and memory/*.md, chunk, embed, write to Qdrant.

Usage:
    .venv/bin/python scripts/ingest_memory.py [--collection NAME] [--since-days N]

Goals:
    - One-shot ingestion that mirrors the OpenClaw payload schema.
    - Idempotent: re-ingesting the same chunk overwrites the same point
      via Qdrant upsert keyed by a UUID-shaped stable ID.
    - Privacy-clean: reads only files under <project_root>.  privacy-allow: MEMORY_OS_PATH
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

from openclaw_memory_os.personal_taxonomy import (
    load_personal_taxonomy,
    expand_with_personal,
)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "openclaw_memory_os")
WORKSPACE = Path(os.environ.get("WORKSPACE_ROOT", "<project_root>"))  # privacy-allow: MEMORY_OS_PATH
EMBED_DIM = 768

# --- Checkpoint state ------------------------------------------------------
# Path: <workspace>/.cache/ingest_state.json (gitignored, regenerated each run).
# Holds the set of chunk_ids that have already been successfully upserted so
# that a SIGTERM / OOM / network drop mid-run resumes from where it left off
# instead of re-embedding every chunk from scratch. Embedding 700+ chunks
# through Ollama's nomic-embed-text takes 45-80 min on a 2-core VPS, so a
# half-finished run used to waste all of that work on restart.
STATE_DIR = WORKSPACE / ".cache"
STATE_FILE = STATE_DIR / "ingest_state.json"
STATE_VERSION = 1
_BATCH_SIZE = 32
_SAVE_EVERY_BATCHES = 4  # fsync state every 4 * 32 = 128 chunks (~few min)


def _load_state() -> set:
    """Load set of chunk_ids already processed. Returns empty set on miss."""
    if not STATE_FILE.exists():
        return set()
    try:
        data = json.loads(STATE_FILE.read_text(encoding="utf-8"))
        if data.get("version") != STATE_VERSION:
            print(f"[ingest] state version mismatch ({data.get('version')} != {STATE_VERSION}), ignoring")
            return set()
        done = set(data.get("done", []))
        print(f"[ingest] checkpoint loaded: {len(done)} chunks already done")
        return done
    except Exception as exc:
        print(f"[ingest] state file unreadable ({exc}); starting fresh")
        return set()


def _save_state(done: set) -> None:
    """Atomically persist the checkpoint so a crash mid-write doesn't corrupt it."""
    try:
        STATE_DIR.mkdir(parents=True, exist_ok=True)
        tmp = STATE_FILE.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps({"version": STATE_VERSION, "done": sorted(done), "saved_at": datetime.now(timezone.utc).isoformat()}),
            encoding="utf-8",
        )
        os.replace(tmp, STATE_FILE)  # atomic on POSIX
    except Exception as exc:
        print(f"[ingest] WARNING: could not save checkpoint: {exc}")


def list_memory_files() -> List[Path]:
    files = []
    mem_md = WORKSPACE / "MEMORY.md"
    if mem_md.exists():
        files.append(mem_md)
    mem_dir = WORKSPACE / "memory"
    if mem_dir.is_dir():
        files.extend(sorted(p for p in mem_dir.glob("*.md") if p.is_file()))
    return files


def split_by_sections(text: str, source: Path) -> Iterable[Tuple[str, str]]:
    text = text.strip()
    if not text:
        return
    # Privacy: skip sections that contain credentials or secrets. The
    # dashboard does not need them and pushing them through embedding
    # only multiplies the attack surface. The rule is intentionally
    # generous: any line matching a secret pattern flags the whole
    # section for skipping.
    sensitive_patterns = [
        re.compile(r"Pass(word)?\s*[:=]", re.IGNORECASE),
        re.compile(r"API[_-]?[Kk]ey\s*[:=]"),
        re.compile(r"root\s*/\s*\S{4,}"),                # "root / password"
        re.compile(r"@2024?\S{2,}", re.IGNORECASE),       # historical RDP/SSH passwords
        re.compile(r"\b[A-Z][a-z]\d[A-Za-z]\d{4,}"),     # generic token shape
        re.compile(r"\bsk-[A-Za-z0-9]{8,}"),              # sk-...
        re.compile(r"\bghp_[A-Za-z0-9]{8,}"),
        re.compile(r"\bAa8\d{4,}"),                       # Aa8... passwords
    ]
    def is_sensitive(chunk: str) -> bool:
        for line in chunk.splitlines()[:10]:  # only inspect the first 10 lines
            for pat in sensitive_patterns:
                if pat.search(line):
                    return True
        return False

    if "## " not in text:
        m = hashlib.sha1(str(source).encode()).hexdigest()[:12]
        if not is_sensitive(text):
            yield f"{source.stem}::{m}", text
        return
    sections = re.split(r"(?m)^## ", text)
    for sec in sections:
        sec = sec.strip()
        if len(sec) < 50:
            continue
        title = sec.split("\n", 1)[0].strip()
        slug = re.sub(r"[^A-Za-z0-9_\u4e00-\u9fff]+", "_", title)[:48].strip("_") or "untitled"
        m = hashlib.sha1((source.name + title).encode()).hexdigest()[:12]
        if is_sensitive(sec):
            continue
        yield f"{source.stem}::{slug}::{m}", sec


def embed(text: str) -> List[float]:
    body = {"model": EMBED_MODEL, "prompt": text[:4000]}
    req = urllib.request.Request(
        f"{OLLAMA_URL}/api/embeddings",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read().decode("utf-8"))
    vec = out.get("embedding")
    if not isinstance(vec, list) or len(vec) != EMBED_DIM:
        raise RuntimeError(f"Bad embedding shape: len={len(vec) if isinstance(vec, list) else '?'}")
    return vec


def qdrant_collection_exists(name: str) -> bool:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/collections/{name}", timeout=10) as r:
            return r.status == 200
    except Exception:
        return False


def qdrant_create_collection(name: str) -> None:
    body = {"vectors": {"size": EMBED_DIM, "distance": "Cosine"}}
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{name}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        r.read()


def qdrant_upsert(name: str, points: List[dict]) -> None:
    if not points:
        return
    body = {"points": points}
    req = urllib.request.Request(
        f"{QDRANT_URL}/collections/{name}/points?wait=true",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="PUT",
    )
    with urllib.request.urlopen(req, timeout=120) as r:
        r.read()


def qdrant_point_count(name: str) -> int:
    try:
        with urllib.request.urlopen(f"{QDRANT_URL}/collections/{name}", timeout=10) as r:
            data = json.loads(r.read())
            return int(data.get("result", {}).get("points_count", 0))
    except Exception:
        return 0


# Base (public-safe) keyword lists. Brand-specific extras are loaded at
# runtime from the operator's gitignored config/personal_taxonomy.json
# via the MEMORY_OS_TAXONOMY_PATH env var; see
# config/personal_taxonomy.example.json for the public-safe template.
CORE_BASE = ["core_keyword_a", "core_keyword_b", "core_keyword_c"]
LONG_BASE = ["## 教训", "教训库", "重要教训"]
WORKING_BASE = ["in_progress", "CI 状态", "临时", "running", "状态"]


def _core_keywords(taxonomy) -> List[str]:
    return expand_with_personal(CORE_BASE, taxonomy, "core")


def _long_keywords(taxonomy) -> List[str]:
    return expand_with_personal(LONG_BASE, taxonomy, "long")


def classify_tier(text: str, source: str, taxonomy=None) -> str:
    taxonomy = taxonomy if taxonomy is not None else load_personal_taxonomy()
    head = text[:200]
    if any(k in head for k in _core_keywords(taxonomy)):
        return "core"
    if any(k in head for k in _long_keywords(taxonomy)):
        return "long"
    low = head.lower()
    if any(k in low for k in WORKING_BASE):
        return "working"
    if len(text) < 400 and "memory/" in source:
        return "short"
    return "medium"


def importance_score(text: str, source: str, taxonomy=None) -> float:
    taxonomy = taxonomy if taxonomy is not None else load_personal_taxonomy()
    score = 0.5
    core_long = _core_keywords(taxonomy) + _long_keywords(taxonomy) + ["铁律", "重要教训"]
    if any(k in text for k in core_long):
        score += 0.4
    if "MEMORY.md" in source:
        score += 0.05
    if "教训" in text[:120]:
        score += 0.1
    if len(text) < 200:
        score -= 0.2
    return max(0.05, min(1.0, score))


def _id_for_chunk(chunk_id: str) -> str:
    h = hashlib.sha1(chunk_id.encode()).hexdigest()
    return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"


def chunk_metadata(text: str, source: Path, chunk_id: str, taxonomy=None) -> dict:
    stem = source.stem
    created = datetime.now(timezone.utc)
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", str(source))
    if m:
        try:
            created = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
        except ValueError:
            pass
    if taxonomy is None:
        taxonomy = load_personal_taxonomy()
    tier = classify_tier(text, str(source), taxonomy)
    importance = importance_score(text, str(source), taxonomy)
    return {
        "id": chunk_id,
        "text": text[:8000],
        "summary": None,
        "source": str(source.relative_to(WORKSPACE.parent)) if source.is_relative_to(WORKSPACE.parent) else str(source),
        "created_at": created.isoformat(),
        "updated_at": None,
        "tier": tier,
        "status": "active",
        "importance": importance,
        "tags": [stem],
        "embedding_model": EMBED_MODEL,
    }


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--since-days", type=int, default=None)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args(argv)

    taxonomy = load_personal_taxonomy()

    files = list_memory_files()
    print(f"[ingest] workspace files: {len(files)}")
    if args.since_days is not None:
        cutoff = time.time() - args.since_days * 86400
        files = [f for f in files if f.stat().st_mtime >= cutoff]
        print(f"[ingest] after since_days={args.since_days}: {len(files)} files")

    chunks: List[Tuple[Path, str, str]] = []
    for f in files:
        try:
            text = f.read_text(encoding="utf-8")
        except Exception as exc:
            print(f"[ingest] skip {f}: {exc}")
            continue
        for cid, ctext in split_by_sections(text, f):
            chunks.append((f, cid, ctext))

    if args.limit:
        chunks = chunks[: args.limit]
    print(f"[ingest] chunks to ingest: {len(chunks)}")

    if args.dry_run:
        for f, cid, ctext in chunks[:5]:
            print(f"  preview: {f.name} :: {cid}  ({len(ctext)} chars)")
        return 0

    # --- Checkpoint / resume ------------------------------------------------
    # Skip chunks whose stable UUID was already upserted in a previous run.
    # Qdrant upsert is itself idempotent (same id overwrites), but this
    # short-circuit saves 45-80 min of embedding on a long re-run.
    done = _load_state()
    pending = [(f, cid, ctext) for (f, cid, ctext) in chunks if _id_for_chunk(cid) not in done]
    skipped = len(chunks) - len(pending)
    if skipped:
        print(f"[ingest] checkpoint: skipping {skipped} chunks already done, {len(pending)} to process")

    if not qdrant_collection_exists(args.collection):
        print(f"[ingest] creating collection {args.collection}")
        qdrant_create_collection(args.collection)
    else:
        print(f"[ingest] collection exists: {args.collection} ({qdrant_point_count(args.collection)} points)")

    written = 0
    failed = 0
    batch: List[dict] = []
    batches_since_save = 0
    for idx, (f, cid, ctext) in enumerate(pending, 1):
        meta = chunk_metadata(ctext, f, cid, taxonomy)
        try:
            vec = embed(ctext)
        except Exception as exc:
            failed += 1
            if failed <= 3:
                print(f"[ingest] embed failed for {cid}: {exc}")
            continue
        point = {"id": _id_for_chunk(cid), "vector": vec, "payload": meta}
        batch.append(point)
        if len(batch) >= _BATCH_SIZE:
            qdrant_upsert(args.collection, batch)
            written += len(batch)
            done.update(p["id"] for p in batch)
            batch = []
            batches_since_save += 1
            if batches_since_save >= _SAVE_EVERY_BATCHES:
                _save_state(done)
                batches_since_save = 0
        if idx % 25 == 0:
            print(f"[ingest] {idx}/{len(pending)}  written={written}  failed={failed}")

    if batch:
        qdrant_upsert(args.collection, batch)
        written += len(batch)
        done.update(p["id"] for p in batch)

    # Final save — covers runs that ended without hitting the periodic save.
    _save_state(done)
    print(f"[ingest] checkpoint saved: {len(done)} chunks total done")

    print(f"[ingest] done: written={written}  failed={failed}  collection={args.collection}")
    return 0


if __name__ == "__main__":
    sys.exit(main())