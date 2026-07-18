"""Robust memory ingestion with checkpoint/resume, skip-existing, and progress tracking.

Features over the basic :mod:`scripts.ingest_memory`:

    * **Checkpoint / resume** — a JSON progress file in XDG state dir lets
      interrupted runs pick up where they left off.
    * **Skip existing** — checks Qdrant point existence before embedding.
    * **Progress state** — :class:`IngestProgress` model with per-chunk
      tracking.
    * **Longer timeout** — 300-second Ollama embed timeout for large or slow
      models.
    * **Safe signal handling** — catches SIGINT/SIGTERM to save a checkpoint
      before exiting.

Usage::

    from openclaw_memory_os.ingestion import IngestionManager
    manager = IngestionManager()
    result = manager.run_ingestion(collection="my_collection")
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import signal
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, List, Optional, Set, Tuple

from .audit import get_audit_store
from .models import IngestProgress, utcnow
from .personal_taxonomy import (
    load_personal_taxonomy,
    expand_with_personal,
)

logger = logging.getLogger(__name__)

OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "nomic-embed-text")
QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
EMBED_DIM = 768
INGEST_TIMEOUT = int(os.environ.get("INGEST_EMBED_TIMEOUT", "300"))

# Signal-safe flag
_interrupted = False


def _signal_handler(signum, frame):
    global _interrupted
    if _interrupted:
        logger.warning("Second interrupt received; exiting immediately.")
        sys.exit(1)
    _interrupted = True
    logger.warning("Interrupt received (signal %d). Will checkpoint and exit.", signum)


def _install_signal_handlers():
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)


def _state_dir() -> Path:
    data_home = Path(os.environ.get("XDG_STATE_HOME", os.path.expanduser("~/.local/state")))
    return data_home / "openclaw-memory-os"


class IngestionManager:
    """Orchestrates memory file ingestion with checkpoint/resume."""

    def __init__(self, workspace_root: Optional[Path] = None):
        self.workspace = workspace_root or Path(
            os.environ.get("WORKSPACE_ROOT", Path(__file__).resolve().parents[2])
        )
        self.state_dir = _state_dir()
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self._checkpoint_path = self.state_dir / "ingest_checkpoint.json"
        self._progress: Optional[IngestProgress] = None
        self._known_point_ids: Set[str] = set()

        _install_signal_handlers()

    # ------------------------------------------------------------------
    # File discovery
    # ------------------------------------------------------------------

    def _list_memory_files(self) -> List[Path]:
        files: List[Path] = []
        mem_md = self.workspace / "MEMORY.md"
        if mem_md.exists():
            files.append(mem_md)
        mem_dir = self.workspace / "memory"
        if mem_dir.is_dir():
            files.extend(sorted(p for p in mem_dir.glob("*.md") if p.is_file()))
        return files

    def _read_file(self, path: Path) -> str:
        return path.read_text(encoding="utf-8")

    # ------------------------------------------------------------------
    # Chunking: shared with scripts/ingest_memory.py
    # ------------------------------------------------------------------

    @staticmethod
    def _split_by_sections(text: str, source: Path) -> Iterable[Tuple[str, str]]:
        text = text.strip()
        if not text:
            return
        sensitive_patterns = [
            re.compile(r"Pass(word)?\s*[:=]", re.IGNORECASE),
            re.compile(r"API[_-]?[Kk]ey\s*[:=]"),
            re.compile(r"root\s*/\s*\S{4,}"),
            re.compile(r"@2024?\S{2,}", re.IGNORECASE),
            re.compile(r"\b[A-Z][a-z]\d[A-Za-z]\d{4,}"),
            re.compile(r"\bsk-[A-Za-z0-9]{8,}"),
            re.compile(r"\bghp_[A-Za-z0-9]{8,}"),
            re.compile(r"\bAa8\d{4,}"),
        ]

        def is_sensitive(chunk: str) -> bool:
            for line in chunk.splitlines()[:10]:
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

    # ------------------------------------------------------------------
    # Embedding
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[List[float]]:
        body = {"model": EMBED_MODEL, "prompt": text[:4000]}
        req = urllib.request.Request(
            f"{OLLAMA_URL}/api/embeddings",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=INGEST_TIMEOUT) as r:
                out = json.loads(r.read().decode("utf-8"))
            vec = out.get("embedding")
            if not isinstance(vec, list) or len(vec) != EMBED_DIM:
                logger.error("Bad embedding shape: len=%s", len(vec) if isinstance(vec, list) else "?")
                return None
            return vec
        except urllib.error.HTTPError as e:
            logger.error("Ollama HTTP %d for embed: %s", e.code, e.reason)
            return None
        except urllib.error.URLError as e:
            logger.error("Ollama unreachable: %s", e.reason)
            return None
        except Exception as e:
            logger.error("Embed error: %s", e)
            return None

    # ------------------------------------------------------------------
    # Qdrant helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _id_for_chunk(chunk_id: str) -> str:
        h = hashlib.sha1(chunk_id.encode()).hexdigest()
        return f"{h[0:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:32]}"

    def _fetch_existing_ids(self, collection: str) -> Set[str]:
        """Fetch all point IDs currently in the Qdrant collection."""
        existing: Set[str] = set()
        try:
            offset = None
            url = f"{QDRANT_URL}/collections/{collection}/points/scroll"
            while True:
                body = {"limit": 1024, "with_payload": False, "with_vector": False}
                if offset:
                    body["offset"] = offset
                req = urllib.request.Request(
                    url,
                    data=json.dumps(body).encode("utf-8"),
                    headers={"Content-Type": "application/json"},
                )
                with urllib.request.urlopen(req, timeout=60) as r:
                    data = json.loads(r.read().decode("utf-8"))
                result = data.get("result", {})
                points = result.get("points", [])
                for p in points:
                    existing.add(str(p["id"]))
                next_offset = result.get("next_page_offset")
                if next_offset is None:
                    break
                offset = next_offset
        except Exception as e:
            logger.warning("Could not fetch existing point IDs: %s", e)
        return existing

    @staticmethod
    def _qdrant_collection_exists(name: str) -> bool:
        try:
            with urllib.request.urlopen(f"{QDRANT_URL}/collections/{name}", timeout=10) as r:
                return r.status == 200
        except Exception:
            return False

    @staticmethod
    def _qdrant_create_collection(name: str) -> None:
        body = {"vectors": {"size": EMBED_DIM, "distance": "Cosine"}}
        req = urllib.request.Request(
            f"{QDRANT_URL}/collections/{name}",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=30) as r:
            r.read()

    @staticmethod
    def _qdrant_upsert(name: str, points: List[dict]) -> None:
        if not points:
            return
        body = {"points": points}
        req = urllib.request.Request(
            f"{QDRANT_URL}/collections/{name}/points?wait=true",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="PUT",
        )
        with urllib.request.urlopen(req, timeout=INGEST_TIMEOUT) as r:
            r.read()

    @staticmethod
    def _qdrant_point_count(name: str) -> int:
        try:
            with urllib.request.urlopen(f"{QDRANT_URL}/collections/{name}", timeout=10) as r:
                data = json.loads(r.read())
                return int(data.get("result", {}).get("points_count", 0))
        except Exception:
            return 0

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------

    # Base (public-safe) keyword lists. Brand-specific extras are loaded at
    # runtime from the operator's gitignored config/personal_taxonomy.json
    # via the MEMORY_OS_TAXONOMY_PATH env var; see
    # config/personal_taxonomy.example.json for the public-safe template.
    @staticmethod
    def _core_keywords(taxonomy) -> List[str]:
        return expand_with_personal(
            ["core_keyword_a", "core_keyword_b", "core_keyword_c"], taxonomy, "core"
        )

    @staticmethod
    def _long_keywords(taxonomy) -> List[str]:
        return expand_with_personal(
            ["## 教训", "教训库", "重要教训"], taxonomy, "long"
        )

    WORKING_BASE = ["in_progress", "ci 状态", "临时", "running", "状态"]

    @classmethod
    def _classify_tier(cls, text: str, source: str, taxonomy=None) -> str:
        taxonomy = taxonomy if taxonomy is not None else load_personal_taxonomy()
        head = text[:200]
        if any(k in head for k in cls._core_keywords(taxonomy)):
            return "core"
        if any(k in head for k in cls._long_keywords(taxonomy)):
            return "long"
        low = head.lower()
        if any(k in low for k in cls.WORKING_BASE):
            return "working"
        if len(text) < 400 and "memory/" in source:
            return "short"
        return "medium"

    @staticmethod
    def _importance_score(text: str, source: str, taxonomy=None) -> float:
        taxonomy = taxonomy if taxonomy is not None else load_personal_taxonomy()
        score = 0.5
        core_long = (
            IngestionManager._core_keywords(taxonomy)
            + IngestionManager._long_keywords(taxonomy)
            + ["铁律", "重要教训"]
        )
        if any(k in text for k in core_long):
            score += 0.4
        if "MEMORY.md" in source:
            score += 0.05
        if "教训" in text[:120]:
            score += 0.1
        if len(text) < 200:
            score -= 0.2
        return max(0.05, min(1.0, score))

    @staticmethod
    def _classify_type(text: str, source: str) -> str:
        """Infer memory type from content and source."""
        head = text[:300].lower()
        if any(k in head for k in ["规则", "rule", "铁律", "承诺", "promise"]):
            return "rule"
        if any(k in head for k in ["教训", "lesson", "learned"]):
            return "lesson"
        if any(k in head for k in ["上下文", "context", "背景"]):
            return "context"
        if any(k in head for k in ["配置", "config", "setting"]):
            return "config"
        if any(k in head for k in ["状态", "status", "in_progress"]):
            return "status"
        return "note"

    @staticmethod
    def _infer_topic(text: str, source: str) -> Optional[str]:
        """Simple topic inference from source file name or first heading."""
        stem = Path(source).stem if isinstance(source, (str, Path)) else str(source)
        if stem and stem.lower() not in ("memory", "memories", "readme", "index"):
            return stem
        lines = text.strip().split("\n", 5)
        for line in lines:
            if line.startswith("# "):
                return line.lstrip("#").strip()[:60]
        return None

    def _build_payload(self, text: str, source: Path, chunk_id: str, taxonomy=None) -> dict:
        stem = source.stem
        created = utcnow()
        m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", str(source))
        if m:
            try:
                created = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
            except ValueError:
                pass
        if taxonomy is None:
            taxonomy = load_personal_taxonomy()
        tier = self._classify_tier(text, str(source), taxonomy)
        importance = self._importance_score(text, str(source), taxonomy)
        mem_type = self._classify_type(text, str(source))
        topic = self._infer_topic(text, source)

        # Estimate line count from text
        line_count = text.count("\n") + 1

        return {
            "id": chunk_id,
            "text": text[:8000],
            "summary": None,
            "source": str(source.relative_to(self.workspace.parent))
            if source.is_relative_to(self.workspace.parent)
            else str(source),
            "created_at": created.isoformat(),
            "updated_at": None,
            "tier": tier,
            "status": "active",
            "importance": importance,
            "tags": [stem],
            "embedding_model": EMBED_MODEL,
            "owner_confirmed": False,
            "line_start": None,
            "line_end": line_count,
            "type": mem_type,
            "topic": topic,
        }

    # ------------------------------------------------------------------
    # Checkpoint save/load
    # ------------------------------------------------------------------

    def _save_checkpoint(self, progress: IngestProgress) -> None:
        data = progress.model_dump(mode="json")
        self._checkpoint_path.write_text(
            json.dumps(data, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _load_checkpoint(self) -> Optional[IngestProgress]:
        if not self._checkpoint_path.exists():
            return None
        try:
            data = json.loads(self._checkpoint_path.read_text(encoding="utf-8"))
            return IngestProgress(**data)
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("Could not load checkpoint: %s", e)
            return None

    def _clear_checkpoint(self) -> None:
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()

    # ------------------------------------------------------------------
    # Main
    # ------------------------------------------------------------------

    def run_ingestion(
        self,
        *,
        collection: str = "openclaw_memory_os",
        since_days: Optional[int] = None,
        dry_run: bool = False,
        limit: int = 0,
        resume: bool = True,
        skip_existing: bool = True,
        batch_size: int = 32,
    ) -> IngestProgress:
        """Run the full ingestion pipeline.

        Returns:
            Final :class:`IngestProgress` with results.
        """
        global _interrupted

        # Try restore from checkpoint
        if resume:
            cp = self._load_checkpoint()
            if cp is not None and cp.status == "running":
                self._progress = cp
                logger.info(
                    "Resumed from checkpoint: %d/%d chunks written, started at %s",
                    cp.written,
                    cp.total_chunks,
                    cp.started_at.isoformat(),
                )

        if self._progress is None:
            self._progress = IngestProgress(
                checkpoint_id=utcnow().strftime("ingest_%Y%m%d_%H%M%S"),
            )

        # Load the operator's personal taxonomy once per ingest run. This
        # is the single source of truth for brand-specific / personal
        # keywords; the real entries live in the gitignored
        # config/personal_taxonomy.json file. Missing / malformed file
        # degrades to an empty dict — the classifier still works, just
        # without operator-specific keyword enrichments.
        taxonomy = load_personal_taxonomy()

        audit = get_audit_store()

        # Discover files
        files = self._list_memory_files()
        if since_days is not None:
            cutoff = time.time() - since_days * 86400
            files = [f for f in files if f.stat().st_mtime >= cutoff]

        self._progress.total_files = len(files)
        logger.info("Files to process: %d", len(files))

        # Chunk
        chunks: List[Tuple[Path, str, str]] = []
        for f in files:
            try:
                text = self._read_file(f)
            except Exception as exc:
                logger.warning("Skip %s: %s", f, exc)
                continue
            for cid, ctext in self._split_by_sections(text, f):
                chunks.append((f, cid, ctext))

        if limit:
            chunks = chunks[:limit]

        self._progress.total_chunks = len(chunks)
        self._progress.source_files = [str(f) for f in files]
        logger.info("Chunks to process: %d", len(chunks))

        # Pre-load existing Qdrant IDs for skip-existing
        existing_ids: Set[str] = set()
        if skip_existing and not dry_run:
            if self._qdrant_collection_exists(collection):
                existing_ids = self._fetch_existing_ids(collection)
                logger.info("Existing Qdrant points: %d", len(existing_ids))
                # Pre-populate completed_chunk_ids
                completed_set = set(self._progress.completed_chunk_ids)
                for f, cid, ctext in chunks:
                    point_id = self._id_for_chunk(cid)
                    if point_id in existing_ids:
                        completed_set.add(cid)
                self._progress.completed_chunk_ids = list(completed_set)
                logger.info("Pre-populated completed chunks: %d", len(self._progress.completed_chunk_ids))

        if dry_run:
            audit.log("ingest_dry_run", detail=f"would ingest {len(chunks)} chunks from {len(files)} files, collection={collection}")
            self._progress.status = "completed"
            self._progress.updated_at = utcnow()
            self._save_checkpoint(self._progress)
            return self._progress

        # Ensure collection exists
        if not self._qdrant_collection_exists(collection):
            logger.info("Creating collection %s", collection)
            self._qdrant_create_collection(collection)
        else:
            logger.info(
                "Collection exists: %s (%d points)",
                collection,
                self._qdrant_point_count(collection),
            )

        written = self._progress.written
        failed = self._progress.failed
        batch: List[dict] = []

        for idx, (f, cid, ctext) in enumerate(chunks, 1):
            # Check resume skip
            completed_set = set(self._progress.completed_chunk_ids)
            if cid in completed_set:
                logger.debug("Skipping already-completed chunk: %s", cid)
                continue

            if _interrupted:
                logger.warning("Interrupted; saving checkpoint.")
                self._progress.status = "running"
                self._progress.updated_at = utcnow()
                self._save_checkpoint(self._progress)
                audit.log("ingest_interrupted", detail=f"interrupted at chunk {idx}/{len(chunks)}")
                return self._progress

            # Check existing in Qdrant (skip-existing)
            point_id = self._id_for_chunk(cid)
            if skip_existing and point_id in existing_ids:
                written += 1
                self._progress.written = written
                self._progress.completed_chunk_ids.append(cid)
                continue

            self._progress.current_chunk = cid
            meta = self._build_payload(ctext, f, cid, taxonomy)

            vec = self._embed(ctext)
            if vec is None:
                failed += 1
                self._progress.failed = failed
                if failed <= 3:
                    logger.warning("Embed failed for chunk %s", cid)
                continue

            point = {"id": point_id, "vector": vec, "payload": meta}
            batch.append(point)

            if len(batch) >= batch_size:
                self._qdrant_upsert(collection, batch)
                written += len(batch)
                for p in batch:
                    self._progress.completed_chunk_ids.append(
                        p["payload"]["id"]  # This is cid stored in payload
                    )
                batch = []
                self._progress.written = written

            if idx % 25 == 0 or idx == len(chunks):
                self._progress.updated_at = utcnow()
                logger.info(
                    "Progress: %d/%d chunks  written=%d  failed=%d",
                    idx,
                    len(chunks),
                    written,
                    failed,
                )
                self._save_checkpoint(self._progress)

        # Flush final batch
        if batch:
            self._qdrant_upsert(collection, batch)
            written += len(batch)
            for p in batch:
                self._progress.completed_chunk_ids.append(
                    p["payload"]["id"]
                )
            self._progress.written = written
            batch = []

        self._progress.status = "completed"
        self._progress.updated_at = utcnow()
        self._clear_checkpoint()

        audit.log(
            "ingest_completed",
            detail=f"written={written} failed={failed} files={len(files)} chunks={len(chunks)} collection={collection}",
        )

        logger.info(
            "Ingestion complete: written=%d  failed=%d  collection=%s",
            written,
            failed,
            collection,
        )
        return self._progress
