#!/usr/bin/env python3
"""Resumable shadow embedding reindex tool for Qdrant collections.

Reads points from a source Qdrant collection, re-embeds the text content
using the configured embed provider (NewAPI qwen3-embedding:0.6b), and
writes the new vectors into a target shadow collection — preserving point
IDs and payloads exactly.

Key contracts:
  * Concurrency = 1 (sequential requests; batch array within a single
    request is allowed).
  * Never write zero / NaN / Inf vectors.
  * Never modify the source collection.
  * Checkpoint + manifest support full resume after interruption.
  * No token/secret leakage to stdout or manifest.
  * Uses embed_provider.py's get_embed_provider() — no HTTP client duplication.

Usage examples:
  # Full reindex
  python scripts/reindex_qdrant_embeddings.py \\
      --source openclaw_memory_os \\
      --target openclaw_memory_os_qwen3e_v1 \\
      --checkpoint /tmp/ckpt.json \\
      --manifest /tmp/manifest.jsonl

  # Resume after interruption
  python scripts/reindex_qdrant_embeddings.py \\
      --source openclaw_memory_os \\
      --target openclaw_memory_os_qwen3e_v1 \\
      --checkpoint /tmp/ckpt.json \\
      --manifest /tmp/manifest.jsonl \\
      --resume

  # Canary (first 10 points only)
  python scripts/reindex_qdrant_embeddings.py \\
      --source openclaw_memory_os \\
      --target openclaw_memory_os_qwen3e_v1 \\
      --checkpoint /tmp/ckpt.json \\
      --manifest /tmp/manifest.jsonl \\
      --limit 10

  # Verify only (no migration)
  python scripts/reindex_qdrant_embeddings.py \\
      --source openclaw_memory_os \\
      --target openclaw_memory_os_qwen3e_v1 \\
      --checkpoint /tmp/ckpt.json \\
      --manifest /tmp/manifest.jsonl \\
      --verify-only

  # Reconcile (find differences)
  python scripts/reindex_qdrant_embeddings.py \\
      --source openclaw_memory_os \\
      --target openclaw_memory_os_qwen3e_v1 \\
      --checkpoint /tmp/ckpt.json \\
      --manifest /tmp/manifest.jsonl \\
      --reconcile
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# ---------------------------------------------------------------------------
# Logging — never echo secrets
# ---------------------------------------------------------------------------

logger = logging.getLogger("reindex_qdrant_embeddings")


def _safe_log(msg: str) -> None:
    """Log a message; strip any accidental token-like patterns."""
    # Never log lines that look like bearer tokens
    if "sk-" in msg and len(msg) > 40:
        msg = msg[:20] + "...<redacted>"
    logger.info(msg)


# ---------------------------------------------------------------------------
# Payload text extraction
# ---------------------------------------------------------------------------

# Ordered list of payload keys to try for embedding text.
# The tool inspects actual payload schema; it does not guess.
_TEXT_KEY_CANDIDATES = ["content", "_text", "text", "body", "message", "data"]


def _extract_text(payload: Dict[str, Any]) -> str:
    """Extract the embedding text from a point payload.

    Tries candidates in order; returns the first non-empty string found.
    Raises ValueError if no suitable key is found.
    """
    for key in _TEXT_KEY_CANDIDATES:
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            return val.strip()
    raise ValueError(
        f"No embedding text found in payload keys: "
        f"{list(payload.keys())} (tried {_TEXT_KEY_CANDIDATES})"
    )


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _payload_canonical_hash(payload: Dict[str, Any]) -> str:
    """Deterministic SHA-256 of a canonical JSON representation of the payload."""
    # Sort keys, ensure_ascii for byte-level determinism
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _text_hash(text: str) -> str:
    """SHA-256 of the embedding text (before truncation)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Vector validation
# ---------------------------------------------------------------------------


def _validate_vector(vec: List[float], expected_dim: int, *, context: str) -> List[float]:
    """Validate a vector: correct dim, finite, non-zero.

    Returns the validated vector list.
    Raises ValueError on any violation.
    """
    if not isinstance(vec, list):
        raise ValueError(f"{context}: vector is not a list (got {type(vec).__name__})")
    if len(vec) != expected_dim:
        raise ValueError(
            f"{context}: dimension mismatch — got {len(vec)}, expected {expected_dim}"
        )
    for i, x in enumerate(vec):
        if not isinstance(x, (int, float)):
            raise ValueError(f"{context}: non-numeric at index {i}: {x!r}")
        if math.isnan(x) or math.isinf(x):
            raise ValueError(f"{context}: NaN/Inf at index {i}")
    if all(v == 0.0 for v in vec):
        raise ValueError(f"{context}: all-zero vector")
    return vec


# ---------------------------------------------------------------------------
# Checkpoint
# ---------------------------------------------------------------------------


@dataclass
class Checkpoint:
    """Resumable checkpoint state."""

    source_collection: str = ""
    target_collection: str = ""
    next_offset: Optional[str] = None  # Qdrant scroll offset (str or None)
    processed: int = 0
    succeeded: int = 0
    failed: int = 0
    skipped: int = 0
    batch_size: int = 8
    embedding_model: str = ""
    embedding_dimensions: int = 768
    started_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Checkpoint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    def save(self, path: str) -> None:
        self.updated_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=True)
        os.replace(tmp, path)

    @classmethod
    def load(cls, path: str) -> "Checkpoint":
        with open(path, "r", encoding="utf-8") as f:
            return cls.from_dict(json.load(f))


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------


@dataclass
class ManifestEntry:
    """One line in the JSONL manifest — no full text stored."""

    point_id: Any  # int or str
    payload_hash: str
    embedded_text_hash: str
    target_written: bool = False


def _manifest_path_for_id(manifest_path: str) -> Dict[Any, ManifestEntry]:
    """Load manifest into a dict keyed by point_id for fast lookup."""
    result: Dict[Any, ManifestEntry] = {}
    if not os.path.exists(manifest_path):
        return result
    with open(manifest_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            entry = ManifestEntry(
                point_id=d["point_id"],
                payload_hash=d["payload_hash"],
                embedded_text_hash=d["embedded_text_hash"],
                target_written=d.get("target_written", False),
            )
            result[entry.point_id] = entry
    return result


def _append_manifest(manifest_path: str, entries: List[ManifestEntry]) -> None:
    """Append entries to the JSONL manifest."""
    with open(manifest_path, "a", encoding="utf-8") as f:
        for entry in entries:
            d = {
                "point_id": entry.point_id,
                "payload_hash": entry.payload_hash,
                "embedded_text_hash": entry.embedded_text_hash,
                "target_written": entry.target_written,
            }
            f.write(json.dumps(d, ensure_ascii=True) + "\n")


# ---------------------------------------------------------------------------
# Embedding batch
# ---------------------------------------------------------------------------


def _embed_batch(
    provider: Any,
    texts: List[str],
    expected_dim: int,
    batch_size: int,
) -> Tuple[List[List[float]], int]:
    """Embed a list of texts, returning vectors and the effective batch_size.

    Tries batch_size first; if the provider rejects the input array or
    returns wrong count, degrades to 8→4→2→1.

    Returns (vectors, effective_batch_size).
    """
    # Try batch embedding with input array
    for try_size in [batch_size, 8, 4, 2, 1]:
        if try_size > len(texts):
            continue
        try:
            vectors = _embed_batch_inner(provider, texts, expected_dim, try_size)
            if len(vectors) == len(texts):
                return vectors, try_size
            # Wrong count — degrade
            logger.warning(
                f"Batch size {try_size} returned {len(vectors)} vectors "
                f"for {len(texts)} texts; degrading"
            )
        except Exception as exc:
            logger.warning(f"Batch size {try_size} failed: {exc}; degrading")
            if try_size == 1:
                raise

    # Should not reach here, but fallback: one-by-one
    vectors = []
    for text in texts:
        vec = provider.embed(text)
        _validate_vector(vec, expected_dim, context="single embed")
        vectors.append(vec)
    return vectors, 1


def _embed_batch_inner(
    provider: Any,
    texts: List[str],
    expected_dim: int,
    batch_size: int,
) -> List[List[float]]:
    """Attempt batch embedding using the provider's embed method.

    For batch_size > 1, we try the OpenAI-compatible batch endpoint
    directly. If the provider only supports single embed(), we fall
    back to sequential calls.
    """
    if batch_size == 1:
        # Single embed via provider
        vec = provider.embed(texts[0])
        _validate_vector(vec, expected_dim, context="single embed")
        return [vec]

    # Try batch via direct API call (OpenAI-compatible /v1/embeddings with input array)
    # We need to use the provider's httpx client and config
    if provider.api_style == "openai" and hasattr(provider, "_get_client"):
        return _embed_batch_openai(provider, texts, expected_dim, batch_size)

    # Fallback: sequential single embeds
    vectors = []
    for text in texts:
        vec = provider.embed(text)
        _validate_vector(vec, expected_dim, context="sequential embed")
        vectors.append(vec)
    return vectors


def _embed_batch_openai(
    provider: Any,
    texts: List[str],
    expected_dim: int,
    batch_size: int,
) -> List[List[float]]:
    """Batch embed via OpenAI-compatible /v1/embeddings with input array."""
    import httpx

    client = provider._get_client()
    url = f"{provider.base_url}/embeddings"
    headers = {"Content-Type": "application/json"}
    if provider.api_key:
        headers["Authorization"] = f"Bearer {provider.api_key}"

    all_vectors: List[List[float]] = []

    # Process in chunks of batch_size
    for i in range(0, len(texts), batch_size):
        chunk = texts[i : i + batch_size]
        body = {
            "model": provider.model,
            "input": chunk,
            "dimensions": expected_dim,
        }
        try:
            resp = client.post(url, json=body, headers=headers)
        except Exception as exc:
            raise RuntimeError(f"Batch embed POST failed: {exc}") from exc

        if resp.status_code != 200:
            body_excerpt = (resp.text or "")[:200]
            raise RuntimeError(
                f"Batch embed returned HTTP {resp.status_code}: {body_excerpt}"
            )

        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"Batch embed returned non-JSON: {exc}") from exc

        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError(f"Batch embed: expected data list, got {type(rows)}")

        # Sort by index to maintain order
        rows_sorted = sorted(rows, key=lambda r: r.get("index", 0))
        if len(rows_sorted) != len(chunk):
            raise RuntimeError(
                f"Batch embed: returned {len(rows_sorted)} vectors for {len(chunk)} texts"
            )

        for row in rows_sorted:
            vec = row.get("embedding") if isinstance(row, dict) else None
            vec = _validate_vector(
                vec if isinstance(vec, list) else [],
                expected_dim,
                context="batch embed",
            )
            all_vectors.append(vec)

    return all_vectors


# ---------------------------------------------------------------------------
# Core reindex logic
# ---------------------------------------------------------------------------


def _create_shadow_collection(client: Any, name: str, dim: int) -> None:
    """Create a shadow collection with the given name and dimension.

    If it already exists, verify schema matches. Hard-stop on mismatch.
    """
    from qdrant_client import models

    existing = None
    try:
        existing = client.get_collection(name)
    except Exception:
        pass

    if existing is not None:
        # Verify schema
        vc = existing.config.params.vectors
        # Handle both single vector and dict of named vectors
        if hasattr(vc, "size"):
            actual_size = vc.size
            actual_dist = vc.distance
        else:
            # Named vectors — get the default or first
            if isinstance(vc, dict):
                first = list(vc.values())[0]
                actual_size = first.size
                actual_dist = first.distance
            else:
                actual_size = getattr(vc, "size", None)
                actual_dist = getattr(vc, "distance", None)

        if actual_size != dim:
            raise RuntimeError(
                f"Collection {name} exists but size={actual_size}, expected {dim}. HARD STOP."
            )
        dist_str = str(actual_dist)
        if "COSINE" not in dist_str.upper():
            raise RuntimeError(
                f"Collection {name} exists but distance={actual_dist}, expected Cosine. HARD STOP."
            )
        logger.info(f"Collection {name} exists with correct schema (size={dim}, Cosine)")
        return

    # Create new
    client.create_collection(
        collection_name=name,
        vectors_config=models.VectorParams(size=dim, distance=models.Distance.COSINE),
    )
    logger.info(f"Created collection {name} (size={dim}, Cosine)")


def _scroll_all_source(
    client: Any,
    collection: str,
    limit: int = 0,
    offset: Optional[str] = None,
) -> List[Any]:
    """Scroll source collection, returning points with payload (no vectors).

    If limit > 0, return at most that many points.
    If offset is provided, start from that scroll offset.
    """
    from qdrant_client import models

    points = []
    batch = 100
    current_offset = offset

    while True:
        pts, next_offset = client.scroll(
            collection,
            limit=batch,
            offset=current_offset,
            with_payload=True,
            with_vectors=False,
        )
        if not pts:
            break
        points.extend(pts)
        if limit > 0 and len(points) >= limit:
            points = points[:limit]
            break
        current_offset = next_offset
        if next_offset is None:
            break

    return points


def _get_source_point_ids(client: Any, collection: str) -> Set[Any]:
    """Get all point IDs from a source collection (no payload, no vectors)."""
    ids = set()
    offset = None
    batch = 1000
    while True:
        pts, next_offset = client.scroll(
            collection,
            limit=batch,
            offset=offset,
            with_payload=False,
            with_vectors=False,
        )
        if not pts:
            break
        for p in pts:
            ids.add(p.id)
        offset = next_offset
        if next_offset is None:
            break
    return ids


def _get_target_point_ids(client: Any, collection: str) -> Set[Any]:
    """Get all point IDs from a target collection."""
    return _get_source_point_ids(client, collection)


# ---------------------------------------------------------------------------
# Main reindex
# ---------------------------------------------------------------------------


def run_reindex(
    source: str,
    target: str,
    checkpoint_path: str,
    manifest_path: str,
    limit: int = 0,
    batch_size: int = 8,
    resume: bool = False,
    verify_only: bool = False,
    reconcile: bool = False,
    progress_file: Optional[str] = None,
    progress_interval: int = 500,
) -> Dict[str, Any]:
    """Run the shadow embedding reindex.

    Returns a summary dict with stats.
    """
    import os as _os

    # Set embed provider to newapi for this process
    _os.environ["EMBED_PROVIDER"] = "newapi"

    from qdrant_client import QdrantClient

    from openclaw_memory_os.embed_provider import get_embed_provider, reset_provider_caches

    # Reset provider cache to pick up newapi
    reset_provider_caches()
    provider = get_embed_provider()

    client = QdrantClient("127.0.0.1", port=6333)

    # --- Reconcile mode ---
    if reconcile:
        return _run_reconcile(client, source, target)

    # --- Verify-only mode ---
    if verify_only:
        return _run_verify(client, source, target, provider.expected_dim)

    # --- Reindex mode ---
    # Load or create checkpoint
    ckpt = Checkpoint()
    manifest_index: Dict[Any, ManifestEntry] = {}

    if resume:
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        ckpt = Checkpoint.load(checkpoint_path)
        if ckpt.source_collection != source or ckpt.target_collection != target:
            raise ValueError(
                f"Checkpoint source/target mismatch: "
                f"ckpt=({ckpt.source_collection}, {ckpt.target_collection}) "
                f"vs args=({source}, {target})"
            )
        manifest_index = _manifest_path_for_id(manifest_path)
        logger.info(
            f"Resuming from checkpoint: processed={ckpt.processed}, "
            f"succeeded={ckpt.succeeded}, next_offset={ckpt.next_offset}"
        )
    else:
        ckpt = Checkpoint(
            source_collection=source,
            target_collection=target,
            batch_size=batch_size,
            embedding_model=provider.model,
            embedding_dimensions=provider.expected_dim,
            started_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        )
        ckpt.save(checkpoint_path)
        # Clear manifest for fresh start
        if os.path.exists(manifest_path):
            os.remove(manifest_path)

    # Ensure shadow collection exists
    _create_shadow_collection(client, target, provider.expected_dim)

    # Scroll source
    scroll_offset = ckpt.next_offset
    total_to_process = limit if limit > 0 else 0
    effective_batch = ckpt.batch_size
    last_progress_time = time.time()
    start_time = time.time()

    # Track already-succeeded point IDs from manifest
    succeeded_ids: Set[Any] = set()
    for pid, entry in manifest_index.items():
        if entry.target_written:
            succeeded_ids.add(pid)

    while True:
        # Scroll a batch of source points
        scroll_batch = max(effective_batch * 2, 50)
        pts, next_offset = client.scroll(
            source,
            limit=scroll_batch,
            offset=scroll_offset,
            with_payload=True,
            with_vectors=False,
        )

        if not pts:
            break

        # Filter out already-succeeded points
        pending = []
        for p in pts:
            pid = p.id
            if pid in succeeded_ids:
                ckpt.skipped += 1
                continue
            pending.append(p)

        if not pending and next_offset is None:
            break

        # Apply limit
        if total_to_process > 0:
            remaining = total_to_process - ckpt.processed
            if remaining <= 0:
                break
            pending = pending[:remaining]

        if not pending:
            # All in this batch were already done; advance offset
            scroll_offset = next_offset
            ckpt.next_offset = next_offset
            ckpt.save(checkpoint_path)
            if next_offset is None:
                break
            continue

        # Extract texts
        texts = []
        payload_hashes = []
        text_hashes = []
        for p in pending:
            payload = p.payload or {}
            try:
                text = _extract_text(payload)
            except ValueError as exc:
                logger.error(f"Point {p.id}: {exc}; skipping")
                ckpt.failed += 1
                ckpt.processed += 1
                continue
            texts.append(text)
            payload_hashes.append(_payload_canonical_hash(payload))
            text_hashes.append(_text_hash(text))

        if not texts:
            scroll_offset = next_offset
            ckpt.next_offset = next_offset
            ckpt.save(checkpoint_path)
            if next_offset is None:
                break
            continue

        # Embed batch
        try:
            vectors, effective_batch = _embed_batch(
                provider, texts, provider.expected_dim, effective_batch
            )
        except Exception as exc:
            logger.error(f"Embedding batch failed: {exc}")
            # Try one-by-one as last resort
            vectors = []
            for i, text in enumerate(texts):
                try:
                    vec = provider.embed(text)
                    _validate_vector(vec, provider.expected_dim, context=f"fallback embed point {pending[i].id}")
                    vectors.append(vec)
                except Exception as e2:
                    logger.error(f"  Point {pending[i].id} embed failed: {e2}")
                    vectors.append(None)
            effective_batch = 1

        # Validate and upsert
        manifest_entries = []
        for i, (p, vec) in enumerate(zip(pending, vectors)):
            pid = p.id
            if vec is None:
                ckpt.failed += 1
                ckpt.processed += 1
                manifest_entries.append(
                    ManifestEntry(
                        point_id=pid,
                        payload_hash=payload_hashes[i],
                        embedded_text_hash=text_hashes[i],
                        target_written=False,
                    )
                )
                continue

            try:
                _validate_vector(vec, provider.expected_dim, context=f"point {pid}")
            except ValueError as exc:
                logger.error(f"Point {pid} vector invalid: {exc}")
                ckpt.failed += 1
                ckpt.processed += 1
                manifest_entries.append(
                    ManifestEntry(
                        point_id=pid,
                        payload_hash=payload_hashes[i],
                        embedded_text_hash=text_hashes[i],
                        target_written=False,
                    )
                )
                continue

            # Upsert to target
            from qdrant_client import models

            client.upsert(
                target,
                points=[
                    models.PointStruct(
                        id=pid,
                        vector=vec,
                        payload=p.payload or {},
                    )
                ],
            )
            ckpt.succeeded += 1
            ckpt.processed += 1
            succeeded_ids.add(pid)
            manifest_entries.append(
                ManifestEntry(
                    point_id=pid,
                    payload_hash=payload_hashes[i],
                    embedded_text_hash=text_hashes[i],
                    target_written=True,
                )
            )

        # Save manifest and checkpoint
        _append_manifest(manifest_path, manifest_entries)
        scroll_offset = next_offset
        ckpt.next_offset = next_offset
        ckpt.batch_size = effective_batch
        ckpt.save(checkpoint_path)

        # Progress reporting
        now = time.time()
        if ckpt.processed % progress_interval < len(pending) or (now - last_progress_time) > 60:
            elapsed = now - start_time
            _safe_log(
                f"[{source}] processed={ckpt.processed} succeeded={ckpt.succeeded} "
                f"failed={ckpt.failed} batch_size={effective_batch} "
                f"offset={scroll_offset} elapsed={elapsed:.0f}s"
            )
            last_progress_time = now

            # Write progress file
            if progress_file:
                _write_progress(progress_file, source, ckpt, elapsed)

        # Check limit
        if total_to_process > 0 and ckpt.processed >= total_to_process:
            break

        if next_offset is None:
            break

    # Final save
    ckpt.save(checkpoint_path)

    elapsed = time.time() - start_time
    summary = {
        "source": source,
        "target": target,
        "processed": ckpt.processed,
        "succeeded": ckpt.succeeded,
        "failed": ckpt.failed,
        "skipped": ckpt.skipped,
        "elapsed_seconds": round(elapsed, 1),
    }
    _safe_log(
        f"Reindex complete: {summary}"
    )

    # Clean up provider
    reset_provider_caches()

    return summary


def _write_progress(progress_file: str, source: str, ckpt: Checkpoint, elapsed: float) -> None:
    """Append a progress line to the JSONL progress file."""
    entry = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "collection": source,
        "processed": ckpt.processed,
        "succeeded": ckpt.succeeded,
        "failed": ckpt.failed,
        "batch_size": ckpt.batch_size,
        "next_offset": ckpt.next_offset,
        "elapsed_seconds": round(elapsed, 1),
    }
    try:
        with open(progress_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=True) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to write progress file: {exc}")


# ---------------------------------------------------------------------------
# Verify
# ---------------------------------------------------------------------------


def _run_verify(
    client: Any,
    source: str,
    target: str,
    expected_dim: int,
) -> Dict[str, Any]:
    """Verify target collection against source."""
    source_info = client.get_collection(source)
    target_info = client.get_collection(target)

    source_count = source_info.points_count
    target_count = target_info.points_count

    result = {
        "source": source,
        "target": target,
        "source_count": source_count,
        "target_count": target_count,
        "count_match": source_count == target_count,
        "id_match": None,
        "hash_match": None,
        "vectors_valid": None,
        "errors": [],
    }

    if source_count != target_count:
        result["errors"].append(f"Count mismatch: source={source_count}, target={target_count}")
        # Still continue to check IDs

    # Check IDs
    source_ids = _get_source_point_ids(client, source)
    target_ids = _get_target_point_ids(client, target)
    result["id_match"] = source_ids == target_ids
    if not result["id_match"]:
        missing = source_ids - target_ids
        extra = target_ids - source_ids
        if missing:
            result["errors"].append(f"Missing in target: {len(missing)} points")
        if extra:
            result["errors"].append(f"Extra in target: {len(extra)} points")

    # Check payload hashes and vectors (sample for large collections)
    hash_mismatches = 0
    vector_errors = 0
    checked = 0

    offset = None
    batch = 100
    while True:
        src_pts, next_offset = client.scroll(
            source, limit=batch, offset=offset, with_payload=True, with_vectors=False
        )
        if not src_pts:
            break

        # Get corresponding target points
        src_ids = [p.id for p in src_pts]
        tgt_pts = client.retrieve(target, ids=src_ids, with_payload=True, with_vectors=True)

        tgt_by_id = {p.id: p for p in tgt_pts}

        for sp in src_pts:
            checked += 1
            tp = tgt_by_id.get(sp.id)
            if tp is None:
                result["errors"].append(f"Point {sp.id} missing in target")
                continue

            # Payload hash
            src_hash = _payload_canonical_hash(sp.payload or {})
            tgt_hash = _payload_canonical_hash(tp.payload or {})
            if src_hash != tgt_hash:
                hash_mismatches += 1

            # Vector validation
            vec = tp.vector if hasattr(tp, "vector") else None
            if vec is None:
                vector_errors += 1
                continue
            try:
                _validate_vector(vec, expected_dim, context=f"verify point {sp.id}")
            except ValueError:
                vector_errors += 1

        offset = next_offset
        if next_offset is None:
            break

    result["hash_match"] = hash_mismatches == 0
    result["vectors_valid"] = vector_errors == 0
    result["checked"] = checked
    result["hash_mismatches"] = hash_mismatches
    result["vector_errors"] = vector_errors

    if hash_mismatches > 0:
        result["errors"].append(f"Payload hash mismatches: {hash_mismatches}")
    if vector_errors > 0:
        result["errors"].append(f"Vector errors: {vector_errors}")

    return result


# ---------------------------------------------------------------------------
# Reconcile
# ---------------------------------------------------------------------------


def _run_reconcile(
    client: Any,
    source: str,
    target: str,
) -> Dict[str, Any]:
    """Find differences between source and target collections."""
    source_ids = _get_source_point_ids(client, source)
    target_ids = _get_target_point_ids(client, target)

    missing_in_target = source_ids - target_ids
    extra_in_target = target_ids - source_ids

    result = {
        "source": source,
        "target": target,
        "source_count": len(source_ids),
        "target_count": len(target_ids),
        "missing_in_target": len(missing_in_target),
        "extra_in_target": len(extra_in_target),
        "payload_changes": 0,
        "missing_ids_sample": sorted(list(missing_in_target))[:20],
        "extra_ids_sample": sorted(list(extra_in_target))[:20],
    }

    # Check payload changes for common IDs
    common_ids = source_ids & target_ids
    payload_changes = 0
    offset = None
    batch = 100
    while True:
        src_pts, next_offset = client.scroll(
            source, limit=batch, offset=offset, with_payload=True, with_vectors=False
        )
        if not src_pts:
            break
        src_ids_batch = [p.id for p in src_pts if p.id in common_ids]
        if src_ids_batch:
            tgt_pts = client.retrieve(target, ids=src_ids_batch, with_payload=True, with_vectors=False)
            tgt_by_id = {p.id: p for p in tgt_pts}
            for sp in src_pts:
                if sp.id not in common_ids:
                    continue
                tp = tgt_by_id.get(sp.id)
                if tp is None:
                    continue
                src_hash = _payload_canonical_hash(sp.payload or {})
                tgt_hash = _payload_canonical_hash(tp.payload or {})
                if src_hash != tgt_hash:
                    payload_changes += 1
        offset = next_offset
        if next_offset is None:
            break

    result["payload_changes"] = payload_changes
    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Resumable shadow embedding reindex for Qdrant collections"
    )
    parser.add_argument("--source", required=True, help="Source collection name")
    parser.add_argument("--target", required=True, help="Target shadow collection name")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint file path (JSON)")
    parser.add_argument("--manifest", required=True, help="Manifest file path (JSONL)")
    parser.add_argument("--limit", type=int, default=0, help="Max points to process (0=all)")
    parser.add_argument("--batch-size", type=int, default=8, help="Initial batch size")
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    parser.add_argument("--verify-only", action="store_true", help="Only verify, no migration")
    parser.add_argument("--reconcile", action="store_true", help="Find source/target differences")
    parser.add_argument("--progress-file", default=None, help="JSONL progress log path")
    parser.add_argument("--progress-interval", type=int, default=500, help="Progress log every N points")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )

    result = run_reindex(
        source=args.source,
        target=args.target,
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        limit=args.limit,
        batch_size=args.batch_size,
        resume=args.resume,
        verify_only=args.verify_only,
        reconcile=args.reconcile,
        progress_file=args.progress_file,
        progress_interval=args.progress_interval,
    )

    # Print summary (never include tokens/secrets)
    print(json.dumps(result, indent=2, ensure_ascii=True, default=str))

    # Exit with error if there were failures
    if result.get("failed", 0) > 0:
        sys.exit(1)
    if result.get("errors"):
        sys.exit(1)


if __name__ == "__main__":
    main()
