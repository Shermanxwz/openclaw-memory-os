#!/usr/bin/env python3
"""MinHash-based duplicate detection.

Why MinHash instead of Jaccard over all pairs:
    - Jaccard over n=25k points = O(n^2) ~= 6.3 * 10^8 pairs per run. With
      shingling + token sets, this takes 1-2 minutes per dashboard load
      and is the main reason the dashboard 504'd when first wired to
      Qdrant.
    - MinHash over a fixed signature size gives approximate Jaccard in
      O(n) per comparison. With a banded LSH index we can find candidate
      duplicates in O(n * bands) ~= O(n).

Outputs a JSON file with the cluster list, matching the shape the
dashboard expects from /api/duplicates. We do not mutate Qdrant here --
that is the job of supersede_detect.py which reads the JSON and writes
back.

Usage:
    .venv/bin/python scripts/dedup_cron.py [--collection NAME] [--threshold 0.6]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "openclaw_memory_os")

NUM_PERM = 128            # signature size
BANDS = 32                # LSH bands
ROWS_PER_BAND = NUM_PERM // BANDS
SHINGLE_K = 4
TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")


def _shingles(text: str, k: int = SHINGLE_K) -> Set[str]:
    tokens = [t.lower() for t in TOKEN_RE.findall(text or "")]
    if len(tokens) < k:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + k]) for i in range(len(tokens) - k + 1)}


def _minhash_signature(shingles: Set[str], num_perm: int = NUM_PERM) -> List[int]:
    sig: List[int] = [math.inf] * num_perm
    for sh in shingles:
        h = int.from_bytes(hashlib.sha1(sh.encode()).digest()[:8], "big")
        for i in range(num_perm):
            # Simple universal hash family: a*i + b mod p
            mixed = (h ^ (i * 0x9E3779B97F4A7C15)) & 0xFFFFFFFFFFFFFFFF
            if mixed < sig[i]:
                sig[i] = mixed
    return [int(v) if v != math.inf else 0 for v in sig]


def _lsh_buckets(sig: List[int]) -> List[Tuple[int, ...]]:
    out: List[Tuple[int, ...]] = []
    for b in range(BANDS):
        start = b * ROWS_PER_BAND
        end = start + ROWS_PER_BAND
        out.append(tuple(sig[start:end]))
    return out


def _exact_jaccard(a: Set[str], b: Set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def scroll_all(collection: str, page_size: int = 512) -> List[dict]:
    """Return all payloads (id, text, source) from a collection."""
    out: List[dict] = []
    offset = None
    while True:
        body = {"limit": page_size, "with_payload": True, "with_vectors": False}
        if offset is not None:
            body["offset"] = offset
        req = urllib.request.Request(
            f"{QDRANT_URL}/collections/{collection}/points/scroll",
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=120) as r:
            data = json.loads(r.read())
        points = data.get("result", {}).get("points", [])
        out.extend(points)
        offset = data.get("result", {}).get("next_page_offset")
        if offset is None:
            break
    return out


def detect_duplicates(collection: str, threshold: float, max_points: int) -> List[dict]:
    points = scroll_all(collection)
    if max_points:
        points = points[:max_points]
    n = len(points)
    print(f"[dedup] loaded {n} points")

    sigs: List[List[int]] = []
    shingle_sets: List[Set[str]] = []
    for p in points:
        payload = p.get("payload") or {}
        text = payload.get("content") or payload.get("text") or ""
        sh = _shingles(text)
        shingle_sets.append(sh)
        sigs.append(_minhash_signature(sh))

    # LSH buckets per band.
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    def union(i: int, j: int) -> None:
        ri, rj = find(i), find(j)
        if ri != rj:
            parent[ri] = rj

    bucket_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band_idx, band in enumerate(_lsh_buckets(sig)):
            bucket_map[(band_idx, hash(band))].append(i)

    candidate_pairs: Set[Tuple[int, int]] = set()
    for bucket in bucket_map.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if a > b:
                    a, b = b, a
                candidate_pairs.add((a, b))

    print(f"[dedup] LSH candidate pairs: {len(candidate_pairs)}")

    clusters: Dict[int, List[int]] = defaultdict(list)
    for a, b in candidate_pairs:
        jac = _exact_jaccard(shingle_sets[a], shingle_sets[b])
        if jac >= threshold:
            union(a, b)

    for i in range(n):
        clusters[find(i)].append(i)

    cluster_list: List[dict] = []
    for indices in clusters.values():
        if len(indices) < 2:
            continue
        # average pair similarity
        sims = []
        for ai in range(len(indices)):
            for bi in range(ai + 1, len(indices)):
                sims.append(_exact_jaccard(shingle_sets[indices[ai]], shingle_sets[indices[bi]]))
        avg_sim = sum(sims) / len(sims) if sims else 0.0
        members = []
        for i in indices:
            payload = points[i].get("payload") or {}
            members.append({
                "id": str(points[i]["id"]),
                "source": payload.get("source"),
                # Privacy contract (Runbook G7.4 / governance-layer): the
                # cluster dump used to write ``text_preview`` = first
                # 200 chars of the memory payload, which is a P0-5
                # leak when this JSON file lands on disk via
                # ``--out``. We now write only the length indicator.
                # See test_preview_logging_removed.py.
                "text_len": len(payload.get("content") or payload.get("text") or ""),
            })
        members.sort(key=lambda m: m["id"])
        representative = members[-1]
        cluster_list.append({
            "representative_id": representative["id"],
            "member_ids": [m["id"] for m in members],
            "score": round(avg_sim, 4),
            "members": members,
            "rationale": f"LSH cluster; avg_jaccard={avg_sim:.2f}; size={len(indices)}",
        })

    cluster_list.sort(key=lambda c: c["score"], reverse=True)
    return cluster_list


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--threshold", type=float, default=0.6)
    p.add_argument("--max-points", type=int, default=0)
    p.add_argument("--out", default="")
    args = p.parse_args(argv)

    started = time.perf_counter()
    clusters = detect_duplicates(args.collection, args.threshold, args.max_points)
    elapsed = time.perf_counter() - started
    out = {
        "collection": args.collection,
        "threshold": args.threshold,
        "cluster_count": len(clusters),
        "elapsed_sec": round(elapsed, 3),
        "clusters": clusters,
    }
    text = json.dumps(out, ensure_ascii=False, indent=2)
    if args.out:
        Path = __import__("pathlib").Path
        Path(args.out).write_text(text, encoding="utf-8")
        print(f"[dedup] wrote {args.out} ({len(text)} bytes, {len(clusters)} clusters, {elapsed:.2f}s)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())