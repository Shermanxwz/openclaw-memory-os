#!/usr/bin/env python3
"""Auto-detect supersede links in a Qdrant collection.

Heuristic triggers (v0.2.3 — conservative defaults):

    1. Keyword-topic supersede (DISABLED by default). Group all points
       that share a topic keyword and supersede older → newer when the
       recency gap is large. This was the original heuristic but it is
       far too broad in production: every MEMORY.md note mentions
       "worker" or "agent", so a single maintenance run marked 21917 of
       25444 points superseded. Enable only via the
       ``ENABLE_TOPIC_SUPERSEDE=1`` environment flag for an opt-in
       audit pass.

    2. High content similarity (Jaccard >= HIGH_CONFIDENCE_JACCARD,
       default 0.95). Above this threshold the documents are near
       identical; supersede older → newer. Requires:
         * at least ``MIN_SHINGLES_FOR_SUPERSEDE`` (default 8) shared
           shingles in both texts (avoids tiny 4-shingle fragments
           scoring high by coincidence),
         * and, when available, matching topic / category so we don't
           merge unrelated "current model" status notes that happen to
           share boilerplate.

    3. Moderate content similarity (Jaccard >= NEAR_DUPLICATE_JACCARD,
       default 0.80, < HIGH_CONFIDENCE_JACCARD). Tag both with
       ``review_reason: "near_duplicate"``. **No status change.**

Safeguards (v0.2.3):

    * ``SUPERSEDE_MAX_APPLY`` env (default 200) caps the number of
      points actually written per collection per run. Excess links are
      dropped with a warning so a future regression can't mass-mark
      points in a single pass. Dry-run is uncapped.
    * Tier guard: any point whose payload tier is ``core`` or ``long``
      (i.e. owner-confirmed durable memory) is excluded from
      auto-supersede regardless of similarity. Apply never touches
      them.

We use Qdrant upsert via ``scripts/_qdrant_helpers.update_payloads`` so
integer IDs reach Qdrant as native ints (the P0 fix for writeback 400s).
The script is idempotent: re-running it only mutates points that are
not already ``superseded``.

Usage:
    .venv/bin/python scripts/supersede_detect.py [--collection NAME] [--dry-run]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Set, Tuple

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "openclaw_memory_os")

# --- Conservative defaults (v0.2.3) ---------------------------------------

# Topic-based supersede is too aggressive in production. Opt-in only.
# Set ENABLE_TOPIC_SUPERSEDE=1 to run the keyword-grouped supersede pass.
ENABLE_TOPIC_SUPERSEDE = os.environ.get("ENABLE_TOPIC_SUPERSEDE", "").lower() in (
    "1", "true", "yes", "on",
)

# Lifecycle-changing auto-supersede is also opt-in by default. The safe
# unattended behavior is to tag near-duplicates for review while keeping
# memories active. Set ENABLE_AUTO_SUPERSEDE=1 for deliberate cleanup runs.
ENABLE_AUTO_SUPERSEDE = os.environ.get("ENABLE_AUTO_SUPERSEDE", "").lower() in (
    "1", "true", "yes", "on",
)

# MinHash-based content similarity thresholds.
# v0.2.3: raised from 0.85 → 0.95 so only near-identical documents auto-merge.
HIGH_CONFIDENCE_JACCARD = float(os.environ.get("HIGH_CONFIDENCE_JACCARD", "0.95"))
NEAR_DUPLICATE_JACCARD = float(os.environ.get("NEAR_DUPLICATE_JACCARD", "0.80"))

# Require at least this many shingles *shared* between the two texts before
# we consider the high-confidence match "real". Prevents a 12-char snippet
# (3 shingles) from scoring 1.0 Jaccard and merging unrelated memories.
MIN_SHINGLES_FOR_SUPERSEDE = int(os.environ.get("MIN_SHINGLES_FOR_SUPERSEDE", "8"))

# Cap writes per collection per run. Excess links are dropped with a
# warning so a regression can't mass-supersede points in a single pass.
# Dry-run is unaffected.
SUPERSEDE_MAX_APPLY = int(os.environ.get("SUPERSEDE_MAX_APPLY", "200"))

# Large collections can produce huge LSH candidate sets. For unattended daily
# maintenance, skip the expensive content-similarity pass above this active
# point count unless explicitly overridden. Tier/snapshot/expiry still run.
CONTENT_SUPERSEDE_MAX_POINTS = int(os.environ.get("CONTENT_SUPERSEDE_MAX_POINTS", "5000"))
FORCE_CONTENT_SUPERSEDE = os.environ.get("FORCE_CONTENT_SUPERSEDE", "").lower() in (
    "1", "true", "yes", "on",
)

NUM_PERM = 128
BANDS = 32
ROWS_PER_BAND = NUM_PERM // BANDS
SHINGLE_K = 4

# Tiers that MUST never be auto-superseded. These represent durable,
# owner-confirmed memory (core = explicit commitments, long = lesson
# library). They require manual review or a deliberate workflow to
# retire.
PROTECTED_TIERS = frozenset({"core", "long"})

TOKEN_RE = re.compile(r"[A-Za-z0-9_\u4e00-\u9fff]+")

SUPERSEDE_KEYWORDS = {
    # topic -> list of keyword triggers
    "worker": ["worker", "工人", "flash", "minimax", "m3"],
    "model": ["模型", "model"],
    "agent": ["agent", "claude", "codex", "主控"],
    "endpoint": ["endpoint", "url", "api"],
    "ip": ["ip", "公网"],
}


def _topic_signature(text: str) -> Optional[str]:
    low = text.lower()
    for topic, kws in SUPERSEDE_KEYWORDS.items():
        if any(kw.lower() in low for kw in kws):
            return topic
    return None


# --- MinHash / content similarity (reused from dedup_cron.py) -------------

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
        out.extend(data.get("result", {}).get("points", []))
        offset = data.get("result", {}).get("next_page_offset")
        if offset is None:
            break
    return out


def _date_from_source(source: str) -> Optional[datetime]:
    if not source:
        return None
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", source)
    if not m:
        return None
    try:
        return datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return None


def _is_protected(payload: Dict) -> bool:
    """Owner-confirmed durable memory must not be auto-superseded."""
    tier = (payload.get("tier") or "").lower()
    return tier in PROTECTED_TIERS


# --- Topic-based supersede detection (original logic) --------------------

def detect_supersedes(
    points: List[dict],
    recency_gap_days: int = 7,
    enabled: bool = ENABLE_TOPIC_SUPERSEDE,
) -> List[Tuple[object, object]]:
    """Return a list of ``(older_id, newer_id)`` supersede links from keyword topics.

    Disabled by default (v0.2.3). The caller should check
    :data:`ENABLE_TOPIC_SUPERSEDE` and skip this pass when it is false.
    Even when enabled, points with ``tier`` in :data:`PROTECTED_TIERS`
    are excluded.

    The id type follows whatever Qdrant returned (int for integer-ID
    collections, str for UUID collections). ``_qdrant_helpers.update_payloads``
    routes both forms through ``coerce_point_id`` so the downstream
    writeback never hits a 400 due to a stringified integer ID.
    """
    if not enabled:
        return []

    bucket: Dict[str, List[dict]] = defaultdict(list)
    for p in points:
        payload = p.get("payload") or {}
        text = payload.get("content") or payload.get("text") or ""
        topic = _topic_signature(text)
        if not topic:
            continue
        if payload.get("status") == "superseded":
            continue
        if _is_protected(payload):
            continue
        # Pass the raw ID through; _qdrant_helpers.update_payloads routes
        # it through coerce_point_id so integer IDs reach Qdrant as
        # native ints (the P0 fix for writeback 400s).
        bucket[topic].append({"id": p["id"], "payload": payload, "text": text})

    links: List[Tuple[object, object]] = []
    for topic, items in bucket.items():
        if len(items) < 2:
            continue
        items.sort(key=lambda it: (
            _date_from_source(it["payload"].get("source") or "") or datetime.min.replace(tzinfo=timezone.utc),
            str(it["id"]),
        ), reverse=True)
        newest = items[0]
        newest_date = _date_from_source(newest["payload"].get("source") or "")
        for older in items[1:]:
            older_date = _date_from_source(older["payload"].get("source") or "")
            if newest_date and older_date and (newest_date - older_date).days < recency_gap_days:
                continue
            links.append((older["id"], newest["id"]))

    return links


# --- Content-similarity-based supersede detection -------------------------

def detect_content_supersedes(
    points: List[dict],
    *,
    high_threshold: float = HIGH_CONFIDENCE_JACCARD,
    near_threshold: float = NEAR_DUPLICATE_JACCARD,
    min_shingles: int = MIN_SHINGLES_FOR_SUPERSEDE,
) -> Tuple[List[Tuple[object, object]], List[object]]:
    """Return (supersede_links, near_duplicate_ids) from content similarity.

    High confidence (>= high_threshold): supersede older → newer, subject to:
      * at least ``min_shingles`` shared shingles between the two texts
        (so a short snippet can't score 1.0 by coincidence),
      * and, when both sides carry a ``topic`` or ``category`` field,
        those fields must agree (avoids merging unrelated "current
        model" status notes that share boilerplate).

    Moderate confidence (>= near_threshold, < high_threshold): tag both
      with ``review_reason: "near_duplicate"`` only — status is NOT
      changed.

    Protected tiers (``core``, ``long``) are never auto-superseded but
    may still be tagged as near-duplicate for review.
    """
    active = [p for p in points
              if (p.get("payload") or {}).get("status") != "superseded"]
    n = len(active)
    if n < 2:
        return [], []

    shingle_sets = []
    for p in active:
        payload = p.get("payload") or {}
        text = payload.get("content") or payload.get("text") or ""
        shingle_sets.append(_shingles(text))

    sigs = [_minhash_signature(sh) for sh in shingle_sets]
    bucket_map: Dict[Tuple[int, int], List[int]] = defaultdict(list)
    for i, sig in enumerate(sigs):
        for band_idx, band in enumerate(_lsh_buckets(sig)):
            bucket_map[(band_idx, hash(band))].append(i)

    candidate_pairs: Set[Tuple[int, int]] = set()
    for bucket in bucket_map.values():
        if len(bucket) < 2 or len(bucket) > 200:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if a > b:
                    a, b = b, a
                candidate_pairs.add((a, b))

    supersede_links: List[Tuple[object, object]] = []
    near_dup_ids: Set[object] = set()

    for a, b in candidate_pairs:
        jac = _exact_jaccard(shingle_sets[a], shingle_sets[b])
        if jac >= high_threshold:
            a_payload = active[a].get("payload") or {}
            b_payload = active[b].get("payload") or {}
            a_date = _date_from_source(a_payload.get("source") or "") or datetime.min.replace(tzinfo=timezone.utc)
            b_date = _date_from_source(b_payload.get("source") or "") or datetime.min.replace(tzinfo=timezone.utc)
            a_id = active[a]["id"]
            b_id = active[b]["id"]

            # Tier guard: never auto-supersede core/long memories.
            if _is_protected(a_payload) or _is_protected(b_payload):
                # Surface as a near-duplicate hint instead of a supersede.
                if jac >= near_threshold:
                    near_dup_ids.add(a_id)
                    near_dup_ids.add(b_id)
                continue

            # Shared-shingle floor. A high Jaccard on a tiny shingle set
            # is a coincidence, not a real match.
            inter = len(shingle_sets[a] & shingle_sets[b])
            if inter < min_shingles:
                if jac >= near_threshold:
                    near_dup_ids.add(a_id)
                    near_dup_ids.add(b_id)
                continue

            # Topic / category agreement when both sides carry the field.
            # Keeps unrelated "current model" status notes from being
            # merged purely because they share boilerplate.
            a_topic = (a_payload.get("topic") or "").strip().lower()
            b_topic = (b_payload.get("topic") or "").strip().lower()
            if a_topic and b_topic and a_topic != b_topic:
                if jac >= near_threshold:
                    near_dup_ids.add(a_id)
                    near_dup_ids.add(b_id)
                continue
            a_cat = (a_payload.get("category") or "").strip().lower()
            b_cat = (b_payload.get("category") or "").strip().lower()
            if a_cat and b_cat and a_cat != b_cat:
                if jac >= near_threshold:
                    near_dup_ids.add(a_id)
                    near_dup_ids.add(b_id)
                continue

            if a_date > b_date:
                supersede_links.append((b_id, a_id))
            else:
                supersede_links.append((a_id, b_id))
        elif jac >= near_threshold:
            near_dup_ids.add(active[a]["id"])
            near_dup_ids.add(active[b]["id"])

    return supersede_links, list(near_dup_ids)


# --- Apply to Qdrant -----------------------------------------------------

def apply_supersedes(
    collection: str,
    links: List[Tuple[object, object]],
    dry_run: bool,
    max_apply: int = SUPERSEDE_MAX_APPLY,
) -> Tuple[int, int]:
    """Write supersede links. Returns ``(applied, dropped_for_cap)``.

    ``dropped_for_cap`` is non-zero when the link list was longer than
    ``max_apply`` and excess links were skipped. ``dry_run`` is uncapped
    so the diagnostic still reports what the algorithm would have done.
    """
    if not links:
        return 0, 0

    if dry_run:
        for old, new in links:
            print(f"[supersede] (dry) {old}  ->  {new}")
        return len(links), 0

    capped_links = links
    dropped = 0
    if len(links) > max_apply:
        capped_links = links[:max_apply]
        dropped = len(links) - max_apply
        print(
            f"[supersede] WARNING: {len(links)} candidate links exceeds "
            f"SUPERSEDE_MAX_APPLY={max_apply}; applying first {max_apply} "
            f"and dropping {dropped}. Raise the cap or run again to "
            f"process the remainder."
        )

    points_payload = []
    for old, new in capped_links:
        points_payload.append({
            "id": old,
            "status": "superseded",
            "superseded_by": new,
            "review_reason": "auto_supersede_content_similarity",
        })
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from _qdrant_helpers import update_payloads as _up
    return _up(collection, points_payload, qdrant_url=QDRANT_URL), dropped


def apply_near_duplicate_tags(collection: str, ids: List[object], dry_run: bool) -> int:
    """Tag points with review_reason: near_duplicate. Does not change status."""
    if not ids:
        return 0
    if dry_run:
        for mid in ids[:10]:
            print(f"[near_dup] (dry) {mid}")
        if len(ids) > 10:
            print(f"[near_dup] (dry) ... and {len(ids) - 10} more")
        return len(ids)

    points_payload = [{"id": mid, "review_reason": "near_duplicate"} for mid in ids]
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from _qdrant_helpers import update_payloads as _up
    return _up(collection, points_payload, qdrant_url=QDRANT_URL)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--recency-gap-days", type=int, default=7)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--enable-topic-supersede",
        action="store_true",
        help="Opt in to the broad keyword-topic supersede pass. Disabled by default.",
    )
    args = p.parse_args(argv)

    points = scroll_all(args.collection)
    print(f"[supersede] loaded {len(points)} points")
    active_count = sum(
        1 for pnt in points if (pnt.get("payload") or {}).get("status") != "superseded"
    )

    topic_enabled = args.enable_topic_supersede or ENABLE_TOPIC_SUPERSEDE

    # Phase 1: keyword-topic-based supersede (opt-in).
    topic_links: List[Tuple[object, object]] = []
    if topic_enabled:
        topic_links = detect_supersedes(points, args.recency_gap_days, enabled=True)
        print(f"[supersede] keyword-topic links: {len(topic_links)} (ENABLE_TOPIC_SUPERSEDE=1)")
    else:
        print(
            "[supersede] keyword-topic links: 0 "
            "(topic pass disabled; set ENABLE_TOPIC_SUPERSEDE=1 to enable)"
        )

    # Phase 2: content-similarity-based supersede + near_duplicate tags.
    if active_count > CONTENT_SUPERSEDE_MAX_POINTS and not FORCE_CONTENT_SUPERSEDE:
        content_links, near_dup_ids = [], []
        print(
            f"[supersede] content pass skipped: active_count={active_count} "
            f"> CONTENT_SUPERSEDE_MAX_POINTS={CONTENT_SUPERSEDE_MAX_POINTS}; "
            "set FORCE_CONTENT_SUPERSEDE=1 for a deliberate deep audit"
        )
    else:
        content_links, near_dup_ids = detect_content_supersedes(points)
    if not ENABLE_AUTO_SUPERSEDE:
        if content_links:
            near_dup_ids = list(set(near_dup_ids) | {pid for pair in content_links for pid in pair})
        content_links = []
        print("[supersede] auto-supersede disabled; content matches will be review tags only")
    print(
        f"[supersede] content-similarity supersede links: {len(content_links)} "
        f"(threshold>={HIGH_CONFIDENCE_JACCARD})"
    )
    print(
        f"[supersede] near-duplicate tags: {len(near_dup_ids)} "
        f"(threshold>={NEAR_DUPLICATE_JACCARD})"
    )

    # Merge all links (dedupe by older_id: first writer wins → topic-based priority).
    seen_old = {old for old, _ in topic_links}
    merged = list(topic_links)
    for old, new in content_links:
        if old not in seen_old:
            merged.append((old, new))
            seen_old.add(old)

    applied, dropped = apply_supersedes(args.collection, merged, args.dry_run)
    if dropped:
        print(f"[supersede] applied {applied} supersede links (dropped {dropped} over SUPERSEDE_MAX_APPLY={SUPERSEDE_MAX_APPLY})")
    else:
        print(f"[supersede] applied {applied} supersede links")

    nd_applied = apply_near_duplicate_tags(args.collection, near_dup_ids, args.dry_run)
    print(f"[supersede] applied {nd_applied} near-duplicate tags")

    return 0


if __name__ == "__main__":
    sys.exit(main())
