#!/usr/bin/env python3
"""Auto-expire tier=working memories that are older than N days.

Hard rules (must NOT change, per Memory OS retention policy):
    * Never touch tier=core or tier=long.
    * Never touch ``status=active`` if ``never_delete`` is True.
    * Never touch anything with importance >= 0.5.
    * Expire only tier=working AND importance < 0.3 AND age >= 30 days.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import urllib.request
from datetime import datetime, timezone
from typing import List

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "openclaw_memory_os")

MAX_IMPORTANCE = 0.3
MIN_AGE_DAYS = 30


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


def _age_days(source: str, now: datetime) -> int:
    if not source:
        return 0
    m = re.search(r"(20\d{2})-(\d{2})-(\d{2})", source)
    if not m:
        return 0
    try:
        d = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
    except ValueError:
        return 0
    return max(0, int((now - d).total_seconds() // 86400))


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--min-age-days", type=int, default=MIN_AGE_DAYS)
    p.add_argument("--max-importance", type=float, default=MAX_IMPORTANCE)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)

    now = datetime.now(timezone.utc)
    points = scroll_all(args.collection)
    print(f"[expire] loaded {len(points)} points")
    candidates: List[dict] = []
    for pt in points:
        payload = pt.get("payload") or {}
        if payload.get("status") == "expired":
            continue
        if payload.get("never_delete") is True:
            continue
        tier = payload.get("tier") or ""
        if tier in ("core", "long"):
            continue
        if tier != "working":
            continue
        try:
            importance = float(payload.get("importance") or 0.5)
        except (TypeError, ValueError):
            importance = 0.5
        if importance >= args.max_importance:
            continue
        age = _age_days(payload.get("source") or "", now)
        if age < args.min_age_days:
            continue
        # Pass raw id through; ``update_payloads`` routes it through
        # ``coerce_point_id`` so integer IDs reach Qdrant as native
        # ints (the P0 fix for writeback 400s).
        candidates.append({
            "id": pt["id"],
            "age_days": age,
            "importance": importance,
            "tier": tier,
            "source": payload.get("source"),
        })

    print(f"[expire] candidates: {len(candidates)}")
    for c in candidates[:5]:
        print(f"  {c['id']}  age={c['age_days']}d  imp={c['importance']}  tier={c['tier']}  source={c['source']}")
    if len(candidates) > 5:
        print(f"  ... and {len(candidates) - 5} more")

    if args.dry_run or not candidates:
        return 0

    body = {"points": [{
        "id": c["id"],
        "status": "expired",
        "review_reason": "auto_expire_working_old"
    } for c in candidates]}
    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from _qdrant_helpers import update_payloads as _up
    _up(args.collection, body["points"], qdrant_url=QDRANT_URL)
    print(f"[expire] expired {len(candidates)} points")
    return 0


if __name__ == "__main__":
    sys.exit(main())