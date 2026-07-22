#!/usr/bin/env python3
"""Auto-classify tier, importance, type, and topic for every point in a
Qdrant collection. Writes back via Qdrant ``set_payload`` / upsert.

v0.2.2 OpenClaw Memory OS graduation release — this runs automatically as
part of ``scripts/maintenance.sh``. Default behaviour is **apply** (writes
to Qdrant). Pass ``--dry-run`` for a read-only preview.

Tier rules (mirrors ``ingest_memory.py`` so post-ingest reclassification
matches):
    core    - "core_keyword_a", "core_keyword_b", "core_keyword_c"
              plus any extras from ``config/personal_taxonomy.json`` (tier=core).
    long    - "## 教训", "教训库", "重要教训"
              plus any extras from the taxonomy (tier=long).
    working - keywords like "in_progress", "CI 状态", "临时"
    short   - memory/YYYY-XX-XX.md and small chunk
    medium  - default

The core / long keyword lists (and the brand list used by the ``amazon``
topic) used to be hard-coded Python literals, which leaked personal brand
names into the public repo. They are now loaded from a gitignored JSON
file at ``config/personal_taxonomy.json`` (overridable via the
``MEMORY_OS_TAXONOMY_PATH`` env var). See
``config/personal_taxonomy.example.json`` for the public-safe template and
``docs/recall-ranking.md`` for details.

Importance rules:
    base 0.5, +0.4 if any of the core/long phrases match, +0.1 if "教训"
    near the top, -0.2 if total text is shorter than 200 chars.

Type rules (v0.2.2):
    rule          - contains 铁律 / 必须 / 永不删 / 默认
    decision      - contains 决定 / 决定 / 改成 / 切换
    lesson        - contains 教训 / 失败 / 注意
    fact          - default
    status        - matches "当前" + 模型/worker/端口/IP
    config        - contains 端口 / 域名 / Nginx / FRP

Topic rules (v0.2.2):
    Look for source file path keywords, then in-text headers. Brand
    keywords for the ``amazon`` topic come from
    ``config/personal_taxonomy.json`` (tier=amazon) — never from Python
    literals.

owner_confirmed (v0.2.2):
    True only for ``tier=core`` or ``tier=long`` memories, OR if the
    source label is ``MEMORY.md``. Otherwise stays False.

This is intentionally conservative: it writes only when the classifier is
confident, otherwise it keeps the existing value.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import urllib.request
from typing import List, Optional, Tuple

from openclaw_memory_os.personal_taxonomy import (
    load_personal_taxonomy,
    expand_with_personal,
)

logger = logging.getLogger(__name__)

QDRANT_URL = os.environ.get("QDRANT_URL", "http://127.0.0.1:6333")
DEFAULT_COLLECTION = os.environ.get("QDRANT_COLLECTION", "openclaw_memory_os")

# Base (public-safe) keyword lists. Brand-specific extras are loaded from
# the operator's gitignored config/personal_taxonomy.json at startup. The
# public repo must NEVER embed real brand names; everything operator-
# specific lives in the taxonomy file.
CORE_BASE = ["core_keyword_a", "core_keyword_b", "core_keyword_c"]
LONG_BASE = ["## 教训", "教训库", "重要教训"]
WORKING_BASE = ["in_progress", "CI 状态", "临时", "running"]

# The ``amazon`` topic base keywords remain generic / public-safe. Operator-
# specific brand names (which used to be embedded here) now live in the
# operator's gitignored taxonomy file under the "amazon" key.
AMAZON_BASE = ["amazon_topic_keyword", "Amazon", "amazon_practice_keyword"]

TYPE_RULES = [
    ("rule",     ["铁律", "必须", "永不删", "默认", "铁律：", "Hard rule"]),
    ("decision", ["决定", "改成", "切换", "替换", "采用"]),
    ("lesson",   ["教训", "失败", "注意", "翻车", "踩坑"]),
    ("status",   ["当前模型", "当前 worker", "当前 OpenClaw", "端口"]),
    ("config",   ["端口", "域名", "Nginx", "FRP", "systemd", "DNS"]),
]


def _build_topic_keywords(taxonomy) -> dict:
    """Build the topic-keyword map, merging public base words with the
    operator's taxonomy file. Brand-specific entries for ``amazon`` come
    from the taxonomy; everything else uses public-safe defaults.
    """
    return {
        "memory_system":   ["记忆", "Qdrant", "Memory OS", "ingest"],
        "infrastructure":  ["OVH", "Oracle", "RackNerd", "天翼云", "Nginx", "FRP", "VPS"],
        "amazon":          expand_with_personal(AMAZON_BASE, taxonomy, "amazon"),
        "ai_agents":       ["worker", "M3", "GPT", "DeepSeek", "agent", "主控"],
        # ``personal`` and ``finance`` are operator-controlled topics.
        # Fork operators populate them in their gitignored
        # ``config/personal_taxonomy.json`` under the matching key; the
        # loader will merge those keywords with this empty base so the
        # public repo can stay free of operator-specific data.
        "personal":        expand_with_personal([], taxonomy, "personal"),
        "finance":         expand_with_personal([], taxonomy, "finance"),
    }


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


def score_importance(text: str, source: str, taxonomy=None) -> float:
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


def classify_type(text: str) -> Optional[str]:
    head = text[:300]
    for t, keys in TYPE_RULES:
        if any(k in head for k in keys):
            return t
    return None


def classify_topic(text: str, source: str, taxonomy=None) -> Optional[str]:
    taxonomy = taxonomy if taxonomy is not None else load_personal_taxonomy()
    head = text[:400] + " " + source
    for topic, keys in _build_topic_keywords(taxonomy).items():
        if any(k in head for k in keys):
            return topic
    return None


def infer_owner_confirmed(tier: str, source: str) -> Optional[bool]:
    if tier in ("core", "long"):
        return True
    if "MEMORY.md" in source:
        return True
    return None  # leave unchanged


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


def build_updates(points: List[dict], taxonomy=None) -> Tuple[List[dict], dict]:
    """Compute updates + return a stats dict for the summary."""
    if taxonomy is None:
        taxonomy = load_personal_taxonomy()

    updates: List[dict] = []
    stats = {
        "scanned":  len(points),
        "changed":  0,
        "by_field": {"tier": 0, "importance": 0, "type": 0, "topic": 0, "owner_confirmed": 0},
    }
    for pt in points:
        payload = pt.get("payload") or {}
        text = payload.get("content") or payload.get("text") or ""
        source = payload.get("source") or ""

        new_tier = classify_tier(text, source, taxonomy)
        new_imp  = round(score_importance(text, source, taxonomy), 4)
        new_type = classify_type(text)
        new_topic = classify_topic(text, source, taxonomy)
        new_owner = infer_owner_confirmed(new_tier, source)

        u = {"id": pt["id"]}
        changed = False
        if payload.get("tier") != new_tier:
            u["tier"] = new_tier
            stats["by_field"]["tier"] += 1
            changed = True
        if abs(float(payload.get("importance") or 0.5) - new_imp) > 1e-3:
            u["importance"] = new_imp
            stats["by_field"]["importance"] += 1
            changed = True
        # type: only fill if currently None
        if new_type and payload.get("type") in (None, ""):
            u["type"] = new_type
            stats["by_field"]["type"] += 1
            changed = True
        # topic: only fill if currently None
        if new_topic and payload.get("topic") in (None, ""):
            u["topic"] = new_topic
            stats["by_field"]["topic"] += 1
            changed = True
        # owner_confirmed: only fill if currently None
        if new_owner is True and not payload.get("owner_confirmed"):
            u["owner_confirmed"] = True
            stats["by_field"]["owner_confirmed"] += 1
            changed = True
        if changed:
            stats["changed"] += 1
            updates.append(u)
    return updates, stats


def apply_updates(collection: str, updates: List[dict], dry_run: bool) -> int:
    if not updates:
        return 0
    if dry_run:
        for u in updates[:5]:
            fields = {k: v for k, v in u.items() if k != "id"}
            print(f"[tier] (dry) {u['id']} -> {fields}")
        if len(updates) > 5:
            print(f"[tier] (dry) ... {len(updates) - 5} more")
        return len(updates)

    import sys as _sys
    from pathlib import Path as _P
    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    from _qdrant_helpers import update_payloads as _up
    return _up(collection, updates, qdrant_url=QDRANT_URL)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description="OpenClaw Memory OS auto-classifier")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--dry-run", action="store_true", help="preview without writing")
    p.add_argument("--collections", nargs="+", help="multiple collections")
    args = p.parse_args(argv)

    taxonomy = load_personal_taxonomy()
    logger.info(
        "[tier] personal taxonomy loaded: keys=%s",
        sorted(taxonomy.keys()) or "(empty)",
    )

    collections = args.collections or [args.collection]
    grand_total = {"scanned": 0, "changed": 0, "applied": 0,
                   "by_field": {"tier": 0, "importance": 0, "type": 0, "topic": 0, "owner_confirmed": 0}}
    for coll in collections:
        print(f"[tier] === collection={coll} ===")
        points = scroll_all(coll)
        print(f"[tier] loaded {len(points)} points")
        updates, stats = build_updates(points, taxonomy)
        print(f"[tier] changed: {stats['changed']} of {len(points)} by_field={stats['by_field']}")
        applied = apply_updates(coll, updates, args.dry_run)
        print(f"[tier] applied: {applied}")
        grand_total["scanned"] += stats["scanned"]
        grand_total["changed"] += stats["changed"]
        grand_total["applied"] += applied
        for k, v in stats["by_field"].items():
            grand_total["by_field"][k] += v
    print("[tier] === grand total ===")
    print(f"[tier] scanned={grand_total['scanned']} changed={grand_total['changed']} applied={grand_total['applied']}")
    print(f"[tier] by_field={grand_total['by_field']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
