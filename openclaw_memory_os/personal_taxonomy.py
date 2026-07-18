"""Personal taxonomy loader for OpenClaw Memory OS tier / topic classification.

Brand lists used to live as Python literals inside ``scripts/tier_classifier.py``
and the ingestion module, which leaked the operator's private brand names into
the public repo. That was bad. This module replaces those literals with a
JSON-backed taxonomy loaded from a path governed by the
``MEMORY_OS_TAXONOMY_PATH`` environment variable (default
``config/personal_taxonomy.json``).

The real file is meant to be **gitignored** — operators populate it with their
own brand names. The public repo only ships a placeholder example at
``config/personal_taxonomy.example.json`` so fork operators understand the
shape without leaking their taxonomy.

Design contract:
    * Missing file -> empty dict, single-line warning, classifier still runs.
    * Malformed file -> empty dict, single-line warning, classifier still runs.
    * Empty dict / empty arrays -> classifier rules collapse to "match
      nothing personal" — the public-classifier default behavior.
    * Tier / topic arrays may be either ``list[str]`` or ``list[dict]``
      with a ``"keyword"`` field; the loader normalises both.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Dict, Iterable, List, Union

logger = logging.getLogger(__name__)

DEFAULT_TAXONOMY_PATH = "config/personal_taxonomy.json"


def _coerce_keywords(values: Iterable[object]) -> List[str]:
    """Normalise keyword list entries to ``List[str]``.

    Supports both plain strings and ``{"keyword": "..."}`` dict shapes so the
    schema can grow without breaking existing files. Whitespace-only or
    empty entries are dropped.
    """
    out: List[str] = []
    for v in values:
        if isinstance(v, str):
            stripped = v.strip()
            if stripped:
                out.append(stripped)
        elif isinstance(v, dict):
            kw = v.get("keyword") or v.get("term") or v.get("name")
            if isinstance(kw, str):
                kw = kw.strip()
                if kw:
                    out.append(kw)
        # Anything else: drop silently. The loader must never crash on
        # operator typos in a local config file.
    return out


def _coerce_taxonomy(raw: object) -> Dict[str, List[str]]:
    """Normalise a parsed-JSON object into ``{tier: [keyword, ...]}``."""
    out: Dict[str, List[str]] = {}
    if not isinstance(raw, dict):
        return out
    for tier, values in raw.items():
        if not isinstance(tier, str):
            continue
        if tier.startswith("_"):
            # convention: leading-underscore keys are comments / metadata
            continue
        if isinstance(values, list):
            out[tier] = _coerce_keywords(values)
        # ignore other shapes (numbers, dicts at tier level, strings
        # are also fine to skip - keys are intended to map to lists)
    return out


def load_personal_taxonomy(
    path: Union[str, os.PathLike[str], None] = None,
    *,
    env_var: str = "MEMORY_OS_TAXONOMY_PATH",
) -> Dict[str, List[str]]:
    """Load the operator's personal taxonomy from disk.

    Args:
        path: Explicit override for the taxonomy file location. Wins over
            the environment variable.
        env_var: Name of the environment variable that overrides the
            default location. Defaults to ``MEMORY_OS_TAXONOMY_PATH``.

    Returns:
        Dict mapping tier / topic name to a list of keywords. Empty dict
        if the file is missing, unreadable, malformed, or contains no
        recognised entries. Never returns ``None`` and never raises.

    Notes:
        A single-line warning is logged at WARNING level (not ERROR) the
        first time a problem is encountered, so operators see it once
        per process in their log without spam. The classifier must keep
        working after a warning — tier == "medium" is the safe default.
    """
    if path is None:
        path = os.environ.get(env_var, DEFAULT_TAXONOMY_PATH)

    try:
        p = Path(path)
    except TypeError:
        logger.warning(
            "personal_taxonomy: invalid path %r; using empty taxonomy", path
        )
        return {}

    if not p.is_file():
        # Quiet info-level hint once per process for the common case
        # (default path not present yet). Not a warning — many forks
        # intentionally ship an empty taxonomy.
        logger.info(
            "personal_taxonomy: no file at %s; using empty taxonomy", p
        )
        return {}

    try:
        text = p.read_text(encoding="utf-8")
    except OSError as exc:
        logger.warning(
            "personal_taxonomy: cannot read %s (%s); using empty taxonomy",
            p, exc,
        )
        return {}

    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        logger.warning(
            "personal_taxonomy: %s is malformed JSON (%s); using empty taxonomy",
            p, exc,
        )
        return {}

    taxonomy = _coerce_taxonomy(raw)
    if not taxonomy:
        logger.warning(
            "personal_taxonomy: %s contains no usable tier / topic entries",
            p,
        )
    return taxonomy


def get_tier_keywords(taxonomy: Dict[str, List[str]], tier: str) -> List[str]:
    """Return the keyword list for ``tier`` (case-sensitive). Empty if absent."""
    if not tier:
        return []
    value = taxonomy.get(tier)
    return list(value) if isinstance(value, list) else []


def expand_with_personal(
    base: Iterable[str],
    taxonomy: Dict[str, List[str]],
    *tiers: str,
) -> List[str]:
    """Merge base keyword list with keywords pulled from the taxonomy.

    Returns a deduplicated list preserving base-first ordering. Safe to call
    with empty base, empty taxonomy, or zero tiers.
    """
    seen = set()
    out: List[str] = []
    for item in base:
        if item and item not in seen:
            seen.add(item)
            out.append(item)
    for tier in tiers:
        for kw in get_tier_keywords(taxonomy, tier):
            if kw and kw not in seen:
                seen.add(kw)
                out.append(kw)
    return out


__all__ = [
    "DEFAULT_TAXONOMY_PATH",
    "load_personal_taxonomy",
    "get_tier_keywords",
    "expand_with_personal",
]
