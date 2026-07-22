"""Shared Qdrant write helpers used by the OS maintenance scripts.

Why this module exists:
    Qdrant 1.18 has a known regression where the /points/payload
    set_payload endpoint returns 400 ("not a valid point ID") for UUID
    string IDs that the /points scroll endpoint happily returns. To
    avoid that, this module issues a full upsert (PUT /points) instead
    of set_payload. We fetch each point's existing vector first so the
    upsert does not silently zero-out the embedding.

ID handling:
    Qdrant accepts UUIDs and signed/unsigned integers as native point
    IDs. When a non-UUID numeric string is sent as a string, the
    writeback endpoints reject it with HTTP 400 ("not a valid point
    ID"). :func:`coerce_point_id` is the single chokepoint that ensures
    we send integer IDs as native ints and let UUIDs / opaque strings
    pass through unchanged. All helpers below route their IDs through
    it so a caller can pass either form safely.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from typing import List, Union

QDRANT_URL_DEFAULT = "http://127.0.0.1:6333"

# Qdrant uses 64-bit signed/unsigned ints; see
# https://qdrant.tech/documentation/concepts/points/#point-ids
# We only need to recognize decimal integers; anything else (UUIDs,
# ULIDs, prefixed IDs like "mem-0007", hashes) stays a string.
_INT_RE = re.compile(r"^-?\d+$")


def coerce_point_id(point_id) -> Union[int, str]:
    """Return a point ID in the native form Qdrant expects.

    Rules:
        * ``int`` (or anything that is already an ``int`` and not a
          ``bool``) -> returned unchanged.
        * ``str`` that is a pure decimal integer (with optional sign)
          -> returned as ``int``. This is the P0 fix: previously the
          helpers stringified integer IDs, which Qdrant then rejected.
        * any other value (UUID, prefixed ID, ``None``) -> returned as
          a string untouched.
        * ``bool`` is excluded from the int path because Python treats
          ``bool`` as a subclass of ``int`` and silently converting
          ``True``/``False`` to ``1``/``0`` would corrupt payloads.
    """
    if isinstance(point_id, bool):
        return str(point_id)
    if isinstance(point_id, int):
        return point_id
    if isinstance(point_id, str):
        if _INT_RE.match(point_id):
            try:
                return int(point_id)
            except ValueError:
                # Pathological; fall through to string return.
                pass
        return point_id
    return str(point_id)


def _post(url: str, body: dict, timeout: int = 60):
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_point(collection: str, point_id, qdrant_url: str = QDRANT_URL_DEFAULT, with_vector: bool = True):
    """Return a single point dict, or None if missing.

    ``point_id`` is coerced through :func:`coerce_point_id` so callers
    can pass int or str interchangeably.
    """
    pid = coerce_point_id(point_id)
    body = {"ids": [pid], "with_payload": True, "with_vector": with_vector}
    headers = {"Content-Type": "application/json"}
    # Forward api-key for protected clusters.
    api_key = os.environ.get("QDRANT_API_KEY")
    if api_key:
        headers["api-key"] = api_key
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                f"{qdrant_url}/collections/{collection}/points",
                data=json.dumps(body).encode("utf-8"),
                headers=headers,
                method="POST",
            ),
            timeout=30,
        ) as r:
            data = json.loads(r.read())
        items = data.get("result") or []
        return items[0] if items else None
    except urllib.error.HTTPError as exc:
        # Don't silently swallow 400s. The caller may have misrouted an
        # ID type (e.g. UUID-shaped id field that was actually a
        # prefixed str) and we want a clean trail.
        try:
            detail = exc.read()[:200].decode("utf-8", "replace")
        except Exception:
            detail = "<no body>"
        print(f"[qdrant] get_point {pid!r} in {collection!r}: HTTP {exc.code} {detail!r}")
        return None
    except Exception as exc:
        print(f"[qdrant] get_point error for {pid!r}: {exc}")
        return None


def update_payloads(
    collection: str,
    updates: List[dict],
    qdrant_url: str = QDRANT_URL_DEFAULT,
    sleep_between: float = 0.0,
) -> int:
    """Update payload fields on a list of points via upsert.

    Each update is a dict with keys ``id`` and the fields to merge into
    ``payload``. The function fetches each point's current payload and
    vector, merges the new fields, and re-upserts the point. Point IDs
    are coerced through :func:`coerce_point_id` so callers can pass int
    or str interchangeably.
    """
    written = 0
    api_key = os.environ.get("QDRANT_API_KEY")
    for u in updates:
        pid_raw = u.get("id")
        if pid_raw is None:
            continue
        pid = coerce_point_id(pid_raw)
        point = get_point(collection, pid, qdrant_url=qdrant_url, with_vector=True)
        if not point:
            continue
        payload = dict(point.get("payload") or {})
        for k, v in u.items():
            if k == "id":
                continue
            payload[k] = v
        vec = point.get("vector")
        upsert_body = {"points": [{"id": pid, "vector": vec, "payload": payload}]}
        try:
            headers = {"Content-Type": "application/json"}
            if api_key:
                headers["api-key"] = api_key
            req = urllib.request.Request(
                f"{qdrant_url}/collections/{collection}/points?wait=true",
                data=json.dumps(upsert_body).encode("utf-8"),
                headers=headers,
                method="PUT",
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                r.read()
            written += 1
        except urllib.error.HTTPError as exc:
            # Concise log: trim Qdrant's verbose JSON body to the
            # human-readable status string. This is the P0 fix for
            # writeback not failing silently.
            try:
                body_text = exc.read()[:200].decode("utf-8", "replace")
            except Exception:
                body_text = "<no body>"
            print(
                f"[qdrant] upsert failed for {pid_raw!r} (coerced={pid!r}) in "
                f"{collection!r}: HTTP {exc.code} {body_text!r}"
            )
        except Exception as exc:
            print(f"[qdrant] upsert error for {pid_raw!r}: {exc}")
        if sleep_between:
            time.sleep(sleep_between)
    return written
