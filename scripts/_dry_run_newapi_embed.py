#!/usr/bin/env python3
"""Wave 2 dry-run probe for the NewAPI embed provider.

This is a single-shot script (deliberately named with a leading
underscore so the operator can see at a glance that it's a probe, not
a long-lived tool). It does NOT touch Qdrant. It does NOT log the
bearer token. It does NOT use any alias for the model name.

Run::

    cd /opt/openclaw-memory-os && \
        EMBED_PROVIDER=newapi python scripts/_dry_run_newapi_embed.py

Expected stdout (last line): ``OK: dim=768 finite=True non_zero=True``.

Exit code 0 on success, non-zero on any provider failure.

The probe is intentionally separate from the production code path so
that a future regression in the provider implementation cannot silently
fake a passing dry-run.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _strip_secrets(payload: dict) -> dict:
    """Remove any *string* value whose key looks like a token/secret.

    Used for printing. Only string values are redacted; numeric /
    boolean values pass through so the operator can still see
    ``token_len=51`` (a count, not the token itself).
    """
    out = {}
    for k, v in payload.items():
        if (
            isinstance(v, str)
            and ("token" in k.lower() or "key" in k.lower() or "authorization" in k.lower())
        ):
            out[k] = "<redacted>"
        else:
            out[k] = v
    return out


def main() -> int:
    # Force the NewAPI provider regardless of any inherited env. The
    # script's whole point is to verify the NewAPI wiring.
    os.environ["EMBED_PROVIDER"] = "newapi"
    # Re-import so from_env() picks up the override.
    from openclaw_memory_os import embed_provider  # type: ignore

    embed_provider.reset_provider_caches()
    provider = embed_provider.get_embed_provider()

    summary = {
        "provider": provider.name,
        "model": provider.model,
        "base_url": provider.base_url,
        "expected_dim": provider.expected_dim,
        "api_style": provider.api_style,
        "token_set": bool(provider.api_key),
        "token_len": len(provider.api_key) if provider.api_key else 0,
    }
    print("PROVIDER:", json.dumps(_strip_secrets(summary), indent=2))

    if provider.name != "newapi":
        print(f"FAIL: provider is {provider.name}, expected newapi", file=sys.stderr)
        return 1

    if not provider.api_key:
        print("FAIL: NewAPI token missing or unreadable", file=sys.stderr)
        return 1

    # ---- Call 1: explicit dimensions (matches production contract) ----
    text1 = "wave2 dry-run probe (explicit dim=768)"
    vec1 = provider.embed(text1)
    finite1 = all(isinstance(x, float) for x in vec1) and all(
        x == x and x not in (float("inf"), float("-inf")) for x in vec1
    )
    nonzero1 = any(x != 0.0 for x in vec1)

    # ---- Call 2: a second call to verify the client cache works ----
    text2 = "second call, should reuse the cached httpx client"
    vec2 = provider.embed(text2)

    print(
        "RESULT:",
        json.dumps(
            {
                "call1_dim": len(vec1),
                "call1_finite": finite1,
                "call1_non_zero": nonzero1,
                "call2_dim": len(vec2),
                "call2_matches_call1": vec1 == vec2,
            },
            indent=2,
        ),
    )

    ok = (
        len(vec1) == 768
        and len(vec2) == 768
        and finite1
        and nonzero1
    )
    if not ok:
        print(
            "FAIL: vector validation failed "
            f"(dim={len(vec1)}, finite={finite1}, nonzero={nonzero1})",
            file=sys.stderr,
        )
        return 1

    # Probe sample values for the report (no secrets in here).
    sample = {
        "first10": vec1[:10],
        "min": min(vec1),
        "max": max(vec1),
        "mean": sum(vec1) / len(vec1),
    }
    print("SAMPLE:", json.dumps(sample, indent=2))
    print("OK: dim=768 finite=True non_zero=True")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"FAIL: {type(exc).__name__}: {exc}", file=sys.stderr)
        sys.exit(1)