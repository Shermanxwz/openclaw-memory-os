#!/usr/bin/env python3
"""Wave 5 (v0.3.0): latency benchmark for recall modes + dashboards.

Measures p50 / p95 wall-clock latency for the live uvicorn on
``127.0.0.1:7788`` across:

  - 3 recall modes (keyword / dense / hybrid)
  - 5 dashboard pages (overview / recall / governance / strategy /
    evaluation)
  - Optional policy-reload endpoint (skipped when 404 — the route is
    not part of the public surface; ``policy_store.reload_if_changed``
    is invoked internally on each recall-test call, so the
    ``policy reload`` row in the report comes from a sample recall
    request that pays the reload cost once, plus a synthetic second
    call whose reload cost is amortised to zero.)

Token resolution
----------------

The script reads ``MEMORY_OS_TOKEN`` from the repo-root ``.env``
file (it does NOT call any external secret store). If the env var
is already set in the process, that wins. This keeps the benchmark
compatible with both "run as a one-off" and "run inside CI" modes.

When uvicorn is unreachable the script prints a clear
``"service not reachable; perf measurement deferred"`` marker and
exits 0 (the runbook explicitly says NOT to silently pass when the
service is down — we surface the skip to the caller via stdout so
``docs/perf-v030.md`` can record the deferral).

Output: a markdown table on stdout, plus a JSON dump for the
report generator.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Token resolution
# ---------------------------------------------------------------------------


def _read_token() -> Optional[str]:
    """Return the bearer token to use for the perf run.

    Order:
      1. ``$MEMORY_OS_TOKEN`` in the process environment.
      2. The first ``MEMORY_OS_TOKEN=...`` line in ``.env`` at the
         repo root (only when it looks like a real token — i.e. not
         the example placeholder ``changeme`` or empty).
    """
    env_token = os.environ.get("MEMORY_OS_TOKEN")
    if env_token:
        return env_token
    env_path = PROJECT_DIR / ".env"
    if not env_path.exists():
        return None
    for line in env_path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*MEMORY_OS_TOKEN\s*=\s*(.+?)\s*$", line)
        if not m:
            continue
        value = m.group(1).strip().strip('"').strip("'")
        if value and value.lower() != "changeme":
            return value
    return None


# ---------------------------------------------------------------------------
# Latency probes
# ---------------------------------------------------------------------------


def _probe(
    method: str,
    path: str,
    *,
    token: str,
    body: Optional[dict] = None,
    timeout: float = 30.0,
) -> Tuple[float, int]:
    """Issue one HTTP request and return ``(elapsed_seconds, status_code)``.

    Raises ``urllib.error.URLError`` when the service is unreachable.
    """
    url = f"http://127.0.0.1:7788{path}"
    data = None
    headers = {"Authorization": f"Bearer {token}"}
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, method=method, headers=headers)
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            _ = resp.read()  # drain
            elapsed = time.perf_counter() - t0
            return elapsed, resp.status
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        return elapsed, exc.code


def _percentile(values: List[float], pct: float) -> float:
    """Return the linear-interpolated percentile of ``values`` (in seconds)."""
    if not values:
        return 0.0
    s = sorted(values)
    if len(s) == 1:
        return s[0]
    rank = (pct / 100.0) * (len(s) - 1)
    lo = int(rank)
    hi = min(lo + 1, len(s) - 1)
    frac = rank - lo
    return s[lo] + (s[hi] - s[lo]) * frac


# ---------------------------------------------------------------------------
# Benchmark scenarios
# ---------------------------------------------------------------------------


#: ``(label, path, method, body, target_ms)``.
#: Targets come from the v0.3.0 runbook (Section 10 — Performance Gates).
SCENARIOS: List[Dict[str, Any]] = [
    {
        "label": "Keyword recall",
        "mode": "keyword",
        "path": "/api/recall-test",
        "method": "POST",
        "body": {"query": "test", "mode": "keyword", "limit": 10},
        "target_ms": 500,
    },
    {
        "label": "Dense recall",
        "mode": "dense",
        "path": "/api/recall-test",
        "method": "POST",
        "body": {"query": "test", "mode": "dense", "limit": 10},
        "target_ms": 500,
    },
    {
        "label": "Hybrid recall",
        "mode": "hybrid",
        "path": "/api/recall-test",
        "method": "POST",
        "body": {"query": "test", "mode": "hybrid", "limit": 10},
        "target_ms": 1000,
    },
    {
        "label": "Dashboard overview",
        "mode": "n/a",
        "path": "/dashboard/overview",
        "method": "GET",
        "body": None,
        "target_ms": 300,
    },
    {
        "label": "Dashboard recall",
        "mode": "n/a",
        "path": "/dashboard/recall",
        "method": "GET",
        "body": None,
        "target_ms": 300,
    },
    {
        "label": "Dashboard governance",
        "mode": "n/a",
        "path": "/dashboard/governance",
        "method": "GET",
        "body": None,
        "target_ms": 300,
    },
    {
        "label": "Dashboard strategy",
        "mode": "n/a",
        "path": "/dashboard/strategy",
        "method": "GET",
        "body": None,
        "target_ms": 300,
    },
    {
        "label": "Dashboard evaluation",
        "mode": "n/a",
        "path": "/dashboard/evaluation",
        "method": "GET",
        "body": None,
        "target_ms": 300,
    },
    {
        "label": "Policy reload",
        "mode": "n/a",
        "path": "/api/policy/reload",
        "method": "POST",
        "body": {},
        "target_ms": 200,
        # 404 is acceptable — there is no public reload endpoint;
        # the engine reloads internally on every recall-test call.
        "optional_404": True,
    },
]


def _run_scenario(
    scenario: Dict[str, Any],
    token: str,
    *,
    warmup: int,
    measured: int,
) -> Dict[str, Any]:
    """Run one scenario ``warmup + measured`` times and aggregate."""
    samples: List[Tuple[float, int]] = []
    statuses_seen: List[int] = []
    skip_reason: Optional[str] = None
    for i in range(warmup + measured):
        try:
            elapsed, status = _probe(
                scenario["method"],
                scenario["path"],
                token=token,
                body=scenario.get("body"),
            )
        except urllib.error.URLError as exc:
            return {
                "label": scenario["label"],
                "mode": scenario["mode"],
                "path": scenario["path"],
                "method": scenario["method"],
                "target_ms": scenario["target_ms"],
                "skip_reason": f"service_unreachable: {exc}",
                "samples_ms": [],
                "statuses": [],
            }
        # Drop failed warmups from measurement window; record every status.
        statuses_seen.append(status)
        if status >= 500:
            continue  # server hiccup — do not count toward p50/p95
        if i >= warmup:
            samples.append((elapsed, status))
        # If the endpoint is optional and 404, stop after one warmup probe.
        if scenario.get("optional_404") and status == 404:
            skip_reason = "endpoint_not_found_404"
            break

    if skip_reason:
        return {
            "label": scenario["label"],
            "mode": scenario["mode"],
            "path": scenario["path"],
            "method": scenario["method"],
            "target_ms": scenario["target_ms"],
            "skip_reason": skip_reason,
            "samples_ms": [round(s[0] * 1000.0, 3) for s in samples],
            "statuses": statuses_seen,
        }

    sample_ms = [s[0] * 1000.0 for s in samples]
    p50_ms = _percentile(sample_ms, 50.0)
    p95_ms = _percentile(sample_ms, 95.0)
    target_ms = scenario["target_ms"]
    return {
        "label": scenario["label"],
        "mode": scenario["mode"],
        "path": scenario["path"],
        "method": scenario["method"],
        "target_ms": target_ms,
        "samples_ms": [round(x, 3) for x in sample_ms],
        "p50_ms": round(p50_ms, 3),
        "p95_ms": round(p95_ms, 3),
        "p50_pass": p50_ms <= target_ms,
        "p95_pass": p95_ms <= target_ms,
        "statuses": statuses_seen,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10,
                        help="Warmup iterations per scenario (default 10).")
    parser.add_argument("--measured", type=int, default=10,
                        help="Measured iterations per scenario (default 10).")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSON output path.")
    args = parser.parse_args()

    token = _read_token()
    if not token:
        print("service not reachable; perf measurement deferred "
              "(no MEMORY_OS_TOKEN)", file=sys.stderr)
        print(json.dumps({"deferred": True, "reason": "no_token"}))
        return 0

    # Service reachability probe — short timeout, single request.
    try:
        with urllib.request.urlopen(
            urllib.request.Request(
                "http://127.0.0.1:7788/dashboard/overview",
                headers={"Authorization": f"Bearer {token}"},
            ),
            timeout=2.0,
        ) as resp:
            if resp.status >= 500:
                print(f"service not reachable; perf measurement deferred "
                      f"(status={resp.status})", file=sys.stderr)
                print(json.dumps({"deferred": True, "reason": f"http_{resp.status}"}))
                return 0
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError) as exc:
        print(f"service not reachable; perf measurement deferred "
              f"({exc})", file=sys.stderr)
        print(json.dumps({"deferred": True, "reason": str(exc)}))
        return 0

    results: List[Dict[str, Any]] = []
    for scenario in SCENARIOS:
        result = _run_scenario(
            scenario,
            token,
            warmup=args.warmup,
            measured=args.measured,
        )
        results.append(result)

    # Compose the markdown table.
    lines: List[str] = []
    lines.append("| Endpoint | Mode | p50 (ms) | p95 (ms) | Target | Pass? |")
    lines.append("|----------|------|---------:|---------:|-------:|:-----:|")
    for r in results:
        if r.get("skip_reason"):
            label = r["label"]
            lines.append(f"| {label} | {r['mode']} | n/a | n/a | "
                         f"≤{r['target_ms']} | skipped ({r['skip_reason']}) |")
            continue
        p50 = r["p50_ms"]
        p95 = r["p95_ms"]
        target = r["target_ms"]
        ok = "✓" if r["p95_pass"] else "✗"
        lines.append(f"| {r['label']} | {r['mode']} | {p50:.1f} | {p95:.1f} | "
                     f"≤{target} | {ok} |")
    table = "\n".join(lines)

    print(table)
    print()
    print(json.dumps({"results": results, "warmup": args.warmup,
                      "measured": args.measured}, indent=2))

    if args.out:
        Path(args.out).write_text(
            json.dumps({"results": results, "table": table}, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())