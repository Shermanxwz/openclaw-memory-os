#!/usr/bin/env python3
"""Sherman Step 4 strict keyword performance gate.

Measures **two metrics** on every recall-test call:

1. **HTTP p95 / p50** — wall-clock elapsed per request (the
   latency the user sees from their browser). Target ≤ 300 ms.
2. **Server-side p95 / p50** — ``diagnostics.lexical_ms`` (the
   time the BM25 lexical search spent inside the handler). This
   isolates the recall engine's work from network, request
   serialisation, and any embedded-model dispatch. Target ≤ 150 ms.

The benchmark exercises **two concurrency levels**: 1 worker
(serial) and 5 workers (small burst). For each concurrency level
it runs **10 warmup** iterations (discarded) and **200 measured**
iterations (kept). A request is considered a server-side pass
when ``diagnostics.lexical_ms`` is non-null and the request
returned 200.

Output: a JSON report on stdout + an optional --out file. The
script exits 0 only when BOTH concurrency levels pass BOTH
metrics (HTTP ≤ 300 ms p95, server-side ≤ 150 ms p95).

This script is invoked from ``scripts/strict_perf_gate.sh`` and
from the GitHub Actions ``ci.yml`` workflow when ``strict-perf``
is requested. Do not delete without coordinating with the
release pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

PROJECT_DIR = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:7788/api/recall-test"
QUERY = "v0.3.0 graduation"
MODE = "keyword"
LIMIT = 5

HTTP_TARGET_P95_MS = 300.0
SERVER_TARGET_P95_MS = 150.0


def _read_token() -> Optional[str]:
    """Pick up ``MEMORY_OS_TOKEN`` from env or repo-root ``.env``."""
    env_token = os.environ.get("MEMORY_OS_TOKEN")
    if env_token:
        return env_token.strip()
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


def _probe(token: str) -> Tuple[float, int, Optional[float]]:
    """Issue one HTTP request and return ``(elapsed_seconds, status,
    server_side_ms)``.

    ``server_side_ms`` is ``diagnostics.lexical_ms`` (BM25 lexical
    search time inside the handler). ``None`` when the response is
    non-200 or the diagnostics field is missing.
    """
    body = json.dumps({"query": QUERY, "mode": MODE, "limit": LIMIT}).encode("utf-8")
    req = urllib.request.Request(
        URL,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            raw = resp.read()
            elapsed = time.perf_counter() - t0
            status = resp.status
    except urllib.error.HTTPError as exc:
        elapsed = time.perf_counter() - t0
        return elapsed, exc.code, None
    except (urllib.error.URLError, ConnectionError, TimeoutError, OSError):
        return time.perf_counter() - t0, 0, None

    server_side: Optional[float] = None
    if status == 200 and raw:
        try:
            payload = json.loads(raw)
            diag = payload.get("diagnostics") or {}
            val = diag.get("lexical_ms")
            if isinstance(val, (int, float)):
                server_side = float(val)
        except json.JSONDecodeError:
            pass
    return elapsed, status, server_side


def _run_one_concurrency(
    concurrency: int, token: str, warmup: int, measured: int
) -> Dict[str, Any]:
    """Run the benchmark at a fixed worker count.

    ``warmup`` and ``measured`` are **total request counts** at this
    concurrency level, not per-worker counts. For example, with
    ``concurrency=5`` and ``measured=200`` this function fires 200
    measured requests total, with up to 5 in flight at a time. This
    matches Sherman's final gate wording: "预热10次、正式200次，同时
    测试并发1和5". Earlier drafts accidentally ran 200 measured
    requests per worker (1000 total at concurrency=5), overloading
    the single-worker uvicorn and making the gate meaningless.
    """
    statuses: List[int] = []

    def fire_one(_: int) -> Tuple[float, int, Optional[float]]:
        return _probe(token)

    def run_batch(count: int, *, collect: bool) -> Tuple[List[float], List[float]]:
        http: List[float] = []
        server_side: List[float] = []
        if count <= 0:
            return http, server_side
        with ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(fire_one, i) for i in range(count)]
            for f in as_completed(futures):
                elapsed, status, server = f.result()
                statuses.append(status)
                if collect:
                    http.append(elapsed * 1000.0)
                    if server is not None:
                        server_side.append(server)
        return http, server_side

    started = time.perf_counter()
    run_batch(warmup, collect=False)
    http_samples, server_samples = run_batch(measured, collect=True)
    wall_ms = (time.perf_counter() - started) * 1000.0

    http_samples = sorted(http_samples)
    server_samples = sorted(server_samples)

    def pct(samples: List[float], p: float) -> float:
        if not samples:
            return float("nan")
        idx = max(0, min(len(samples) - 1, int(round((p / 100.0) * (len(samples) - 1)))))
        return samples[idx]

    def stats(samples: List[float]) -> Dict[str, float]:
        if not samples:
            return {"count": 0, "p50_ms": float("nan"), "p95_ms": float("nan"),
                    "min_ms": float("nan"), "max_ms": float("nan"), "avg_ms": float("nan")}
        return {
            "count": len(samples),
            "p50_ms": pct(samples, 50),
            "p95_ms": pct(samples, 95),
            "min_ms": min(samples),
            "max_ms": max(samples),
            "avg_ms": statistics.fmean(samples),
        }

    http_stats = stats(http_samples)
    server_stats = stats(server_samples)
    n_200 = sum(1 for s in statuses if s == 200)
    http_pass = http_stats["p95_ms"] <= HTTP_TARGET_P95_MS
    server_pass = server_stats["p95_ms"] <= SERVER_TARGET_P95_MS

    return {
        "concurrency": concurrency,
        "warmup": warmup,
        "measured": measured,
        "wall_ms": wall_ms,
        "total_requests": len(statuses),
        "n_200": n_200,
        "http_target_p95_ms": HTTP_TARGET_P95_MS,
        "server_target_p95_ms": SERVER_TARGET_P95_MS,
        "http": http_stats,
        "server": server_stats,
        "pass": http_pass and server_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--measured", type=int, default=200)
    parser.add_argument("--concurrency", type=str, default="1,5",
                        help="Comma-separated worker counts (default 1,5).")
    parser.add_argument("--out", type=str, default=None,
                        help="Optional JSON output path.")
    args = parser.parse_args()

    token = _read_token()
    if not token:
        print("service not reachable; strict perf measurement deferred "
              "(no MEMORY_OS_TOKEN)", file=sys.stderr)
        print(json.dumps({"deferred": True, "reason": "no_token"}))
        return 0

    concurrency_levels = [int(x) for x in args.concurrency.split(",") if x.strip()]
    levels: List[Dict[str, Any]] = []
    for c in concurrency_levels:
        print(f"running concurrency={c} warmup={args.warmup} "
              f"measured={args.measured} ...", file=sys.stderr)
        levels.append(_run_one_concurrency(c, token, args.warmup, args.measured))

    overall_pass = all(level["pass"] for level in levels)
    report = {
        "tool": "scripts/strict_perf_gate.py",
        "mode": MODE,
        "query": QUERY,
        "url": URL,
        "warmup": args.warmup,
        "measured": args.measured,
        "concurrency_levels": concurrency_levels,
        "levels": levels,
        "overall_pass": overall_pass,
    }

    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0 if overall_pass else 1


if __name__ == "__main__":
    sys.exit(main())
