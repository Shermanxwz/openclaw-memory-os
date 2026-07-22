# Final real-host acceptance

Run this once on the production host after `sudo deploy/deploy.sh`. It is the
only gate intentionally left outside GitHub-hosted CI because it requires the
real systemd service, Qdrant corpus, Ollama models, session database, network
latency, and operator secrets.

OpenClaw must first perform two controlled drills and save owner-private JSON
evidence:

* a Qdrant snapshot restore into a **disposable** instance or clone, without
  mutating production collections;
* an isolated evolution drill proving the same candidate passes two windows at
  least ten minutes apart, rollback restores `Previous`, the circuit breaker
  trips, and the original production policy/state is restored afterward.

Restore proof schema:

```json
{
  "status": "passed",
  "environment": "disposable",
  "production_mutated": false,
  "restored_points": 20000,
  "source_snapshot_sha256": "<64 hex characters>",
  "tested_at": "<ISO-8601 timestamp>"
}
```

Evolution proof schema:

```json
{
  "status": "passed",
  "same_candidate_two_windows": true,
  "rollback_to_previous": true,
  "circuit_breaker": true,
  "production_policy_restored": true,
  "tested_at": "<ISO-8601 timestamp>"
}
```

Then run the fail-closed gate with a query known to return at least one real
hit in keyword, dense, and hybrid mode:

```bash
sudo FINAL_ACCEPTANCE_ACK=YES \
  ACCEPTANCE_COLLECTIONS="openclaw_memory_os other_collection" \
  ACCEPTANCE_MIN_POINTS=20000 \
  ACCEPTANCE_QUERY="a known phrase present in the corpus" \
  RESTORE_PROOF_FILE=/var/lib/openclaw-memory-os/acceptance-input/restore-proof.json \
  EVOLUTION_PROOF_FILE=/var/lib/openclaw-memory-os/acceptance-input/evolution-proof.json \
  scripts/final_host_acceptance.sh
```

The two real governance invocations are separated by at least 610 seconds; the
value may be increased with `ACCEPTANCE_GOVERNANCE_GAP_SECONDS` but cannot be
lowered below the product's ten-minute distinct-window contract.

The gate fails closed and writes owner-private evidence below
`/var/lib/openclaw-memory-os/acceptance/<UTC timestamp>/`. It verifies:

1. The web service runs as `openclaw-memory-os`, and both persistent timers are active.
2. A fresh CPython 3.12 environment installs the audited locks and passes the full test suite, compile check, pip check, and privacy scan; a wheel is then built and smoke-installed in a second clean environment.
3. Every named Qdrant collection exists and the combined corpus meets the requested scale.
4. The fixed local models `nomic-embed-text` and `qwen2.5:1.5b` are available.
5. Keyword, dense, and hybrid recall return real hits, qualified identities, policy versions, and diagnostics.
6. The complete host authentication/session restart smoke passes.
7. The varied-query performance gate passes at concurrency 1 and 5.
8. Two real governance runs, separated by the production window gap, persist an honest `ok` result.
9. The disposable restore and controlled evolution evidence satisfies the schemas above.
10. A complete-history gitleaks scan passes; skipping it is not allowed.
11. The exact wheel and `git archive` source bundle are preserved with SHA-256 digests.
12. A CPython 3.12 runtime wheelhouse and its checksum manifest are saved for offline reinstall without future dependency resolution.

The script never invokes a physical-delete path. A successful run ends with
`HOST_ACCEPTANCE_PASSED` and an evidence directory. Preserve that directory; it contains the final source archive, built wheel, CPython 3.12 offline wheelhouse, dependency locks, host proofs, and SHA-256 checksums.
