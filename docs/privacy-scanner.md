# Privacy scanner

The repo ships a tiny in-house privacy scanner at
`openclaw_memory_os/privacy.py`. It is a fast first-pass safety net. It is
**not** a replacement for a dedicated secret scanner like `gitleaks` (which
the CI runs in addition).

## What it scans

File extensions:

```
.py  .md  .txt  .json  .yml  .yaml  .toml
.html  .j2  .sh  .cfg  .ini  .env
```

Anything `.git/` is always skipped. Files named
`sample_memories.json` or `privacy_baseline.json` are skipped by default
(they ship in-repo, fully audited).

## Rules

| Rule ID         | Pattern |
| --------------- | --- |
| PRIVATE_HOSTNAME | `vps-[a-f0-9]{8,}` (case-insensitive) |
| MEMORY_OS_PATH   | `/<project-root>`  privacy-allow: MEMORY_OS_PATH |
| PROVIDER_ID      | `\b(new-api-\d{3,}\|internal-(?:[a-f0-9]{4,}\|.*\d.*)\b\|model-aware-worker)\b`  privacy-allow: PROVIDER_ID |
| OPENAI_KEY       | `\bsk-[A-Za-z0-9_-]{20,}` or `sk-ant-...` |
| GITHUB_TOKEN     | `\bghp_[A-Za-z0-9]{30,}` |
| JWT_BEARER       | `Bearer eyJ...` (3 dot-separated base64urls) |
| QQ_ACCOUNT       | `\bQQ[:\s]+\d{6,12}\b` |
| IPV4             | Any IPv4 address that is **not** in the loopback / private / link-local / multicast reserved ranges (`127/8`, `0/8`, `10/8`, `172.16/12`, `192.168/16`, `169.254/16`, `224/4`, `240/4`, `255/8`). |

False positives by design: the scanner errs on the side of "report". A
suppression mechanism is the primary way to clear them.

## Suppression

Two equally good mechanisms. Pick whichever fits the case.

### Per-line marker

Append to a flagged line:

```text
... /<project-root> ... privacy-allow: MEMORY_OS_PATH
```

`privacy-allow: *` suppresses every rule on that line. The marker only
applies to the **specific rule named** (or wildcard), so an unrelated
secret on the same line is still caught.

### Baseline JSON

For multi-line blocks (e.g. a JSON fixture with a quoted credential
example), put exact matches in a baseline file:

```json
{
  "findings": [
    {"file": "examples/aws-credentials.md", "line": 8, "rule_id": "OPENAI_KEY"}
  ]
}
```

Pass it via `--baseline scripts/privacy_baseline.json` (or point the
`scripts/privacy_scan.sh` script at the file you choose).

## Running

```bash
# via python module
python -m openclaw_memory_os.privacy .

# via CLI
openclaw-memory-os privacy-scan .

# via repo script (writes JSON, exits non-zero on findings)
./scripts/privacy_scan.sh
```

Exit code is `1` if any findings, `0` otherwise. Useful in CI.

## Threat model — what it does catch, what it doesn't

* **Catches**: real-looking API keys (OpenAI, GitHub, ...), JWTs, internal
  hostnames, IPv4 addresses, the project's own private path.
* **Doesn't catch**: encrypted secrets, secrets split across multiple
  adjacent files, secrets that don't match a pattern the scanner knows,
  secrets in images or binary attachments.

For comprehensive coverage run `gitleaks` as well (see
`.github/workflows/secret-scan.yml`).
