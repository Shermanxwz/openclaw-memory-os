# Privacy

OpenClaw Memory OS is a personal/single-operator reference distribution.
This document records what the public release does and does not contain,
and the practical privacy posture of an operator who runs it.

## What is in the public release

The public repository contains the frozen runtime source, sanitized
documentation and metadata, and a test-harness portability correction.
It also contains:

- Generic configuration templates (`.env.example`).
- A neutral personal-taxonomy example (`config/personal_taxonomy.example.json`)
  with placeholder brand names.
- Sample data fixtures (`data/sample_memories.json`) that contain no
  real memories.
- Deployment templates (systemd units, nginx example, acme.sh hook).

## What is not in the public release

The public release does **not** contain:

- Operator-owned memory records.
- Operator-owned audit events, sessions, recall feedback, evaluation
  results, or policy-evolution state.
- Credentials (operator bearer tokens, password hashes, TOTP secrets,
  Qdrant API keys, GitHub tokens).
- Real or deployment-specific hostnames, public IP addresses, private
  domains, account names, device names, and operator-specific
  filesystem paths.
- Operator-specific taxonomy overrides (real brand names, custom
  keyword lists).
- Private Git history, internal release evidence, and production
  deployment identifiers.

Operational state lives under
`$MEMORY_OS_RECALL_STATE_DIR` (default
`~/.local/state/openclaw-memory-os/`). That directory is intentionally
outside the repository so an uninstall does not destroy operator-owned
signals.

## Documentation conventions

The documentation may use the following generic, non-sensitive
identifiers to make examples concrete:

- `127.0.0.1` and `0.0.0.0` for local-loopback examples.
- `example.com` (and similar reserved documentation placeholders) for
  domain and URL examples.
- Generic installer paths such as `/usr/local/bin` or
  `~/.local/state/openclaw-memory-os/` for deployment examples.

These are documentation placeholders only. They are not operator
configuration and they are not derived from any real host.

## Privacy scanner

The repo ships a privacy scanner at `scripts/privacy_scan.sh` and a
Python implementation at `openclaw_memory_os/privacy.py`. The scanner
is intentionally conservative:

- It flags host-shaped strings, path-shaped strings, brand-shaped
  strings, credential-shaped strings, and the operator's personal
  collection names.
- Each flagged line can be acknowledged via either an inline
  `privacy-allow:` marker (for single-line documentation examples)
  or a JSON baseline file (for multi-line fixtures).
- A failed scan is non-zero; CI runs the scanner on every push.

The scanner is a safety net for obvious accidents, not a substitute
for a real credential scanner. Operators who want deeper coverage
should run `gitleaks detect --source .` as well.

## Reporting

If you find a privacy issue in this public repository, please open an
issue or contact the maintainer privately before publishing a fix
publicly so the maintainer can rotate any potentially exposed
credentials first.
