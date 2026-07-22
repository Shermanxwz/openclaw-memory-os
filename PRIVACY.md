# Privacy

OpenClaw Memory OS is a personal/single-operator reference distribution.
This document records what the public release does and does not contain,
and the practical privacy posture of an operator who runs it.

## What is in the public release

The public repository contains runtime source, sanitized documentation,
test fixtures, and deployment templates. It may contain:

- Generic configuration templates such as `.env.example`.
- Neutral examples such as `config/personal_taxonomy.example.json`.
- Sample data fixtures that contain no real memories.
- Systemd/nginx/acme examples and scripts with placeholder paths or domains.
- Tests that use fake tokens/keys to verify redaction behavior.

## What is not in the public release

The public release does **not** contain:

- Operator-owned memory records.
- Operator-owned audit events, sessions, recall feedback, evaluation results,
  or policy-evolution state.
- Credentials: bearer tokens, password hashes, TOTP secrets, Qdrant API keys,
  GitHub tokens, SSH credentials, or API provider keys.
- Real deployment-specific hostnames, public IP addresses, private domains,
  account names, device names, and operator-specific filesystem paths.
- Operator-specific taxonomy overrides, real brand lists, or private keyword
  lists.
- Private Git history, local release evidence, production backups, or audit
  archives.

Operational state lives outside the repository, typically under
`$MEMORY_OS_RECALL_STATE_DIR` or `$XDG_STATE_HOME/openclaw-memory-os/`.
Production deployments should keep `.env`, `.secrets/`, `.audit-*`,
`.cleanup-*`, `.venv/`, build artifacts, and Qdrant snapshots untracked.

## Runtime privacy posture

Memory OS is designed for a **single trusted operator**:

- Dashboard sessions use password + TOTP when configured.
- Browser sessions are server-side and stored as hashed tokens.
- CSRF protection is required for cookie-authenticated write endpoints.
- Query-string token auth is intentionally not accepted.
- The dashboard is review-only for deletion/governance surfaces; it does not
  execute physical memory deletion.
- Public releases should be scanned before publishing with the repository's
  privacy scanner and with `git ls-files`-scoped secret checks.

## Release checklist

Before publishing a release, verify:

1. `git ls-files --others --exclude-standard` is empty.
2. `.env`, `.secrets/`, `.audit-*`, `.cleanup-*`, and backup snapshots are
   ignored and untracked.
3. Tracked-file secret scans report only synthetic fixtures or documented
   placeholders.
4. Commit author/committer metadata does not expose private hostnames or
   operator email addresses.
5. Generated archives and release assets are built from the sanitized tree only.
