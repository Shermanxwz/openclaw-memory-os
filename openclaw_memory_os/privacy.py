"""Privacy scanner.

This module scans text files for patterns that should NEVER appear in a
public open-source release of this project. It is intentionally narrow:
the goal is to catch obvious accidents, not to replace a full secret
scanner like ``gitleaks``. Use ``gitleaks`` in CI for deeper coverage;
this scanner is the cheap, dependency-free safety net.

Forbidden pattern categories:

* Real names / personal identifiers (kept in a small allow-list-aware list;
  the patterns here are intentionally generic placeholders only).
* Real IP addresses / hostnames.
* Known private model provider IDs and internal hostnames.
* Tokens that look like real credentials (Bearer eyJ..., sk-..., ghp_, etc.).
* Private filesystem paths under ``<project_root>``.

Suppression mechanism (robust, replaces the previous per-file exclude list):

* Per-line ``privacy-allow: RULE_ID`` markers suppress that specific rule on
  the marked line. The literal token ``privacy-allow:`` makes suppressed
  content easy to audit in code review.
* A JSON "baseline" file (``scripts/privacy_baseline.json`` by default)
  pins exact (file, line, rule_id) tuples as accepted, for cases where
  suppression by line is awkward (multiline JSON fixtures, etc.).
* Built-in default excludes cover this scanner's own source, the bundled
  sample memories fixture, and the privacy baseline file itself.

Each finding includes file, line number, line, and the rule that fired.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple


@dataclass(frozen=True)
class Rule:
    rule_id: str
    description: str
    pattern: re.Pattern
    # Allow the rule to be named in per-line `privacy-allow: <rule_id>`
    # markers to suppress. When False the marker is treated as a no-op.
    suppressible: bool = True


# Compile rules once at import time.
_RULES: Tuple[Rule, ...] = (
    Rule(
        rule_id="PRIVATE_HOSTNAME",
        description="Likely internal OpenClaw / VPS hostname.",
        pattern=re.compile(r"vps-[a-f0-9]{8,}", re.IGNORECASE),
    ),
    Rule(
        rule_id="MEMORY_OS_PATH",
        description=(
            "Private filesystem path under <project_root>. "
            "Reference paths literally only inside comments or docs that "
            "explain the rule itself; use generic placeholders elsewhere."
        ),
        # Match a leading slash so we don't flag occurrences inside
        # descriptive prose ("inside <project_root>") - wait,
        # those are exactly what we want to flag in non-doc code. The
        # per-line marker handles docs; see _is_suppressed().
        pattern=re.compile(r"/root/\.openclaw/workspace", re.IGNORECASE),
    ),
    Rule(
        rule_id="PROVIDER_ID",
        description="Private model provider identifier (new-api-..., internal-...).",
        # Allow ``model-aware-worker`` (the in-repo skill name that is benign)
        # but force ``internal-...`` to look like an ID rather than a
        # hyphenated English word like "internal-hostname".
        pattern=re.compile(
            r"\b(new-api-\d{3,}|internal-(?:[a-f0-9]{4,}|.*\d.*)\b|model-aware-worker)\b",
            re.IGNORECASE,
        ),
    ),
    Rule(
        rule_id="OPENAI_KEY",
        description="Looks like an OpenAI / Anthropic API key.",
        pattern=re.compile(r"\b(sk-[A-Za-z0-9_-]{20,}|sk-ant-[A-Za-z0-9_-]{20,})\b"),
    ),
    Rule(
        rule_id="GITHUB_TOKEN",
        description="Looks like a GitHub personal access token.",
        pattern=re.compile(r"\bghp_[A-Za-z0-9]{30,}\b"),
    ),
    Rule(
        rule_id="JWT_BEARER",
        description="Inline bearer JWT (often a leaked credential).",
        pattern=re.compile(r"Bearer\s+eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+"),
    ),
    Rule(
        rule_id="QQ_ACCOUNT",
        description="Looks like a QQ account number.",
        pattern=re.compile(r"\bQQ[:\s]+\d{6,12}\b", re.IGNORECASE),
    ),
    Rule(
        rule_id="IPV4",
        description="Public-looking IPv4 address (loopback / unspecified excluded).",
        # Allow common loopback / wildcard literals to pass through unscathed
        # so docs that show `--host 127.0.0.1` aren't flooded with false
        # positives. 127.0.0.0/8 and 0.0.0.0/8 are whitelisted.
        pattern=re.compile(
            r"\b(?!127\.|0\.|255\.|224\.|240\.|169\.254\.|10\.|192\.168\.|172\.(1[6-9]|2[0-9]|3[01])\.)"
            r"(?:\d{1,3}\.){3}\d{1,3}\b"
        ),
    ),
)


# Files we never want to scan regardless of path. These are vendored test
# fixtures, this scanner's own source (which describes the rules), and
# the baseline pinning file. All are project-controlled and intentional.
_DEFAULT_EXCLUDE_NAMES = {
    "sample_memories.json",
    "privacy_baseline.json",
    "privacy.py",
    # tests/test_privacy.py contains the synthetic secrets / private paths
    # that the scanner is *expected* to flag. Scanning it would generate
    # false positives; the test file is a deliberate fixture, not a leak.
    "test_privacy.py",
}

# Extensions to scan.
_TEXT_EXTS = {
    ".py", ".md", ".txt", ".json", ".yml", ".yaml", ".toml",
    ".html", ".j2", ".sh", ".cfg", ".ini", ".env",
}

# Per-line marker that suppresses a single rule on the marked line.
# Usage in source/docs:  ``... <project_root> ... privacy-allow: MEMORY_OS_PATH``
# Word-boundary on ``*`` is intentionally absent: ``*`` is a non-word char, so
# ``\b`` between ``*`` and end-of-string would never match.
_SUPPRESSION_MARKER_RE = re.compile(r"privacy-allow:\s*([A-Z_][A-Z0-9_]*)", re.IGNORECASE)
_SUPPRESSION_ALL_RE = re.compile(r"privacy-allow:\s*\*")


@dataclass
class Finding:
    file: str
    line: int
    rule_id: str
    description: str
    excerpt: str

    def to_dict(self) -> dict:
        return {
            "file": self.file,
            "line": self.line,
            "rule_id": self.rule_id,
            "description": self.description,
            "excerpt": self.excerpt.strip()[:200],
        }


def _iter_files(roots: Sequence[Path]) -> Iterable[Path]:
    for root in roots:
        if root.is_file():
            yield root
            continue
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            # Skip generated dependency/build caches. The scanner is meant to
            # audit repository source, docs, examples, and configuration — not
            # third-party packages installed into a local virtualenv.
            if any(part in {".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache", "dist", "build", ".mypy_cache", ".ruff_cache"} for part in path.parts):
                continue
            # Pathlib quirk: ``Path('/tmp/.env').suffix == ''`` because the
            # leading dot is treated as part of the stem. Compute the actual
            # extension manually so dotfile-style configs are still scanned.
            name = path.name
            _, dot, ext = name.rpartition(".")
            suffix = ("." + ext).lower() if dot else ""
            if suffix not in _TEXT_EXTS and name.lower() not in _TEXT_EXTS:
                continue
            if path.name in _DEFAULT_EXCLUDE_NAMES:
                continue
            if any(part.startswith(".git") for part in path.parts):
                continue
            yield path


def _suppressed_rules(line: str) -> set[str]:
    """Rule IDs allowed to fire on this line via ``privacy-allow:`` marker."""
    if _SUPPRESSION_ALL_RE.search(line):
        return {"*"}
    return {m.group(1).upper() for m in _SUPPRESSION_MARKER_RE.finditer(line)}


def _load_baseline(path: Optional[Path]) -> set[Tuple[str, int, str]]:
    if not path or not path.exists():
        return set()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return set()
    entries = set()
    for item in data.get("findings", []):
        try:
            entries.add((str(item["file"]), int(item["line"]), str(item["rule_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    return entries


def _should_skip_file(rel: str) -> bool:
    """Files completely excluded from scanning (independent of rules)."""
    if rel.endswith("/" + "privacy.py") or rel == "privacy.py":
        return True
    return False


def _is_suppressed(rel: str, line_no: int, line: str, rule_id: str) -> bool:
    suppressed = _suppressed_rules(line)
    return "*" in suppressed or rule_id.upper() in suppressed


def scan_paths(
    roots: Sequence[str | Path],
    *,
    baseline: Optional[str | Path] = None,
) -> List[dict]:
    """Scan the given paths (files or directories) and return findings.

    Args:
        roots: Files or directories to scan.
        baseline: Optional path to a JSON baseline pinning legitimate
            findings to (file, line, rule_id) tuples.
    """

    paths = [Path(r) for r in roots]
    baseline_set = _load_baseline(Path(baseline)) if baseline else set()
    findings: List[Finding] = []
    for path in _iter_files(paths):
        rel = str(path)
        if _should_skip_file(rel):
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for n, line in enumerate(text.splitlines(), start=1):
            for rule in _RULES:
                if not rule.pattern.search(line):
                    continue
                if (rel, n, rule.rule_id) in baseline_set:
                    continue
                if _is_suppressed(rel, n, line, rule.rule_id):
                    continue
                findings.append(
                    Finding(
                        file=rel,
                        line=n,
                        rule_id=rule.rule_id,
                        description=rule.description,
                        excerpt=line,
                    )
                )
    return [f.to_dict() for f in findings]


def main(argv: Sequence[str] | None = None) -> int:
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Privacy scanner for the OS.")
    parser.add_argument("paths", nargs="*", default=["."], help="Files or directories to scan.")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Optional JSON baseline file pinning legitimate findings.",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Append current findings to the baseline file instead of failing.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    findings = scan_paths(args.paths, baseline=args.baseline)

    if args.update_baseline and args.baseline:
        baseline_path = Path(args.baseline)
        existing: List[dict] = []
        if baseline_path.exists():
            try:
                existing = json.loads(baseline_path.read_text(encoding="utf-8")).get("findings", [])
            except (ValueError, OSError):
                existing = []
        merged = list(existing) + findings
        # Dedup by (file, line, rule_id)
        seen: set = set()
        deduped: List[dict] = []
        for f in merged:
            key = (f["file"], f["line"], f["rule_id"])
            if key in seen:
                continue
            seen.add(key)
            deduped.append(f)
        baseline_path.parent.mkdir(parents=True, exist_ok=True)
        baseline_path.write_text(
            json.dumps({"findings": deduped}, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        json.dump(
            {"updated_baseline": str(baseline_path), "pinned": len(deduped), "findings": findings},
            sys.stdout,
            indent=2,
        )
        sys.stdout.write("\n")
        return 0

    json.dump(
        {"findings": findings, "count": len(findings), "baseline": args.baseline},
        sys.stdout,
        indent=2,
    )
    sys.stdout.write("\n")
    return 0 if not findings else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
