"""Tests for the personal taxonomy loader and de-brand contract.

These tests cover:

    1. Loader behavior on happy path (JSON object -> dict[str, list[str]]).
    2. Loader behavior on list-of-dicts shape (keyword field normalization).
    3. Missing-file fall-through (returns empty dict, INFO log emitted).
    4. Malformed-JSON warning fall-through (returns empty dict, WARNING).
    5. Env var override (MEMORY_OS_TAXONOMY_PATH changes the path).
    6. Tier classifier still classifies correctly when the taxonomy is
       loaded from the operator's gitignored file.
    7. **No-real-brand-leak guard**: ``config/personal_taxonomy.example.json``
       (the file checked into the public repo) must NOT contain
       generic brand placeholders (case-insensitive),
       and no ``.py`` / public config / docs file may embed those strings.
    8. ``expand_with_personal`` dedups base + taxonomy entries and
       preserves order.

These tests intentionally run without Qdrant / Ollama. They import
:mod:`openclaw_memory_os.personal_taxonomy` directly and avoid the
package-level conftest env scrub for ``MEMORY_OS_TAXONOMY_PATH`` by
clearing the env var themselves before each test that exercises
fall-through semantics.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

# Make sure the project root is importable when pytest runs from the repo.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from openclaw_memory_os.personal_taxonomy import (  # noqa: E402
    expand_with_personal,
    get_tier_keywords,
    load_personal_taxonomy,
)


# Generic brand placeholders that must never leak into the public repo.
# Tests use these tokens as the baseline so the assertion self-documents
# what operators should keep out of tracked files. Operators replace the
# keys in ``config/personal_taxonomy.json`` with their own keywords.
PUBLIC_FORBIDDEN_BRANDS = ("brand_alpha", "brand_beta", "brand_gamma")


@pytest.fixture(autouse=True)
def _scrub_taxonomy_env(monkeypatch):
    """Force the loader through its default-path / missing-file branches."""
    monkeypatch.delenv("MEMORY_OS_TAXONOMY_PATH", raising=False)


# --- Loader behavior -----------------------------------------------------


def test_load_from_existing_default_file_loads_amazon_keywords():
    """The local gitignored config (when present) provides amazon
    keywords. We don't depend on the real contents, only that the
    loader doesn't crash and returns a dict with the right shape.
    """
    taxonomy = load_personal_taxonomy()
    assert isinstance(taxonomy, dict)
    # No crash, no exception. Empty dict is fine — missing file path.
    for tier, kws in taxonomy.items():
        assert isinstance(tier, str)
        assert isinstance(kws, list)
        for k in kws:
            assert isinstance(k, str)
            assert k  # no empty strings


def test_missing_file_returns_empty_dict(tmp_path, caplog):
    """When the config file does not exist, the loader returns {} and
    logs a single info-level line — never raises.
    """
    missing = tmp_path / "does-not-exist.json"
    with caplog.at_level(logging.INFO, logger="openclaw_memory_os.personal_taxonomy"):
        result = load_personal_taxonomy(path=str(missing))
    assert result == {}
    info_records = [r for r in caplog.records if r.levelno == logging.INFO]
    assert any("no file at" in r.getMessage() for r in info_records), (
        f"expected an info-level 'no file at' record, got: {[r.getMessage() for r in info_records]}"
    )


def test_malformed_json_returns_empty_dict_warns(tmp_path, caplog):
    """Malformed JSON degrades to an empty dict with a single WARNING."""
    bad = tmp_path / "broken.json"
    bad.write_text("{not-valid-json", encoding="utf-8")
    with caplog.at_level(logging.WARNING, logger="openclaw_memory_os.personal_taxonomy"):
        result = load_personal_taxonomy(path=str(bad))
    assert result == {}
    warn_records = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("malformed JSON" in r.getMessage() for r in warn_records), (
        f"expected a 'malformed JSON' WARNING, got: {[r.getMessage() for r in warn_records]}"
    )


def test_env_var_overrides_default_path(tmp_path, monkeypatch):
    """MEMORY_OS_TAXONOMY_PATH points the loader at the operator file."""
    override = tmp_path / "personal.json"
    override.write_text(
        json.dumps({"amazon": ["mytestbrand"]}, ensure_ascii=False),
        encoding="utf-8",
    )
    monkeypatch.setenv("MEMORY_OS_TAXONOMY_PATH", str(override))
    taxonomy = load_personal_taxonomy()
    assert taxonomy.get("amazon") == ["mytestbrand"]


def test_env_var_missing_file_still_returns_empty(tmp_path, monkeypatch, caplog):
    """When the env var points at a non-existent file, still empty + info."""
    monkeypatch.setenv("MEMORY_OS_TAXONOMY_PATH", str(tmp_path / "absent.json"))
    with caplog.at_level(logging.INFO, logger="openclaw_memory_os.personal_taxonomy"):
        result = load_personal_taxonomy()
    assert result == {}


def test_dict_shape_normalized_to_strings(tmp_path):
    """List-of-dicts values are normalized via the .keyword field."""
    p = tmp_path / "t.json"
    p.write_text(
        json.dumps(
            {
                "core": [
                    {"keyword": "core keyword 1"},
                    {"keyword": "core keyword 2"},
                    "raw string",
                    {"keyword": ""},  # empty -> dropped
                    {"keyword": "  "},  # whitespace -> dropped
                    "duplicate",
                ],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    taxonomy = load_personal_taxonomy(path=str(p))
    assert taxonomy["core"] == [
        "core keyword 1",
        "core keyword 2",
        "raw string",
        "duplicate",
    ]


def test_underscore_keys_ignored(tmp_path):
    """Leading-underscore keys are treated as comments / metadata."""
    p = tmp_path / "t.json"
    p.write_text(
        json.dumps(
            {
                "_comment": "internal only",
                "_meta": {"version": 99},
                "core": ["k1"],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    taxonomy = load_personal_taxonomy(path=str(p))
    assert "_comment" not in taxonomy
    assert "_meta" not in taxonomy
    assert taxonomy == {"core": ["k1"]}


def test_non_dict_root_falls_back_to_empty(tmp_path):
    """JSON root that is not an object (e.g. a list) yields empty dict."""
    p = tmp_path / "t.json"
    p.write_text(json.dumps(["not", "a", "dict"]), encoding="utf-8")
    taxonomy = load_personal_taxonomy(path=str(p))
    assert taxonomy == {}


def test_tier_values_drop_non_lists(tmp_path):
    """Tier values that are not lists / strings are silently skipped."""
    p = tmp_path / "t.json"
    p.write_text(
        json.dumps(
            {
                "core": ["k1"],
                "long": 42,  # unsupported -> dropped
                "amazon": {"raw": "dict"},  # unsupported -> dropped
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    taxonomy = load_personal_taxonomy(path=str(p))
    assert taxonomy == {"core": ["k1"]}


# --- helpers -------------------------------------------------------------


def test_get_tier_keywords_safe():
    """get_tier_keywords never raises; returns empty list on miss."""
    assert get_tier_keywords({}, "missing") == []
    assert get_tier_keywords({"core": ["a", "b"]}, "core") == ["a", "b"]
    # Empty tier name -> empty list (no crash)
    assert get_tier_keywords({"core": ["a"]}, "") == []


def test_expand_with_personal_dedups_and_preserves_order():
    base = ["a", "b"]
    taxonomy = {"core": ["b", "c", "A"]}  # 'b' duplicates, 'A' is new
    merged = expand_with_personal(base, taxonomy, "core")
    # base comes first in order, then taxonomy entries not already seen
    assert merged[:2] == ["a", "b"]
    assert "c" in merged
    assert "A" in merged
    # duplicates collapsed
    assert merged.count("b") == 1


def test_expand_with_personal_handles_missing_tier():
    base = ["x"]
    assert expand_with_personal(base, {}, "nope", "alsonope") == ["x"]


def test_expand_with_personal_empty_everything():
    assert expand_with_personal([], {}, "core") == []
    assert expand_with_personal(["x", "y"], {}) == ["x", "y"]


# --- Public-file no-leak guard ------------------------------------------


def _public_repo_root() -> Path:
    """Repo root used by tests. This file lives at
    ``<repo>/tests/test_personal_taxonomy.py``.
    """
    return Path(__file__).resolve().parent.parent


def _gather_public_files() -> list[Path]:
    """Return every public source / config / docs path that is checked
    into the repo. Excludes the gitignored real taxonomy and the
    ``.venv`` / ``.git`` / build artefacts.
    """
    root = _public_repo_root()
    skip_dirs = {
        ".git", ".venv", "venv", "env",
        "build", "dist", "__pycache__",
        ".pytest_cache", ".mypy_cache", ".ruff_cache",
        "node_modules",
        "openclaw_memory_os.egg-info",
        ".egg-info",
    }
    # Files / dirs we explicitly don't scan.
    skip_paths = {
        root / "config" / "personal_taxonomy.json",  # local, gitignored
        root / ".env",                                # local secrets
        root / ".env.local",
    }
    candidates: list[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        rel = path.relative_to(root)
        if any(part in skip_dirs for part in rel.parts):
            continue
        if path in skip_paths:
            continue
        # Limit the file types we care about for the no-leak guard.
        if path.suffix.lower() not in {
            ".py", ".md", ".json", ".yml", ".yaml",
            ".toml", ".cfg", ".ini", ".txt",
            ".sh", ".example", ".env.example",
        } and path.name not in {".gitignore", ".env.example"}:
            continue
        candidates.append(path)
    return candidates


@pytest.mark.parametrize("brand", PUBLIC_FORBIDDEN_BRANDS)
def test_public_files_have_no_brand_leak(brand):
    """Every public **production** source / config / docs file checked
    into the repo must NOT contain the redacted brand names,
    case-insensitive. This is the de-brand guard.

    This test file is intentionally allowed to mention the brand names
    so the test can assert them; we exclude it from the scan, plus any
    other file in ``tests/`` (the brand strings appear only in the
    test's input data, never in production code).
    """
    needle = brand.lower()
    offenders: list[str] = []
    tests_root = Path(__file__).resolve().parent
    for path in _gather_public_files():
        # Allow the test files themselves; they contain the brand names
        # only as test inputs / constant definitions.
        try:
            path.relative_to(tests_root)
            continue
        except ValueError:
            pass
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if needle in text.lower():
            offenders.append(str(path))
    assert not offenders, (
        f"Brand literal '{brand}' leaked into: {offenders}"
    )


def test_example_taxonomy_has_only_placeholder_brands():
    """The committed example file uses brand_a / brand_b / brand_c. It
    must not contain any of the real brand names.
    """
    example = _public_repo_root() / "config" / "personal_taxonomy.example.json"
    assert example.is_file(), f"missing example file: {example}"
    raw = example.read_text(encoding="utf-8")
    for brand in PUBLIC_FORBIDDEN_BRANDS:
        assert brand.lower() not in raw.lower(), (
            f"example file leaked brand '{brand}'"
        )
    # And it should actually contain the placeholder markers so a fork
    # operator immediately sees they must edit it.
    for placeholder in ("brand_a", "brand_b", "brand_c"):
        assert placeholder in raw, (
            f"example file missing placeholder '{placeholder}'"
        )


# --- Classifier integration smoke test -----------------------------------


def test_tier_classifier_does_not_crash_without_taxonomy(tmp_path, monkeypatch):
    """End-to-end smoke test: scripts/tier_classifier.classify_tier
    works with no taxonomy file present.
    """
    # Point every code path that loads the taxonomy at a non-existent file.
    monkeypatch.setenv("MEMORY_OS_TAXONOMY_PATH", str(tmp_path / "missing.json"))
    # Drop the cached module if it was imported elsewhere, so the env var
    # reset takes effect.
    for mod_name in list(sys.modules):
        if mod_name.startswith("scripts.tier_classifier") or mod_name == "scripts.tier_classifier":
            del sys.modules[mod_name]

    sys.path.insert(0, str(ROOT / "scripts"))
    try:
        from tier_classifier import classify_tier, classify_topic  # type: ignore
    finally:
        sys.path.pop(0)

    # core / long / amazon / fallback must still work with no operator config.
    assert classify_tier("core_keyword_a today", "MEMORY.md") == "core"
    assert classify_tier("## 教训 - the lesson is here", "memory/x.md") == "long"
    assert classify_tier("random text", "memory/short.md") in {"short", "medium"}

    topic = classify_topic("纯文本片段 with amazon_topic_keyword mentioned", "memory/x.md")
    assert topic in {"amazon", "personal", "memory_system", None, "infrastructure", "ai_agents", "finance"}
