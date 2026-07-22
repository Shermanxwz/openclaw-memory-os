"""Tests for the v0.3.0 ingestion validation layer (S5).

Covers: Pydantic schema enforcement, JSON extraction strategies,
length / count caps, the corrective retry, the deterministic
fallback, and the prompt_version / classification_status
metadata fields.
"""

from __future__ import annotations

import pytest

from openclaw_memory_os.ingestion_validation import (
    ALLOWED_TOPICS,
    ALLOWED_TYPES,
    CORRECTIVE_PROMPT,
    MAX_ENTITIES,
    MAX_KEYWORDS,
    MAX_SUMMARY_LEN,
    MAX_TRIGGERS,
    PROMPT_VERSION,
    _extract_first_json,
    annotate_payload,
    build_fallback_payload,
    validate_classification,
)


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------


def test_extract_first_json_from_fenced_json_block():
    text = "Here is the result:\n```json\n{\"type\": \"fact\", \"importance\": 0.7}\n```\n"
    out = _extract_first_json(text)
    assert out == {"type": "fact", "importance": 0.7}


def test_extract_first_json_from_fenced_no_language():
    text = "Result:\n```\n{\"type\": \"decision\"}\n```\n"
    out = _extract_first_json(text)
    assert out == {"type": "decision"}


def test_extract_first_json_from_bare_object():
    text = "Random prose {\"type\": \"event\", \"topic\": \"personal\"} trailing prose"
    out = _extract_first_json(text)
    assert out == {"type": "event", "topic": "personal"}


def test_extract_first_json_handles_nested_braces():
    text = "{\n  \"type\": \"fact\",\n  \"metadata\": {\"score\": 0.9}\n}"
    out = _extract_first_json(text)
    assert out == {"type": "fact", "metadata": {"score": 0.9}}


def test_extract_first_json_returns_none_for_invalid_json():
    text = "not json at all"
    out = _extract_first_json(text)
    assert out is None


def test_extract_first_json_returns_none_for_empty():
    assert _extract_first_json("") is None
    assert _extract_first_json(None) is None  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# Schema validation
# ---------------------------------------------------------------------------


def test_validate_classification_accepts_well_formed():
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "importance": 0.7,
        "confidence": 0.8,
        "summary": "x" * 50,
        "keywords": ["a", "b"],
        "entities": ["nginx"],
        "recall_triggers": ["how does nginx work"],
        "actionable": False,
    }
    parsed, err = validate_classification(payload)
    assert err is None
    assert parsed is not None
    assert parsed.type == "fact"
    assert parsed.importance == 0.7


def test_validate_classification_rejects_unknown_type():
    payload = {"type": "garbage_type", "topic": "infrastructure"}
    parsed, err = validate_classification(payload)
    assert parsed is None
    assert err == "type_invalid"


def test_validate_classification_rejects_unknown_topic():
    payload = {"type": "fact", "topic": "garbage_topic"}
    parsed, err = validate_classification(payload)
    assert parsed is None
    assert err == "topic_invalid"


def test_validate_classification_rejects_unknown_field():
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "unknown_field": "value",
    }
    parsed, err = validate_classification(payload)
    assert parsed is None
    assert err == "schema_invalid"


def test_validate_classification_rejects_importance_out_of_range():
    payload = {"type": "fact", "topic": "infrastructure", "importance": 1.5}
    parsed, err = validate_classification(payload)
    assert parsed is None
    assert err == "schema_invalid"


def test_validate_classification_rejects_empty_payload():
    parsed, err = validate_classification({})
    assert parsed is None
    assert err == "json_missing"


# ---------------------------------------------------------------------------
# Length / count caps
# ---------------------------------------------------------------------------


def test_classification_truncates_long_summary():
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "summary": "x" * (MAX_SUMMARY_LEN + 50),
    }
    parsed, _ = validate_classification(payload)
    assert len(parsed.summary) == MAX_SUMMARY_LEN


def test_classification_caps_keyword_count():
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "keywords": [f"kw{i}" for i in range(MAX_KEYWORDS + 5)],
    }
    parsed, _ = validate_classification(payload)
    assert len(parsed.keywords) == MAX_KEYWORDS


def test_classification_caps_entity_count():
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "entities": [f"ent{i}" for i in range(MAX_ENTITIES + 5)],
    }
    parsed, _ = validate_classification(payload)
    assert len(parsed.entities) == MAX_ENTITIES


def test_classification_caps_trigger_count():
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "recall_triggers": [f"t{i}" for i in range(MAX_TRIGGERS + 5)],
    }
    parsed, _ = validate_classification(payload)
    assert len(parsed.recall_triggers) == MAX_TRIGGERS


def test_classification_dedupes_case_insensitive():
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "keywords": ["Foo", "foo", "FOO", "bar"],
    }
    parsed, _ = validate_classification(payload)
    assert parsed.keywords == ["Foo", "bar"]


def test_classification_handles_string_keyword():
    payload = {"type": "fact", "topic": "infrastructure", "keywords": "single"}
    parsed, _ = validate_classification(payload)
    assert parsed.keywords == ["single"]


# ---------------------------------------------------------------------------
# Fallback
# ---------------------------------------------------------------------------


def test_build_fallback_payload_truncates_long_text():
    text = "a" * 200
    fb = build_fallback_payload(text, error_code="llm_unreachable")
    assert len(fb["summary"]) == MAX_SUMMARY_LEN
    assert fb["_classification_status"] == "fallback"
    assert fb["_classification_error_code"] == "llm_unreachable"


def test_build_fallback_payload_keeps_short_text_verbatim():
    fb = build_fallback_payload("short text", error_code="json_missing")
    assert fb["summary"] == "short text"
    assert fb["keywords"] == []


def test_fallback_importance_is_deterministic():
    fb1 = build_fallback_payload("x", error_code="llm_unreachable")
    fb2 = build_fallback_payload("y", error_code="llm_unreachable")
    assert fb1["importance"] == fb2["importance"] == 0.6


# ---------------------------------------------------------------------------
# Annotate
# ---------------------------------------------------------------------------


def test_annotate_payload_attaches_classification_metadata():
    parsed, _ = validate_classification({"type": "fact", "topic": "infrastructure"})
    out = annotate_payload(parsed, status="retry_ok")
    assert out["_classification_status"] == "retry_ok"
    assert out["prompt_version"] == PROMPT_VERSION


def test_annotate_payload_default_prompt_version():
    parsed, _ = validate_classification({"type": "fact", "topic": "infrastructure"})
    out = annotate_payload(parsed)
    assert out["prompt_version"] == PROMPT_VERSION


# ---------------------------------------------------------------------------
# Allow-list coverage
# ---------------------------------------------------------------------------


def test_allowed_types_includes_expected():
    assert {"fact", "decision", "event", "preference"} <= ALLOWED_TYPES


def test_allowed_topics_includes_expected():
    assert {
        "infrastructure",
        "business",
        "personal",
        "ai_model",
        "memory_system",
    } <= ALLOWED_TOPICS


# ---------------------------------------------------------------------------
# Corrective prompt
# ---------------------------------------------------------------------------


def test_corrective_prompt_specifies_schema():
    assert "type" in CORRECTIVE_PROMPT
    assert "topic" in CORRECTIVE_PROMPT
    assert "importance" in CORRECTIVE_PROMPT
    assert "0.0" in CORRECTIVE_PROMPT or "0" in CORRECTIVE_PROMPT


# ---------------------------------------------------------------------------
# v0.3.0 Batch 4 — Finding B4-1: the memory_brain ingest prompts ask
# the LLM for an *auxiliary* shape (sentiment, recall_triggers,
# prerequisite_memories, valid_until) that the global
# ``ClassificationSchema`` rejects with ``extra='forbid'``. The brain
# script must therefore use its own permissive ``MemoryBrainSchema``
# and fall back cleanly on validation failure.
# ---------------------------------------------------------------------------


def _load_memory_brain_ingest_module():
    """Load ``scripts/memory_brain_ingest.py`` as a stand-alone module.

    Mirrors the pattern in ``test_memory_brain_importance_normalize``
    so we can exercise the private schema without polluting
    ``sys.path`` for the rest of the test suite.
    """
    import importlib.util
    import sys
    from pathlib import Path

    repo_root = Path(__file__).resolve().parent.parent
    ingest_path = repo_root / "scripts" / "memory_brain_ingest.py"
    scripts_dir = str(ingest_path.parent)
    added = False
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
        added = True
    # Drop cached copies so we get a fresh module each test invocation.
    for mod_name in list(sys.modules):
        if mod_name in {"memory_brain_ingest"}:
            del sys.modules[mod_name]
    spec = importlib.util.spec_from_file_location(
        "memory_brain_ingest", ingest_path
    )
    assert spec is not None and spec.loader is not None, (
        f"could not build import spec for {ingest_path}"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if added:
        sys.path.pop(0)
    return module


@pytest.fixture(scope="module")
def brain_module():
    return _load_memory_brain_ingest_module()


def test_memory_brain_ingest_recall_triggers_schema_accepts_prompt_shape(brain_module):
    """A representative LLM response — including auxiliary fields
    (``sentiment``, ``recall_triggers``, ``prerequisite_memories``,
    ``valid_until``) that the global schema would reject under
    ``extra='forbid'`` — must validate against the local
    ``MemoryBrainSchema``.
    """
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "importance": 0.6,
        "summary": "smoke test memory",
        "entities": ["nginx"],
        "keywords": ["web", "proxy"],
        "sentiment": "positive",
        "actionable": False,
        "recall_triggers": ["how does nginx work", "reverse proxy"],
        "prerequisite_memories": ["knows what nginx is"],
        "valid_until": None,
    }
    parsed, error = brain_module.validate_memory_brain(payload)
    assert error is None, f"expected validation success, got error={error!r}"
    assert parsed is not None
    # The well-known auxiliary fields must survive intact so the
    # Qdrant payload can carry them.
    assert parsed["sentiment"] == "positive"
    assert parsed["recall_triggers"] == ["how does nginx work", "reverse proxy"]
    assert parsed["prerequisite_memories"] == ["knows what nginx is"]
    assert parsed["valid_until"] is None
    assert parsed["type"] == "fact"
    assert parsed["topic"] == "infrastructure"


def test_memory_brain_ingest_recall_triggers_falls_back_cleanly(brain_module):
    """An invalid LLM response must produce a deterministic fallback
    with ``recall_triggers=[]`` and ``prerequisite_memories=[]`` so
    the ingest pipeline never raises on a malformed model output.

    The local ``MemoryBrainSchema`` accepts the auxiliary fields
    the global ``ClassificationSchema`` would reject (``sentiment``,
    ``prerequisite_memories``), but it still enforces the
    ``ALLOWED_TYPES`` allow-list. A bogus ``type`` therefore routes
    through the fallback path the same way a missing ``sentiment``
    or malformed JSON would.
    """
    invalid_payload = {
        "type": "this_type_is_not_in_the_allowlist",
        "topic": "infrastructure",
        "importance": 0.6,
        "sentiment": "neutral",
        "recall_triggers": ["trigger"],
        "prerequisite_memories": ["prereq"],
    }
    parsed, error = brain_module.validate_memory_brain(invalid_payload)
    assert parsed is None
    assert error in {"schema_invalid", "json_missing"}

    fb = brain_module.build_memory_brain_fallback(
        "test fallback text", error_code=error or "schema_invalid"
    )
    assert fb["recall_triggers"] == []
    assert fb["prerequisite_memories"] == []
    assert fb["sentiment"] == "neutral"
    assert fb["_classification_status"] == "fallback"
    assert fb["_classification_error_code"] in {"schema_invalid", "json_missing"}


def test_memory_brain_ingest_schema_accepts_missing_sentiment(brain_module):
    """``sentiment`` has a default (``"neutral"``) on the local
    schema, so an LLM that omits it must still validate. This
    pins the relaxed semantics the brain script relies on.
    """
    payload = {
        "type": "fact",
        "topic": "infrastructure",
        "importance": 0.6,
        "recall_triggers": ["trigger"],
        "prerequisite_memories": ["prereq"],
    }
    parsed, error = brain_module.validate_memory_brain(payload)
    assert error is None, f"expected validation success, got error={error!r}"
    assert parsed is not None
    assert parsed["sentiment"] == "neutral"
    assert parsed["recall_triggers"] == ["trigger"]
    assert parsed["prerequisite_memories"] == ["prereq"]
