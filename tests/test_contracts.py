"""Tests for ``openclaw_memory_os.contracts`` — hard invariants and identity model.

These tests pin the v0.3.0 invariants that must never regress:

* The candidate-key contract ``"collection:memory_id"`` is the
  canonical identity across the OS.
* ``MemoryRef`` round-trips through ``key`` / ``from_key`` and is
  immutable.
* ``normalize_string_list`` handles every legacy payload shape we
  have ever shipped (list, JSON string, comma-separated, bare
  string, None).
* The hard-contract constants are present and well-formed.
"""

from __future__ import annotations

import pytest

from openclaw_memory_os.contracts import (
    ACTIVE_FIRST,
    EMBEDDING_FAILURE_DEGRADED,
    HARD_CONTRACTS,
    MemoryRef,
    NO_PHYSICAL_DELETION,
    NO_ZERO_VECTOR_FAKE_SUCCESS,
    QWEN_NOT_IN_ONLINE_PATH,
    SUPERSEDED_BELOW_ACTIVE,
    SUPERSEDED_FALLBACK_ONLY,
    UNJUDGED_IS_NOT_NEGATIVE,
    candidate_key,
    normalize_string_list,
)


# ---------------------------------------------------------------------------
# Hard-contract constants
# ---------------------------------------------------------------------------


def test_hard_contracts_present() -> None:
    expected = {
        ACTIVE_FIRST,
        SUPERSEDED_FALLBACK_ONLY,
        SUPERSEDED_BELOW_ACTIVE,
        NO_PHYSICAL_DELETION,
        EMBEDDING_FAILURE_DEGRADED,
        NO_ZERO_VECTOR_FAKE_SUCCESS,
        UNJUDGED_IS_NOT_NEGATIVE,
        QWEN_NOT_IN_ONLINE_PATH,
    }
    assert expected.issubset(set(HARD_CONTRACTS))
    assert len(HARD_CONTRACTS) == len(expected), "duplicate hard contracts"


def test_hard_contracts_are_strings() -> None:
    for c in HARD_CONTRACTS:
        assert isinstance(c, str)
        assert c, "hard contract constant must be a non-empty string"


# ---------------------------------------------------------------------------
# MemoryRef / candidate_key
# ---------------------------------------------------------------------------


def test_memory_ref_key_format() -> None:
    ref = MemoryRef(collection="openclaw_memories", memory_id="abc-123")
    assert ref.key == "openclaw_memories:abc-123"


def test_memory_ref_from_key_roundtrip() -> None:
    ref = MemoryRef(collection="openclaw_memories", memory_id="abc-123")
    rebuilt = MemoryRef.from_key(ref.key)
    assert rebuilt == ref


def test_memory_ref_from_key_with_colon_in_memory_id() -> None:
    # partition on first ':' keeps memory_ids containing ':' intact.
    ref = MemoryRef.from_key("collection:abc:123")
    assert ref.collection == "collection"
    assert ref.memory_id == "abc:123"
    assert ref.key == "collection:abc:123"


def test_memory_ref_from_key_rejects_empty() -> None:
    with pytest.raises(ValueError):
        MemoryRef.from_key("")
    with pytest.raises(ValueError):
        MemoryRef.from_key("no-colon")
    with pytest.raises(ValueError):
        MemoryRef.from_key(":missing-collection")
    with pytest.raises(ValueError):
        MemoryRef.from_key("missing-memory:")


def test_memory_ref_is_immutable() -> None:
    ref = MemoryRef(collection="c", memory_id="m")
    with pytest.raises(Exception):
        ref.collection = "other"  # type: ignore[misc]


def test_memory_ref_rejects_unknown_fields() -> None:
    with pytest.raises(Exception):
        MemoryRef(collection="c", memory_id="m", extra="nope")  # type: ignore[call-arg]


def test_memory_ref_rejects_empty() -> None:
    with pytest.raises(Exception):
        MemoryRef(collection="", memory_id="m")  # type: ignore[call-arg]
    with pytest.raises(Exception):
        MemoryRef(collection="c", memory_id="")  # type: ignore[call-arg]


def test_candidate_key_helper() -> None:
    assert candidate_key("c", "m") == "c:m"
    # None collection falls back to "unknown" — useful for legacy log lines.
    assert candidate_key(None, "m") == "unknown:m"
    with pytest.raises(ValueError):
        candidate_key("c", None)
    with pytest.raises(ValueError):
        candidate_key("c", "")


def test_candidate_key_different_collections_dont_collide() -> None:
    # The whole point of the composite key.
    assert candidate_key("a", "1") != candidate_key("b", "1")


# ---------------------------------------------------------------------------
# normalize_string_list
# ---------------------------------------------------------------------------


def test_normalize_string_list_from_list() -> None:
    assert normalize_string_list(["a", "b", "c"]) == ["a", "b", "c"]


def test_normalize_string_list_from_list_with_whitespace() -> None:
    assert normalize_string_list(["a", "  b  ", "", None, "c"]) == ["a", "b", "c"]


def test_normalize_string_list_from_json_array() -> None:
    assert normalize_string_list('["a", "b", "c"]') == ["a", "b", "c"]


def test_normalize_string_list_from_json_string() -> None:
    assert normalize_string_list('"solo"') == ["solo"]


def test_normalize_string_list_from_comma_separated() -> None:
    assert normalize_string_list("a, b ,c") == ["a", "b", "c"]


def test_normalize_string_list_from_bare_string() -> None:
    assert normalize_string_list("hello") == ["hello"]


def test_normalize_string_list_from_none() -> None:
    assert normalize_string_list(None) == []


def test_normalize_string_list_from_empty_string() -> None:
    assert normalize_string_list("") == []
    assert normalize_string_list("   ") == []


def test_normalize_string_list_from_empty_list() -> None:
    assert normalize_string_list([]) == []


def test_normalize_string_list_from_int() -> None:
    # Unknown scalar types coerce to a single string entry.
    assert normalize_string_list(42) == ["42"]


def test_normalize_string_list_never_raises() -> None:
    # Whatever the input, the helper returns a list — never raises.
    for value in (None, "", [], {}, object(), 0, False):
        result = normalize_string_list(value)
        assert isinstance(result, list)