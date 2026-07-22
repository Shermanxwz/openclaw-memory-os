"""Tests for G5.2 ``normalize_query`` and the new time-based
``split_cases`` in ``openclaw_memory_os.evaluation``.

The contract under test is:

1. ``normalize_query`` collapses case differences, whitespace
   differences, fullwidth / CJK-width glyphs, and zero-width /
   format characters so near-duplicate queries produce the same
   canonical form.
2. ``split_cases`` performs a 60 / 20 / 20 split ordered by
   ``created_at`` (NOT by ``query_id`` hash).
3. ``split_cases`` collapses near-duplicate queries via
   ``normalize_query`` so the same logical query in different
   forms doesn't fragment the eval set.
4. ``split_cases`` is deterministic from ``created_at`` alone; the
   legacy ``seed`` parameter is ignored.
"""

from __future__ import annotations


from openclaw_memory_os.evaluation import normalize_query, split_cases


# ---------------------------------------------------------------------------
# normalize_query tests
# ---------------------------------------------------------------------------


def test_normalize_query_collapses_case_differences():
    """Case folding: "Foo", "foo", "FOO" all collapse."""
    a = normalize_query("Foo")
    b = normalize_query("foo")
    c = normalize_query("FOO")
    assert a == b == c
    assert a == "foo"


def test_normalize_query_collapses_whitespace():
    """Internal / surrounding whitespace collapses to single spaces."""
    assert normalize_query("a  b") == normalize_query("a b")
    assert normalize_query("  hello world  ") == "hello world"
    assert normalize_query("a\tb\nc") == normalize_query("a b c")


def test_normalize_query_collapses_fullwidth():
    """Fullwidth ASCII (common in CJK input methods) collapses to ASCII."""
    # "ＡＢＣ" is fullwidth A B C.
    assert normalize_query("ＡＢＣ") == normalize_query("ABC") == "abc"


def test_normalize_query_handles_empty():
    """Empty input round-trips to empty."""
    assert normalize_query("") == ""
    # Defensive: ``None`` is treated as empty (the function is
    # called on user-provided input that may be missing).
    assert normalize_query(None) == ""  # type: ignore[arg-type]


def test_normalize_query_handles_unicode_zwj():
    """Unicode zero-width joiners / format chars don't crash.

    A ZWJ-joined family emoji like "👨\u200d👩\u200d👧" must survive
    normalisation (we only strip format chars, not the visible
    glyphs). At minimum, the function must not raise on it.
    """
    zwj = "👨\u200d👩\u200d👧"
    out = normalize_query(zwj)
    # We don't pin the exact form (depends on NFKC behaviour for
    # emoji presentation sequences); just assert non-empty and
    # crash-free.
    assert isinstance(out, str)
    assert out  # not empty


def test_normalize_query_collapses_combining_marks():
    """Combining-mark variants (e\u0301 vs é) collapse via NFKC."""
    decomposed = "e\u0301"  # 'e' + combining acute accent
    precomposed = "\u00e9"  # 'é' (single codepoint)
    assert normalize_query(decomposed) == normalize_query(precomposed)


def test_normalize_query_strips_bom_and_zero_width_space():
    """BOM and zero-width space don't poison the canonical form."""
    assert normalize_query("\ufeffhello") == "hello"
    assert normalize_query("hel\u200blo") == "hello"


# ---------------------------------------------------------------------------
# split_cases tests
# ---------------------------------------------------------------------------


def test_split_cases_is_time_based():
    """With 10 cases whose created_at is 1..10 and ratios 60/20/20,
    train contains the first 6, val the next 2, test the last 2.

    Each case has a unique normalised query so dedup is a no-op.
    """
    cases = [
        {"query_id": f"q{i}", "query_text": f"q{i}", "created_at": str(i)}
        for i in range(1, 11)
    ]
    train, val, test = split_cases(cases)
    assert [c["query_id"] for c in train] == [
        "q1", "q2", "q3", "q4", "q5", "q6"
    ]
    assert [c["query_id"] for c in val] == ["q7", "q8"]
    assert [c["query_id"] for c in test] == ["q9", "q10"]


def test_split_cases_dedups_near_duplicate_queries():
    """5 cases with the same normalised query collapse to a single
    eval-set entry (the earliest); the other four go to train.

    Layout (chronological order):

      case A: created_at=5, "Foo"
      case B: created_at=1, "foo"   <-- earliest, becomes the rep
      case C: created_at=3, " FOO "
      case D: created_at=2, "FOO"
      case E: created_at=4, "foo"

    Only case B is the representative. With 1 representative and
    60/20/20, the rep lands in train (n=1 < 3, so val_end collapses
    onto train_end). The siblings follow the rep into train.
    """
    cases = [
        {"query_id": "A", "query_text": "Foo", "created_at": "5"},
        {"query_id": "B", "query_text": "foo", "created_at": "1"},
        {"query_id": "C", "query_text": " FOO ", "created_at": "3"},
        {"query_id": "D", "query_text": "FOO", "created_at": "2"},
        {"query_id": "E", "query_text": "foo", "created_at": "4"},
    ]
    train, val, test = split_cases(cases)
    assert len(val) == 0
    assert len(test) == 0
    assert len(train) == 5
    # Every input case is present in train (none lost).
    assert {c["query_id"] for c in train} == {"A", "B", "C", "D", "E"}


def test_split_cases_seed_is_ignored_for_time_split():
    """The ``seed`` parameter is accepted for backward compatibility
    with the legacy ``evolution.split_cases(cases, seed=...)`` callers
    but the time-based split is deterministic from ``created_at``
    alone. Different seeds produce the same partition.
    """
    cases = [
        {"query_id": f"q{i}", "query_text": f"q{i}", "created_at": str(i)}
        for i in range(1, 11)
    ]
    t1, v1, s1 = split_cases(cases, seed=42)
    t2, v2, s2 = split_cases(cases, seed=99)
    t3, v3, s3 = split_cases(cases, seed=0)
    # Same train/val/test ids under every seed.
    assert [c["query_id"] for c in t1] == [c["query_id"] for c in t2]
    assert [c["query_id"] for c in t1] == [c["query_id"] for c in t3]
    assert [c["query_id"] for c in v1] == [c["query_id"] for c in v2]
    assert [c["query_id"] for c in s1] == [c["query_id"] for c in s2]


def test_split_cases_handles_iso_timestamps():
    """ISO 8601 timestamps sort correctly (real recall_runs.created_at
    shape: 'YYYY-MM-DD HH:MM:SS' produced by SQLite's datetime('now'))."""
    cases = [
        {"query_id": f"q{i}", "query_text": f"q{i}",
         "created_at": f"2026-07-{i:02d} 12:00:00"}
        for i in range(1, 11)
    ]
    train, val, test = split_cases(cases)
    assert [c["query_id"] for c in train] == [
        "q1", "q2", "q3", "q4", "q5", "q6"
    ]
    assert [c["query_id"] for c in val] == ["q7", "q8"]
    assert [c["query_id"] for c in test] == ["q9", "q10"]


def test_split_cases_empty_input():
    """Empty input -> three empty lists."""
    train, val, test = split_cases([])
    assert train == []
    assert val == []
    assert test == []


def test_split_cases_accepts_attribute_style_cases():
    """Cases can be dataclass-style objects, not just dicts."""
    from dataclasses import dataclass

    @dataclass
    class _Case:
        query_id: str
        query_text: str
        created_at: str

    cases = [
        _Case(query_id=f"q{i}", query_text=f"q{i}", created_at=str(i))
        for i in range(1, 11)
    ]
    train, val, test = split_cases(cases)
    assert [c.query_id for c in train] == [
        "q1", "q2", "q3", "q4", "q5", "q6"
    ]
    assert [c.query_id for c in val] == ["q7", "q8"]
    assert [c.query_id for c in test] == ["q9", "q10"]


def test_split_cases_fullwidth_dedup_isolates_eval_set():
    """Three different surface forms of the same logical query
    (case + fullwidth + whitespace variants) collapse to ONE entry
    in the eval set, so the offline pipeline sees a single
    representative rather than three fragments.

    Layout: 3 variants of "abc" at created_at=1/2/3, plus 8 unique
    queries at created_at=4..11. Total 11 cases. After dedup: 9
    representatives (1 abc-group + 8 unique). 60% of 9 = 5.4 -> 5
    train reps; 80% -> 7 (so val has 2); test has 2.
    """
    cases = [
        {"query_id": "abc-a", "query_text": "ABC", "created_at": "1"},
        {"query_id": "abc-b", "query_text": "ＡＢＣ", "created_at": "2"},
        {"query_id": "abc-c", "query_text": "  abc  ", "created_at": "3"},
    ]
    cases += [
        {"query_id": f"q{i}", "query_text": f"q{i}", "created_at": str(i + 3)}
        for i in range(1, 9)  # created_at 4..11
    ]
    train, val, test = split_cases(cases)
    # All 3 abc-variants share the same normalised form, so they
    # ALL go to the same bucket as their representative (the
    # earliest, created_at=1, which is "ABC"). That's train.
    train_ids = {c["query_id"] for c in train}
    val_ids = {c["query_id"] for c in val}
    test_ids = {c["query_id"] for c in test}
    assert {"abc-a", "abc-b", "abc-c"}.issubset(train_ids)
    # The 3 variants are NOT fragmented across val / test.
    assert "abc-a" not in val_ids and "abc-a" not in test_ids
    assert "abc-b" not in val_ids and "abc-b" not in test_ids
    assert "abc-c" not in val_ids and "abc-c" not in test_ids
    # No input case was dropped.
    all_ids = train_ids | val_ids | test_ids
    expected_ids = {f"q{i}" for i in range(1, 9)} | {"abc-a", "abc-b", "abc-c"}
    assert all_ids == expected_ids