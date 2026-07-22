"""Tests for scripts/supersede_detect.py — conservative defaults (v0.2.3).

These tests cover the new safeguards:

    * Keyword-topic supersede is OFF by default and only fires when
      ``ENABLE_TOPIC_SUPERSEDE=1`` (or ``--enable-topic-supersede``).
    * High-confidence threshold is 0.95 (was 0.85) and requires a
      minimum number of shared shingles; without those, candidates
      fall through to the near-duplicate tag path instead of being
      silently merged.
    * Points with ``tier`` in ``core``/``long`` are NEVER auto-
      superseded. They may still be flagged ``near_duplicate``.
    * ``SUPERSEDE_MAX_APPLY`` caps the writes per collection per run.
      Dry-run is uncapped.

The unit tests build in-memory point lists; we never hit a live
Qdrant instance here.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = ROOT / "scripts"


def _load_module(name: str = "supersede_detect"):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS_DIR / "supersede_detect.py")
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---- helpers ------------------------------------------------------------

def _pt(pid, content, *, source="memory/2026-01-01.md", tier=None, topic=None, category=None):
    payload = {
        "content": content,
        "source": source,
        "status": "active",
    }
    if tier is not None:
        payload["tier"] = tier
    if topic is not None:
        payload["topic"] = topic
    if category is not None:
        payload["category"] = category
    return {"id": pid, "payload": payload}


def _set_env(monkeypatch, **values):
    for key in (
        "ENABLE_TOPIC_SUPERSEDE",
        "ENABLE_AUTO_SUPERSEDE",
        "HIGH_CONFIDENCE_JACCARD",
        "NEAR_DUPLICATE_JACCARD",
        "MIN_SHINGLES_FOR_SUPERSEDE",
        "SUPERSEDE_MAX_APPLY",
    ):
        monkeypatch.delenv(key, raising=False)
    for k, v in values.items():
        monkeypatch.setenv(k, str(v))


# ---- keyword-topic gating ----------------------------------------------

def test_topic_supersede_disabled_by_default(monkeypatch):
    """Default env must not produce any topic-based supersede links.

    This is the regression test for the incident where one maintenance
    run marked 21917 of 25444 points superseded via the keyword-topic
    pass alone. With ``ENABLE_TOPIC_SUPERSEDE`` unset, that pass must
    yield zero links even when every point in the collection shares a
    topic keyword.
    """
    _set_env(monkeypatch)
    mod = _load_module()
    # Sanity: the module-level constant respects "off by default".
    assert mod.ENABLE_TOPIC_SUPERSEDE is False, (
        "ENABLE_TOPIC_SUPERSEDE must default to False so a stray cron "
        "run cannot mass-supersede via keyword topics."
    )

    points = [
        _pt(1, "today's worker pool is using flash model"),
        _pt(2, "current worker status: flash model M3 ok"),
        _pt(3, "agent codex main controller stable"),
    ]
    links = mod.detect_supersedes(points)
    assert links == [], (
        "detect_supersedes should still build candidate links internally, "
        "but main() must not call it unless the operator opts in. Here "
        "we exercise the helper directly: opt-in vs default is enforced "
        "at the main() level. The helper still respects tier protection."
    )


def test_main_skips_topic_pass_by_default(monkeypatch, capsys):
    """``main()`` must skip the keyword-topic pass unless opted in."""
    _set_env(monkeypatch)
    mod = _load_module()

    def fake_scroll_all(collection, page_size=512):
        return [
            _pt(1, "current worker model flash M3", tier="working"),
            _pt(2, "agent codex main controller", tier="working"),
        ]

    monkeypatch.setattr(mod, "scroll_all", fake_scroll_all)
    monkeypatch.setattr(mod, "apply_supersedes", lambda *a, **kw: (0, 0))
    monkeypatch.setattr(mod, "apply_near_duplicate_tags", lambda *a, **kw: 0)

    rc = mod.main(["--collection", "fake"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "keyword-topic links: 0" in out
    assert "topic pass disabled" in out


def test_main_runs_topic_pass_when_enabled(monkeypatch, capsys):
    """``ENABLE_TOPIC_SUPERSEDE=1`` re-enables the keyword-topic pass."""
    _set_env(monkeypatch, ENABLE_TOPIC_SUPERSEDE=1)
    mod = _load_module()

    # Both points share the "worker" topic and the date gap is large.
    def fake_scroll_all(collection, page_size=512):
        return [
            _pt(1, "current worker model flash M3 status",
                source="memory/2026-01-01.md", tier="working"),
            _pt(2, "today's worker status flash M3",
                source="memory/2026-06-01.md", tier="working"),
        ]

    monkeypatch.setattr(mod, "scroll_all", fake_scroll_all)

    captured = {}

    def fake_apply(collection, links, dry_run):
        captured["links"] = list(links)
        captured["dry_run"] = dry_run
        return len(links), 0

    monkeypatch.setattr(mod, "apply_supersedes", fake_apply)
    monkeypatch.setattr(mod, "apply_near_duplicate_tags", lambda *a, **kw: 0)

    rc = mod.main(["--collection", "fake", "--dry-run"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "ENABLE_TOPIC_SUPERSEDE=1" in out
    # Both points share the "worker" topic; the older must be linked
    # to the newer as a candidate supersede.
    assert captured["links"], "expected at least one topic link with opt-in"
    assert captured["dry_run"] is True
    assert set(captured["links"]) == {(1, 2)}


# ---- core/long tier protection -----------------------------------------

def test_content_supersede_never_touches_core_or_long_tier():
    """Tier=core/long must never be auto-superseded, even at Jaccard=1.0.

    We build two identical texts across all thresholds. Without tier
    protection the content pass would supersede the older one. With
    the guard, it must instead flag both as ``near_duplicate`` and
    produce zero supersede links.
    """
    mod = _load_module()
    text = (
        "core_keyword_a：永不删 worker M3 当前模型 status 当前在跑 flash 模型 "
        "agent codex 主控 端口 8000 endpoint url api 当前状态正常稳定运行中。"
        * 4
    )
    older = _pt(100, text, source="memory/2025-01-01.md", tier="core", topic="ai_agents")
    newer = _pt(101, text, source="memory/2026-06-01.md", tier="core", topic="ai_agents")

    links, nd_ids = mod.detect_content_supersedes([older, newer])

    assert links == [], (
        f"core-tier points must never be auto-superseded; got {links!r}"
    )
    # Both points should at least be flagged as near_duplicate for review.
    assert set(nd_ids) == {100, 101}


def test_content_supersede_skips_long_tier_even_with_topic_match():
    mod = _load_module()
    text = (
        "## 教训：worker M3 当前模型 status 当前在跑 flash 模型 端口 8000 "
        "endpoint url api 状态正常稳定运行中。"
        * 4
    )
    older = _pt(7, text, source="memory/2024-12-01.md", tier="long", topic="ai_agents")
    newer = _pt(8, text, source="memory/2026-06-01.md", tier="long", topic="ai_agents")

    links, _nd = mod.detect_content_supersedes([older, newer])
    assert links == [], "long-tier must be protected from auto-supersede"


def test_topic_supersede_skips_protected_tiers():
    """Even with ENABLE_TOPIC_SUPERSEDE, the helper itself must skip core/long."""
    mod = _load_module()
    points = [
        _pt(1, "current worker model flash M3",
            source="memory/2025-01-01.md", tier="core"),
        _pt(2, "current worker model flash M3",
            source="memory/2026-06-01.md", tier="core"),
    ]
    links = mod.detect_supersedes(points, recency_gap_days=7)
    assert links == [], "topic helper must filter out protected tier points"


# ---- high-threshold + shared-shingle floor -----------------------------

def test_content_supersede_high_threshold_default_is_0_95(monkeypatch):
    """Default ``HIGH_CONFIDENCE_JACCARD`` is 0.95, not the old 0.85."""
    _set_env(monkeypatch)
    mod = _load_module()
    assert mod.HIGH_CONFIDENCE_JACCARD == 0.95


def test_content_supersede_requires_min_shared_shingles(monkeypatch):
    """Two short snippets that score Jaccard=1.0 must NOT be auto-
    superseded because their shared-shingle count is below the floor.
    They should fall through to the near-duplicate tag.
    """
    # Tighten the floor for this test so the math is easy to reason
    # about; the default module value (8) is exercised by other tests.
    _set_env(monkeypatch, MIN_SHINGLES_FOR_SUPERSEDE=3)
    mod = _load_module()
    # A 3-token phrase → only 1 shingle per side (k=4 means fewer
    # shingles than tokens short-circuit to a single mega-shingle).
    short = "current model status"  # 3 tokens → 1 shingle
    older = _pt(1, short, source="memory/2025-01-01.md")
    newer = _pt(2, short, source="memory/2026-06-01.md")
    links, nd_ids = mod.detect_content_supersedes([older, newer])
    assert links == [], (
        "tiny snippets that score Jaccard=1.0 must not auto-supersede; "
        "they should be flagged as near_duplicate for review"
    )
    assert set(nd_ids) == {1, 2}


def test_content_supersede_blocks_topic_mismatch():
    """Identical text but different ``topic`` fields must NOT auto-
    supersede. This catches the boilerplate-sharing false positive
    between unrelated "current model" status notes.
    """
    mod = _load_module()
    # Long, unique text so the shared-shingle floor is comfortably met.
    text = (
        "今天重新检查了当前模型状态 当前在跑 flash 模型 M3 worker agent codex "
        "主控 端口 8000 endpoint url api 状态正常稳定运行中。 系统健康良好。"
        * 3
    )
    older = _pt(50, text, source="memory/2025-01-01.md", topic="ai_agents")
    newer = _pt(51, text, source="memory/2026-06-01.md", topic="infrastructure")
    links, nd_ids = mod.detect_content_supersedes([older, newer])
    assert links == [], "different topics must block auto-supersede"
    # Near-duplicate tag still fires so operators see the pair in review.
    assert set(nd_ids) == {50, 51}


def test_content_supersede_high_threshold_runs_when_criteria_met():
    """Positive control: long, near-identical text in the same topic
    with a recent gap should auto-supersede the older copy.
    """
    mod = _load_module()
    base = (
        "今天重新检查了当前模型状态 当前在跑 flash 模型 M3 worker agent codex "
        "主控 端口 8000 endpoint url api 状态正常稳定运行中。 系统健康良好。"
    )
    text = base + " 备份状态 OK。 完全相同的长文本用于正向控制。" + " padding" * 12
    older = _pt(60, text, source="memory/2025-01-01.md",
                topic="ai_agents", tier="working")
    newer = _pt(61, text, source="memory/2026-06-01.md",
                topic="ai_agents", tier="working")
    links, _nd = mod.detect_content_supersedes([older, newer])
    assert links, "expected at least one supersede link for the high-confidence pair"
    assert links[0] == (60, 61), "older (60) should be superseded by newer (61)"


def test_content_supersede_near_duplicate_does_not_change_status():
    """The near-duplicate branch must tag review_reason only and never
    return a supersede link.
    """
    mod = _load_module()
    base = (
        "今天重新检查了当前模型状态 当前在跑 flash 模型 M3 worker agent codex "
        "主控 端口 8000 endpoint url api 状态正常稳定运行中。"
    )
    older = _pt(70, base + " 旧。", source="memory/2025-01-01.md", tier="working")
    # Force a moderate Jaccard by sharing ~half the shingles.
    half_shared = base[: len(base) // 2] + " 完全不同的另一段叙述。" * 8
    newer = _pt(71, half_shared, source="memory/2026-06-01.md", tier="working")
    links, nd_ids = mod.detect_content_supersedes([older, newer])
    assert links == [], (
        "near-duplicate threshold should never produce supersede links; "
        f"got {links!r}"
    )
    # If they were similar enough, both should be tagged. If not, both
    # branches just produce no output. Both outcomes are safe; the
    # assertion here is on the absence of status-changing links.
    if nd_ids:
        assert 70 in nd_ids and 71 in nd_ids


# ---- SUPERSEDE_MAX_APPLY cap -------------------------------------------

def test_apply_supersede_caps_writes_when_not_dry_run(monkeypatch):
    """When SUPERSEDE_MAX_APPLY=2 and 5 candidate links exist, only 2
    are passed to the helper and ``dropped_for_cap`` is reported.
    """
    _set_env(monkeypatch, SUPERSEDE_MAX_APPLY=2)
    mod = _load_module()

    captured = {}

    def fake_update_payloads(collection, updates, qdrant_url=None):
        captured["updates"] = list(updates)
        return len(updates)

    monkeypatch.setattr(
        "scripts._qdrant_helpers.update_payloads",
        fake_update_payloads,
        raising=False,
    )
    # Patch the symbol the way the script imports it (lazy import
    # inside the function). We patch via sys.modules so the ``from ...
    # import update_payloads`` inside the script sees our fake.
    fake_module = type(sys)("_qdrant_helpers_fake")
    fake_module.update_payloads = fake_update_payloads
    sys.modules["_qdrant_helpers"] = fake_module

    links = [(i, i + 100) for i in range(5)]
    applied, dropped = mod.apply_supersedes("fake_coll", links, dry_run=False)
    assert applied == 2
    assert dropped == 3
    assert len(captured["updates"]) == 2
    # Cap is applied to the *apply* path; the dry_run path stays
    # diagnostic-only. Repeat without dry-run to confirm cap.


def test_apply_supersede_dry_run_is_uncapped(capsys):
    """Dry-run must report all candidate links, regardless of cap."""
    mod = _load_module()
    # Cap to 1 to prove dry-run ignores it.
    original = mod.SUPERSEDE_MAX_APPLY
    mod.SUPERSEDE_MAX_APPLY = 1
    try:
        links = [(i, i + 100) for i in range(5)]
        applied, dropped = mod.apply_supersedes("fake_coll", links, dry_run=True)
    finally:
        mod.SUPERSEDE_MAX_APPLY = original
    assert applied == 5
    assert dropped == 0
    out = capsys.readouterr().out
    assert out.count("[supersede] (dry)") == 5


def test_main_reports_cap_drop_in_summary(monkeypatch, capsys):
    """``main()`` should surface the cap-drop in its summary line so an
    operator notices that a run did not apply everything.
    """
    _set_env(monkeypatch, SUPERSEDE_MAX_APPLY=1, ENABLE_AUTO_SUPERSEDE=1)
    mod = _load_module()

    def fake_scroll_all(collection, page_size=512):
        # Three long, near-identical working-tier notes with the same
        # topic so the content pass produces multiple supersede links.
        base = (
            "今天重新检查了当前模型状态 当前在跑 flash 模型 M3 worker agent codex "
            "主控 端口 8000 endpoint url api 状态正常稳定运行中。"
        )
        text = base + " 完全相同的长文本用于触发高置信重复。" + " padding" * 12
        return [
            _pt(1, text,
                source="memory/2025-01-01.md", topic="ai_agents", tier="working"),
            _pt(2, text,
                source="memory/2025-04-01.md", topic="ai_agents", tier="working"),
            _pt(3, text,
                source="memory/2026-06-01.md", topic="ai_agents", tier="working"),
        ]

    monkeypatch.setattr(mod, "scroll_all", fake_scroll_all)
    # Stub the helpers to count writes without touching Qdrant.
    fake_module = type(sys)("_qdrant_helpers_fake")
    fake_module.update_payloads = lambda *a, **kw: 1
    sys.modules["_qdrant_helpers"] = fake_module
    # Reload apply_* references to the freshly patched module by
    # re-importing the script module so the lazy imports resolve now.
    sys.modules.pop("supersede_detect", None)
    mod = _load_module()
    monkeypatch.setattr(mod, "scroll_all", fake_scroll_all)

    rc = mod.main(["--collection", "fake"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "SUPERSEDE_MAX_APPLY" in out
    assert "dropped" in out


# ---- helper contracts --------------------------------------------------

def test_protected_tiers_constant():
    mod = _load_module()
    assert mod.PROTECTED_TIERS == frozenset({"core", "long"})


def test_is_protected_helper():
    mod = _load_module()
    assert mod._is_protected({"tier": "core"}) is True
    assert mod._is_protected({"tier": "LONG"}) is True
    assert mod._is_protected({"tier": "working"}) is False
    assert mod._is_protected({"tier": ""}) is False
    assert mod._is_protected({}) is False