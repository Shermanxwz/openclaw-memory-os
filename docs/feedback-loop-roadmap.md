# Feedback-Loop Roadmap

This document describes the feedback learning loop for OpenClaw Memory OS.

## Current State (2026-07-14) — Minimal Demo

- `record_feedback()` writes `action="feedback"` entries to the SQLite audit log.
- `scripts/replay_feedback.py` reads the audit log, aggregates useful/not-useful ratios
  over 24h / 7d / 30d windows, and writes a weight snapshot to
  `~/.local/state/openclaw-memory-os/feedback-weights.json`.
- `openclaw_memory_os/ranking.py` accepts an optional `feedback_weights` dict that
  additively scales the `importance_boost` coefficient when present.
- `/api/feedback-summary` returns the aggregated ratios and current weights.
- `scripts/autonomous_governance.sh` runs `replay_feedback.py` before the deep
  content audit so the ranking is tuned with the latest signal.

## Future Milestones

### Phase 1 — Offline Replay Validation (next)
- Take a weight snapshot, rerank a recent recall query, and compare hit
  ordering vs the unweighted version. Surface a diff statistic.

### Phase 2 — Auto-Publish / Rollback
- When a weight snapshot improves a held-out validation set, auto-publish
  it as the live ranking parameter. If precision / recall degrades,
  auto-rollback to the previous snapshot.

### Phase 3 — A/B Inference
- Run the weighted and unweighted ranking side-by-side for a fraction of
  recall-test requests, log which version's hits the user gave feedback
  on, and compare.

### Phase 4 — Main Formula Tuning
- Replace the additive `importance_boost` scaling with learned per-tag or
  per-tier coefficients sourced from the feedback aggregate.

## Non-Goals (Phase 0)
- ❌ Offline replay validation
- ❌ Auto-publish / rollback
- ❌ A/B test vs old version
- ❌ Changing the main ranking formula (only additive `importance_boost` scaling)
