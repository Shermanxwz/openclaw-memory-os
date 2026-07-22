# Deletion policy

## TL;DR — read this first

| Aspect | Default behavior | Power-user escape hatch |
| --- | --- | --- |
| HTTP API | No `DELETE` route exposed | (none — never added) |
| Web UI | "Deletion" tab shows **candidates only**, `recommended_action: "review"` | (none — never added) |
| Governance runner (`autonomous_governance.sh`) | Never deletes; only audits & reports | Never deletes; escape hatch **not** wired into the timer pipeline |
| `scripts/memory_brain_consolidate.py` consolidation step | Surfaces stale candidates (count only) for human review | (none — never added) |

**Default promise (unchanged):** the OS never deletes memories on its own.

Memory Brain never deletes memories. The previous MEMORY_BRAIN_ALLOW_DELETE opt-in
has been removed; consolidation only flags stale candidates for human review.

This file is the single source of truth for the deletion contract. If
any other doc contradicts it, **this file wins**.

## How to enable (and why you almost certainly should not)

There is no longer any opt-in to enable. The previous
`MEMORY_BRAIN_ALLOW_DELETE=1` escape hatch has been removed; the
consolidation script only reports how many candidates it would have
deleted and logs them for human review. To actually remove memory
points, do it through the Qdrant admin API directly (or edit the JSON
store on disk) **after** reviewing the candidate list, with a backup
in hand. That keeps the audit trail honest and matches the rest of
the deletion-policy contract below.

## Default behavior — review-only candidate flow

The OS never deletes memories. This document explains why and how to use
the review-only candidate flow.

## Why review-only?

A memory store is more like a ledger than a cache. A record that is
"obviously expired" to one user may be a load-bearing audit entry for
another. By design the OS:

* does not include a `DELETE /api/memories/...` route,
* does not instruct the configured backend to drop records,
* does not automatically act on the candidate list it produces.

A human must always decide.

## Candidate list

`GET /api/deletion-candidates` and the **Deletion** dashboard section both
return the same list. Each entry has:

* `id` — the memory id.
* `text` — its content (truncated by the renderer).
* `tier`, `status` — for context.
* `reason` — a `;`-joined list of why-rules that fired.
* `recommended_action` — always the literal string `review`.

The reasons currently implemented:

| Trigger | Reason string |
| --- | --- |
| `status == expired` | `status=expired` |
| `status == superseded` | `status=superseded` |
| `expires_at` in the past | `expires_at in the past` |
| `tier == working` | `tier=working (session-scoped)` |
| `importance <= 0.1` | `low importance (<score>)` |

## Suggested workflow

1. Open the dashboard's **Deletion** section once a week or month.
2. Skim the candidate list.
3. For each candidate, decide outside the OS: keep, archive elsewhere, or
   let it remain. The OS does not need to know.
4. To actually remove a memory, do it in the underlying store directly
   (edit the JSON file, or call the Qdrant admin API).

## What this is NOT

It is **not** a garbage-collector. It is **not** an automatic retention
policy. It is a checklist for a human.

If you need either of those, consider building one on top of an explicit
allowlist of memory ids — not this UI.
