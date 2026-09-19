---
name: hairball-collab-v2
description: Mailbox single-rail parallel subagents via collab_*.
version: 0.5.0
author: Hairball Agent
license: MIT
platforms: [linux, macos, windows]
metadata:
  hairball:
    tags: [collab, multi-agent, parallel, mailbox, v2]
---

# Hairball Collab V2 (Single Rail)

Primary multi-agent orchestration. Children run in the background; results
surface via mailbox autosurface (wait wake / tool-tail reminder / cross-turn)
or optional `collab_collect` — **never** via async inject turns.

Default: `delegation.collab_v2.enabled: true` and
`delegation.collab_v2.autosurface_enabled: true`.

## When to Use

Use `collab_*` for any parallel / multi-agent work. `delegate_task` background
shares the same mailbox gate but prefer `collab_spawn` for new work.

## Workflow

1. `collab_spawn(goal, task_name?, …)` ×N — fire-and-forget
2. Continue local work (do **not** busy-poll)
3. `collab_wait(timeout_ms)` — wakes on `agent_completed` (or parent steer),
   **not** on spawn. On wake, `results[]` usually already includes summaries.
   If `still_running>0`, wait again before final synthesis
4. Synthesize once. `collab_collect` is optional/idempotent (returns `[]` if
   already autosurfaced)
5. Optional: `collab_send` (QueueOnly) then `collab_followup` (TriggerTurn)

## Tools

- `collab_spawn` / `collab_wait` / `collab_collect`
- `collab_send` / `collab_followup` / `collab_interrupt` / `collab_list`

## Hard rules

- Do **not** wait for an inject / `[ASYNC DELEGATION COMPLETE]` follow-up turn
- Do **not** bypass with `terminal(..., background=true)` when collab tools exist
- Prefer disjoint file writes across children
- Interrupt / timeout waits do **not** drain the mailbox

## Verification

```bash
scripts/run_tests.sh tests/tools/test_collab_v2.py -q
```
