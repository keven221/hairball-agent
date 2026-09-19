---
name: dispatching-parallel-coding
description: "Dispatch parallel coding agents for independent domains."
version: 1.0.0
author: Hairball Agent (pattern adapted from reference/superpowers dispatching-parallel-agents)
license: MIT
platforms: [linux, macos, windows]
metadata:
  hairball:
    tags: [delegation, parallel, coding, explore, subagent, dispatch]
    related_skills: [simplify-code, systematic-debugging, test-driven-development, plan]
---

# Dispatching Parallel Coding Agents

Fan out independent coding investigations or fixes with `delegate_task`
batch mode. Each child gets an isolated context — you construct exactly
what they need; they never inherit your conversation history.

**Core principle:** One agent per independent problem domain. Never
parallelize writers that share files or mutable state.

## When to Use

Use when:
- 2+ test files / subsystems fail with different root causes
- Broad codebase search needs fan-out (`profile="explore"`)
- Independent features can be researched or patched without overlap

Do NOT use when:
- Fixes share the same module or would race on the same files
- Understanding one failure requires the other (sequential)
- A single tight investigation is enough

## Profiles

Pass `profile` on `delegate_task` (top-level or per-task):

| Profile | Use for |
|---------|---------|
| `explore` | Read-only search / locate code. Strips `write_file`/`patch`. |
| `default` / `worker` / omit | Inherit parent toolsets (normal coding leaf). |

Prefer `explore` for reconnaissance, then a separate `worker` (or omit)
pass if implementation is needed. Do not ask `explore` children to edit.

## Pattern

### 1. Identify independent domains

Group by what is broken or what must change. Domains that touch the same
files, shared config, or a single sequential dependency stay sequential.

### 2. Craft self-contained tasks

Each task needs:
- **Specific scope** — one file / subsystem
- **Clear goal** — what done looks like
- **Context** — absolute paths, error text, constraints, language/tone
- **Expected output** — summary shape (findings, files changed, verification)

### 3. Dispatch in one batch

```text
delegate_task(
  tasks=[
    {
      "goal": "Locate every caller of loadCatalog in hairball-ui frontend",
      "context": "Repo root: /path/to/repo. Focus hairball-ui/frontend/.",
      "profile": "explore",
    },
    {
      "goal": "Locate PATCH handlers for hairball-config in hairball-ui server",
      "context": "Repo root: /path/to/repo. Focus hairball-ui/server/routes/.",
      "profile": "explore",
    },
  ]
)
```

Stay within `delegation.max_concurrent_children` (default 3). Split into
multiple `delegate_task` calls if you need more.

### 4. Review and integrate

After `collab_wait` → `collab_collect` (or equivalent mailbox drain):
- Verify claims (paths exist, tests actually pass)
- Check for conflicting edits across workers
- Run the relevant test suite yourself before declaring done

## Hard rules

- Subagents have **no memory** of your chat — put paths and errors in `context`.
- Copy user-named paths/IDs into each task's `goal`/`context` **verbatim**; do not swap in a "similar" file from another app surface.
- Parallel **writes** only when file sets are disjoint. If unsure, serialize.
- Treat child summaries as self-reports; verify side effects.
- Prefer `explore` for search fan-out; do not grant write work to explore.
- Background delegations are not durable across `/new` or process exit.
- Do not treat a missing inject turn as failure — collect via wait→collect before synthesizing or parent-side fallback.

## Common mistakes

- Too broad: "fix all the tests" → one agent per failing domain
- No context: "fix the race" without file paths / logs
- Shared-state parallel: two agents editing the same module
- Using `explore` then expecting patches back
