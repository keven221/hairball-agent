# Langfuse Observability Plugin

This plugin ships bundled with Hairball but is **opt-in** — it only loads when
you explicitly enable it.

## Enable

Pick one:

```bash
# Interactive: walks you through credentials + SDK install + enable
hairball tools  # → Langfuse Observability

# Manual
pip install langfuse
hairball plugins enable observability/langfuse
```

## Required credentials

Set these in `~/.hairball/.env` (or via `hairball tools`):

```bash
HAIRBALL_LANGFUSE_PUBLIC_KEY=pk-lf-...
HAIRBALL_LANGFUSE_SECRET_KEY=sk-lf-...
HAIRBALL_LANGFUSE_BASE_URL=https://cloud.langfuse.com   # or your self-hosted URL
```

Without the SDK or credentials the hooks no-op silently — the plugin fails
open.

## Verify

```bash
hairball plugins list                 # observability/langfuse should show "enabled"
hairball chat -q "hello"              # then check Langfuse for a "Hairball turn" trace
```

## Optional tuning

```bash
HAIRBALL_LANGFUSE_ENV=production       # environment tag
HAIRBALL_LANGFUSE_RELEASE=v1.0.0       # release tag
HAIRBALL_LANGFUSE_SAMPLE_RATE=0.5      # sample 50% of traces
HAIRBALL_LANGFUSE_MAX_CHARS=12000      # max chars per field (default: 12000)
HAIRBALL_LANGFUSE_DEBUG=true           # verbose plugin logging
```

## Disable

```bash
hairball plugins disable observability/langfuse
```
