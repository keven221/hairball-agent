<p align="center">
  <img src="assets/banner.png" alt="Hairball Agent" width="100%">
</p>

# Hairball Agent ☤
<p align="center">
  <a href="./">Hairball Agent</a> | <a href="./">Hairball Desktop</a>
</p>
<p align="center">
  <a href="./website/docs/"><img src="https://img.shields.io/badge/Docs-local-FFD700?style=for-the-badge" alt="Documentation"></a>
<a href="./LICENSE"><img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License: MIT"></a>
<a href="README.zh-CN.md"><img src="https://img.shields.io/badge/Lang-中文-red?style=for-the-badge" alt="中文"></a>
  <a href="README.ur-pk.md"><img src="https://img.shields.io/badge/Lang-اردو-green?style=for-the-badge" alt="اردو"></a>
  <a href="README.es.md"><img src="https://img.shields.io/badge/Lang-Español-orange?style=for-the-badge" alt="Español"></a>
</p>

**Hairball Agent — a self-improving personal AI agent.** It's the only agent with a built-in learning loop — it creates skills from experience, improves them during use, nudges itself to persist knowledge, searches its own past conversations, and builds a deepening model of who you are across sessions. Run it on a $5 VPS, a GPU cluster, or serverless infrastructure that costs nearly nothing when idle. It's not tied to your laptop — talk to it from Telegram while it works on a cloud VM.

Use any model you want — OpenRouter, OpenAI, Anthropic, your own endpoint, and [many others](./website/docs/integrations/providers). Configure with `hairball setup` / `hairball model` — bring your own API keys, no lock-in.

<table>
<tr><td><b>A real terminal interface</b></td><td>Full TUI with multiline editing, slash-command autocomplete, conversation history, interrupt-and-redirect, and streaming tool output.</td></tr>
<tr><td><b>Lives where you do</b></td><td>Telegram, Discord, Slack, WhatsApp, Signal, and CLI — all from a single gateway process. Voice memo transcription, cross-platform conversation continuity.</td></tr>
<tr><td><b>A closed learning loop</b></td><td>Agent-curated memory with periodic nudges. Autonomous skill creation after complex tasks. Skills self-improve during use. FTS5 session search with LLM summarization for cross-session recall. <a href="https://github.com/plastic-labs/honcho">Honcho</a> dialectic user modeling. Compatible with the <a href="https://agentskills.io">agentskills.io</a> open standard.</td></tr>
<tr><td><b>Scheduled automations</b></td><td>Built-in cron scheduler with delivery to any platform. Daily reports, nightly backups, weekly audits — all in natural language, running unattended.</td></tr>
<tr><td><b>Delegates and parallelizes</b></td><td>Spawn isolated subagents for parallel workstreams. Write Python scripts that call tools via RPC, collapsing multi-step pipelines into zero-context-cost turns.</td></tr>
<tr><td><b>Runs anywhere, not just your laptop</b></td><td>Six terminal backends — local, Docker, SSH, Singularity, Modal, and Daytona. Daytona and Modal offer serverless persistence — your agent's environment hibernates when idle and wakes on demand, costing nearly nothing between sessions. Run it on a $5 VPS or a GPU cluster.</td></tr>
<tr><td><b>Research-ready</b></td><td>Batch trajectory generation, trajectory compression for training the next generation of tool-calling models.</td></tr>
</table>

---

## Quick Install

### Linux, macOS, WSL2, Termux

```bash
curl -fsSL ./scripts/install.sh | bash
```

### Windows (native, PowerShell)

> **Heads up:** Native Windows runs Hairball without WSL — CLI, gateway, TUI, and tools all work natively. If you'd rather use WSL2, the Linux/macOS one-liner above works there too. Found a bug? Please [file issues](hairball-agent).

Run this in PowerShell:

```powershell
iex (irm ./scripts/install.ps1)
```

The installer handles everything: uv, Python 3.11, Node.js, ripgrep, ffmpeg, **and a portable Git Bash** (MinGit, unpacked to `%LOCALAPPDATA%\hairball\git` — no admin required, completely isolated from any system Git install). Hairball uses this bundled Git Bash to run shell commands.

If you already have Git installed, the installer detects it and uses that instead. Otherwise a ~45MB MinGit download is all you need — it won't touch or interfere with any system Git.

> **Android / Termux:** The tested manual path is documented in the [Termux guide](./website/docs/getting-started/termux). On Termux, Hairball installs a curated `.[termux]` extra because the full `.[all]` extra currently pulls Android-incompatible voice dependencies.
>
> **Windows:** Native Windows is fully supported — the PowerShell one-liner above installs everything. If you'd rather use WSL2, the Linux command works there too. Native Windows install lives under `%LOCALAPPDATA%\hairball`; WSL2 installs under `~/.hairball` as on Linux.

After installation:

```bash
source ~/.bashrc    # reload shell (or: source ~/.zshrc)
hairball              # start chatting!
```

### Troubleshooting

#### Windows Defender or antivirus flags `uv.exe` as malware

If your antivirus (Bitdefender, Windows Defender, etc.) quarantines `uv.exe` from the Hairball `bin` folder (`%LOCALAPPDATA%\hairball\bin\uv.exe`), this is a **false positive**. The file is Astral's `uv` — the Rust Python package manager Hairball bundles to manage its Python environment. ML-based antivirus engines commonly flag unsigned Rust binaries that download and install packages.

**To verify your copy is authentic:**

```powershell
# Install GitHub CLI if needed
winget install --id GitHub.cli

# Login to GitHub
gh auth login

# Run verification
$uv = "$env:LOCALAPPDATA\hairball\bin\uv.exe"
$ver = (& $uv --version).Split(' ')[1]
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
$zip = "$env:TEMP\uv.zip"
Invoke-WebRequest "https://github.com/astral-sh/uv/releases/download/$ver/uv-x86_64-pc-windows-msvc.zip" -OutFile $zip -UseBasicParsing
gh attestation verify $zip --repo astral-sh/uv
Expand-Archive $zip "$env:TEMP\uv_x" -Force
(Get-FileHash "$env:TEMP\uv_x\uv.exe").Hash -eq (Get-FileHash $uv).Hash
```

If attestation says "Verification succeeded" and the last line prints `True`, you're good.

**To whitelist Hairball:**
- **Windows Defender:** Run PowerShell as Admin → `Add-MpPreference -ExclusionPath "$env:LOCALAPPDATA\hairball\bin"`
- **Bitdefender:** Add an exception in the Bitdefender console (Protection > Antivirus > Settings > Manage Exceptions)
- Whitelist the **folder**, not the file hash — Hairball updates `uv` and the hash changes every version

For more context, see the upstream Astral reports: [astral-sh/uv#13553](https://github.com/astral-sh/uv/issues/13553), [astral-sh/uv#15011](https://github.com/astral-sh/uv/issues/15011), [astral-sh/uv#10079](https://github.com/astral-sh/uv/issues/10079).

---

## Getting Started

```bash
hairball              # Interactive CLI — start a conversation
hairball model        # Choose your LLM provider and model
hairball tools        # Configure which tools are enabled
hairball config set   # Set individual config values
hairball gateway      # Start the messaging gateway (Telegram, Discord, etc.)
hairball setup        # Run the full setup wizard (configures everything at once)
hairball claw migrate # Migrate from legacy (if coming from legacy)
hairball update       # Update to the latest version
hairball doctor       # Diagnose any issues
```

📖 **[Full documentation →](./website/docs/)**

---

## Bring your own API keys

Hairball works with whatever provider you want. Configure models with `hairball setup` / `hairball model`, and tool backends (web search, image generation, TTS, browser) with `hairball tools` or `.env` keys.

Managed Portal / Tool Gateway onboarding is disabled by default (no third-party portal dependency). If you previously used that path, switch to direct provider and tool keys.

```bash
hairball setup
hairball tools
```

Full provider docs: [providers](./website/docs/integrations/providers).

---

## CLI vs Messaging Quick Reference

Hairball has two entry points: start the terminal UI with `hairball`, or run the gateway and talk to it from Telegram, Discord, Slack, WhatsApp, Signal, or Email. Once you're in a conversation, many slash commands are shared across both interfaces.

| Action                         | CLI                                           | Messaging platforms                                                              |
| ------------------------------ | --------------------------------------------- | -------------------------------------------------------------------------------- |
| Start chatting                 | `hairball`                                      | Run `hairball gateway setup` + `hairball gateway start`, then send the bot a message |
| Start fresh conversation       | `/new` or `/reset`                            | `/new` or `/reset`                                                               |
| Change model                   | `/model [provider:model]`                     | `/model [provider:model]`                                                        |
| Set a personality              | `/personality [name]`                         | `/personality [name]`                                                            |
| Retry or undo the last turn    | `/retry`, `/undo`                             | `/retry`, `/undo`                                                                |
| Compress context / check usage | `/compress`, `/usage`, `/insights [--days N]` | `/compress`, `/usage`, `/insights [days]`                                        |
| Browse skills                  | `/skills` or `/<skill-name>`                  | `/<skill-name>`                                                                  |
| Interrupt current work         | `Ctrl+C` or send a new message                | `/stop` or send a new message                                                    |
| Platform-specific status       | `/platforms`                                  | `/status`, `/sethome`                                                            |

For the full command lists, see the [CLI guide](./website/docs/user-guide/cli) and the [Messaging Gateway guide](./website/docs/user-guide/messaging).

---

## Documentation

All documentation lives at **[local Hairball docs/docs](./website/docs/)**:

| Section                                                                                             | What's Covered                                             |
| --------------------------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| [Quickstart](./website/docs/getting-started/quickstart)                 | Install → setup → first conversation in 2 minutes          |
| [CLI Usage](./website/docs/user-guide/cli)                              | Commands, keybindings, personalities, sessions             |
| [Configuration](./website/docs/user-guide/configuration)                | Config file, providers, models, all options                |
| [Messaging Gateway](./website/docs/user-guide/messaging)                | Telegram, Discord, Slack, WhatsApp, Signal, Home Assistant |
| [Security](./website/docs/user-guide/security)                          | Command approval, DM pairing, container isolation          |
| [Tools & Toolsets](./website/docs/user-guide/features/tools)            | 40+ tools, toolset system, terminal backends               |
| [Skills System](./website/docs/user-guide/features/skills)              | Procedural memory, Skills Hub, creating skills             |
| [Memory](./website/docs/user-guide/features/memory)                     | Persistent memory, user profiles, best practices           |
| [MCP Integration](./website/docs/user-guide/features/mcp)               | Connect any MCP server for extended capabilities           |
| [Cron Scheduling](./website/docs/user-guide/features/cron)              | Scheduled tasks with platform delivery                     |
| [Context Files](./website/docs/user-guide/features/context-files)       | Project context that shapes every conversation             |
| [Architecture](./website/docs/developer-guide/architecture)             | Project structure, agent loop, key classes                 |
| [Contributing](./website/docs/developer-guide/contributing)             | Development setup, PR process, code style                  |
| [CLI Reference](./website/docs/reference/cli-commands)                  | All commands and flags                                     |
| [Environment Variables](./website/docs/reference/environment-variables) | Complete env var reference                                 |

---

## Migrating from legacy

If you're coming from legacy, Hairball can automatically import your settings, memories, skills, and API keys.

**During first-time setup:** The setup wizard (`hairball setup`) automatically detects `~/.openclaw` and offers to migrate before configuration begins.

**Anytime after install:**

```bash
hairball claw migrate              # Interactive migration (full preset)
hairball claw migrate --dry-run    # Preview what would be migrated
hairball claw migrate --preset user-data   # Migrate without secrets
hairball claw migrate --overwrite  # Overwrite existing conflicts
```

What gets imported:

- **SOUL.md** — persona file
- **Memories** — MEMORY.md and USER.md entries
- **Skills** — user-created skills → `~/.hairball/skills/openclaw-imports/`
- **Command allowlist** — approval patterns
- **Messaging settings** — platform configs, allowed users, working directory
- **API keys** — allowlisted secrets (Telegram, OpenRouter, OpenAI, Anthropic, ElevenLabs)
- **TTS assets** — workspace audio files
- **Workspace instructions** — AGENTS.md (with `--workspace-target`)

See `hairball claw migrate --help` for all options, or use the `openclaw-migration` skill for an interactive agent-guided migration with dry-run previews.

---

## Contributing

We welcome contributions! See the [Contributing Guide](./website/docs/developer-guide/contributing) for development setup, code style, and PR process.

Quick start for contributors — use the standard installer, then work from the
full git checkout it creates at `$HAIRBALL_HOME/hairball-agent` (usually
`~/.hairball/hairball-agent`). This matches the layout used by `hairball update`, the
managed venv, lazy dependencies, gateway, and docs tooling.

```bash
curl -fsSL ./scripts/install.sh | bash
cd "${HAIRBALL_HOME:-$HOME/.hairball}/hairball-agent"
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

Manual clone fallback (for throwaway clones/CI where you intentionally do not
want the managed install layout):

Create the venv outside the cloned source tree — a venv inside the directory
the agent operates from can be wiped by a relative-path command the agent runs
against its own checkout, destroying the running runtime mid-session.

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv ~/.hairball/venvs/hairball-dev --python 3.11
source ~/.hairball/venvs/hairball-dev/bin/activate
uv pip install -e ".[all,dev]"
scripts/run_tests.sh
```

---

## Community

- 💬 [Discord](#community)
- 📚 [Skills Hub](https://agentskills.io)
- 🐛 [Issues](hairball-agent)
- 🔌 [computer-use-linux](https://github.com/avifenesh/computer-use-linux) — Linux desktop-control MCP server for Hairball and other MCP hosts, with AT-SPI accessibility trees, Wayland/X11 input, screenshots, and compositor window targeting.
- 🔌 [HairballClaw](https://github.com/AaronWong1999/hairballclaw) — Community WeChat bridge: Run Hairball Agent and legacy on the same WeChat account.

---

## License

MIT — see [LICENSE](LICENSE).

maintained as Hairball Agent.
