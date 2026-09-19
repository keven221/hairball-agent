"""Windows browser installer must time-box the Chromium child process tree."""

from __future__ import annotations

import re
from pathlib import Path


INSTALL_PS1 = Path(__file__).resolve().parents[1] / "scripts" / "install.ps1"


def _source() -> str:
    return INSTALL_PS1.read_text(encoding="utf-8")


def test_windows_agent_browser_install_uses_timeout_wrapper() -> None:
    text = _source()
    install = re.search(
        r"^function Install-AgentBrowser \{(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert install is not None
    body = install.group("body")
    chromium_tail = body[body.index("$abExe =") :]

    assert "Invoke-AgentBrowserInstallWithTimeout" in chromium_tail
    assert "& $abExe install" not in chromium_tail
    assert "-TimeoutSeconds 600" in chromium_tail
    assert "$abResult.TimedOut" in chromium_tail


def test_windows_playwright_install_uses_same_timeout_wrapper() -> None:
    """The full installer must not hang forever in Playwright's downloader."""
    text = _source()
    install = re.search(
        r"^function Install-NodeDeps \{(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert install is not None
    body = install.group("body")
    playwright_tail = body[body.index("$pwLog =") :]

    assert "Invoke-AgentBrowserInstallWithTimeout" in playwright_tail
    assert '-CommandArguments "--yes playwright install chromium"' in playwright_tail
    assert "-TimeoutSeconds 600" in playwright_tail
    assert "-StreamOutput" in playwright_tail
    assert "$pwResult.TimedOut" in playwright_tail
    assert "& $npxExe --yes playwright install chromium" not in playwright_tail


def test_windows_timeout_stops_entire_process_tree() -> None:
    text = _source()
    helper = re.search(
        r"^function Invoke-AgentBrowserInstallWithTimeout \{(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert helper is not None
    body = helper.group("body")

    assert "$proc.WaitForExit(" in body
    assert "taskkill.exe" in body
    assert "& $taskkill /PID $proc.Id /T /F" in body
    assert "Stop-Process -Id $proc.Id -Force" in body
    assert "ExitCode = 124; TimedOut = $true" in body


def test_windows_timeout_runs_batch_shim_through_cmd() -> None:
    text = _source()
    assert "$commandLine = \"/d /s /c" in text
    assert "FilePath = $env:ComSpec" in text
    assert "$proc = Start-Process @startParams" in text
    assert "$CommandArguments > `\"$escapedLog`\" 2>&1" in text


def test_windows_timeout_wrapper_can_stream_and_set_working_directory() -> None:
    text = _source()
    helper = re.search(
        r"^function Invoke-AgentBrowserInstallWithTimeout \{(?P<body>.*?)^\}",
        text,
        re.MULTILINE | re.DOTALL,
    )
    assert helper is not None
    body = helper.group("body")

    assert "[string]$CommandArguments = \"install\"" in body
    assert "[string]$WorkingDirectory" in body
    assert "[switch]$StreamOutput" in body
    assert "$startParams.WorkingDirectory = $WorkingDirectory" in body
    assert "Write-Host $line" in body
