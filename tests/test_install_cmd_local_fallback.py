from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
INSTALL_CMD = ROOT / "scripts" / "install.cmd"


def test_cmd_prefers_adjacent_powershell_installer_and_forwards_arguments() -> None:
    script = INSTALL_CMD.read_text(encoding="utf-8")

    local_check = 'if not exist "%~dp0install.ps1" goto online_installer'
    local_call = 'powershell -ExecutionPolicy ByPass -NoProfile -File "%~dp0install.ps1" %*'
    online_call = (
        'powershell -ExecutionPolicy ByPass -NoProfile -Command '
        '"iex (irm https://raw.githubusercontent.com/hairball-agent/'
        'hairball-agent/main/scripts/install.ps1)"'
    )

    assert script.index(local_check) < script.index(local_call)
    assert script.index(local_call) < script.index(":online_installer")
    assert script.index(":online_installer") < script.index(online_call)
    assert script.count('set "INSTALL_EXIT=%ERRORLEVEL%"') == 2
    assert "exit /b %INSTALL_EXIT%" in script
    assert "pause" not in script.lower()
