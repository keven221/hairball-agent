"""
Hairball Agent Uninstaller.

Provides options for:
- Full uninstall: Remove everything including configs and data
- Keep data: Remove code but keep ~/.hairball/ (configs, sessions, logs)
"""

import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from hairball_constants import get_hairball_home

from hairball_cli.colors import Colors, color

def log_info(msg: str):
    print(f"{color('→', Colors.CYAN)} {msg}")

def log_success(msg: str):
    print(f"{color('✓', Colors.GREEN)} {msg}")

def log_warn(msg: str):
    print(f"{color('⚠', Colors.YELLOW)} {msg}")


def _path_is_within(path: Path, parent: Path) -> bool:
    """Return whether *path* resolves inside *parent* without string-prefix traps."""
    try:
        path.resolve().relative_to(parent.resolve())
        return True
    except (OSError, ValueError):
        return False


def _should_handoff_windows_uninstall(project_root: Path) -> bool:
    """True when this Windows process is executing from the tree it must delete."""
    if not _is_windows() or os.environ.get("HAIRBALL_UNINSTALL_WORKER") == "1":
        return False
    return _path_is_within(Path(sys.executable), project_root)


def _python_version_is_supported(candidate: Path) -> bool:
    """Probe a real Python executable; never accepts the Microsoft Store shim."""
    try:
        completed = subprocess.run(
            [
                str(candidate),
                "-c",
                "import sys; print(sys.executable); "
                "raise SystemExit(0 if (3, 11) <= sys.version_info[:2] < (3, 14) else 1)",
            ],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            timeout=8,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return completed.returncode == 0


def _find_external_windows_python(project_root: Path) -> "Path | None":
    """Find Python 3.11-3.13 outside the install tree for self-deletion.

    This is the Python equivalent of the existing desktop ``findSystemPython``
    ladder: venv base interpreter, PEP 514 registry, standard install paths,
    then ``py.exe`` with an explicit supported version.  Plain ``python`` on
    PATH is deliberately excluded because it may be the Microsoft Store shim.
    """
    if not _is_windows():
        return None

    candidates: list[Path] = []
    for raw in (
        getattr(sys, "_base_executable", None),
        str(Path(sys.base_prefix) / "python.exe") if sys.base_prefix else None,
    ):
        if raw:
            candidates.append(Path(raw))

    try:
        import winreg

        for hive in (winreg.HKEY_LOCAL_MACHINE, winreg.HKEY_CURRENT_USER):
            for version in ("3.11", "3.12", "3.13"):
                key_name = rf"SOFTWARE\Python\PythonCore\{version}\InstallPath"
                for access in (winreg.KEY_READ | getattr(winreg, "KEY_WOW64_64KEY", 0), winreg.KEY_READ):
                    try:
                        with winreg.OpenKey(hive, key_name, 0, access) as key:
                            install_path, _kind = winreg.QueryValueEx(key, None)
                        candidates.append(Path(install_path) / "python.exe")
                        break
                    except OSError:
                        continue
    except ImportError:
        pass

    program_files = Path(os.environ.get("ProgramFiles") or r"C:\Program Files")
    local_app_data = os.environ.get("LOCALAPPDATA")
    for compact in ("311", "312", "313"):
        candidates.append(program_files / f"Python{compact}" / "python.exe")
        if local_app_data:
            candidates.append(
                Path(local_app_data) / "Programs" / "Python" / f"Python{compact}" / "python.exe"
            )

    seen: set[str] = set()
    for candidate in candidates:
        key = os.path.normcase(os.path.abspath(str(candidate)))
        if key in seen:
            continue
        seen.add(key)
        if (
            candidate.is_file()
            and not _path_is_within(candidate, project_root)
            and "windowsapps" not in key.lower()
            and _python_version_is_supported(candidate)
        ):
            return candidate.resolve()

    py_launcher = shutil.which("py.exe")
    if py_launcher:
        for version in ("3.11", "3.12", "3.13"):
            try:
                resolved = subprocess.run(
                    [py_launcher, f"-{version}", "-c", "import sys; print(sys.executable)"],
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    timeout=8,
                    check=False,
                )
            except (OSError, subprocess.TimeoutExpired):
                continue
            candidate = Path(resolved.stdout.strip()) if resolved.returncode == 0 else Path()
            if (
                str(candidate)
                and candidate.is_file()
                and not _path_is_within(candidate, project_root)
                and _python_version_is_supported(candidate)
            ):
                return candidate.resolve()
    return None


def _windows_detached_flags() -> int:
    """Same no-console detached flags used by Hairball's existing process helpers."""
    return 0x00000200 | 0x00000008 | 0x08000000


def _launch_windows_uninstall_worker(
    *, project_root: Path, hairball_home: Path, full_uninstall: bool
) -> bool:
    """Hand self-deletion to an external Python and return after it is spawned."""
    external_python = _find_external_windows_python(project_root)
    if external_python is None:
        log_warn(
            "Cannot safely uninstall on Windows: no Python 3.11-3.13 exists "
            "outside the Hairball install tree. The installation was left intact."
        )
        return False

    # Disarm the gateway respawner while the dependency-rich venv is still
    # available.  The external worker intentionally stays stdlib-only.
    log_info("Stopping gateway services before Windows cleanup handoff...")
    uninstall_gateway_service()

    mode = "full" if full_uninstall else "lite"
    if full_uninstall:
        artifact_dir = Path(tempfile.gettempdir()) / "hairball-uninstall"
    else:
        artifact_dir = hairball_home / "logs"
    artifact_dir.mkdir(parents=True, exist_ok=True)
    stamp = f"{int(time.time())}-{os.getpid()}"
    log_path = artifact_dir / f"uninstall-{stamp}.log"
    status_path = artifact_dir / f"uninstall-{stamp}.json"

    env = os.environ.copy()
    env["HAIRBALL_HOME"] = str(hairball_home)
    env["HAIRBALL_UNINSTALL_WORKER"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    existing_pythonpath = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join(
        part for part in (str(project_root), existing_pythonpath) if part
    )
    cmd = [
        str(external_python),
        "-m",
        "hairball_cli.uninstall",
        "--mode",
        mode,
        "--worker",
        "--wait-pid",
        str(os.getpid()),
        "--status-file",
        str(status_path),
    ]

    try:
        log_file = open(log_path, "ab", buffering=0)
        try:
            proc = subprocess.Popen(
                cmd,
                cwd=tempfile.gettempdir(),
                stdin=subprocess.DEVNULL,
                stdout=log_file,
                stderr=subprocess.STDOUT,
                env=env,
                creationflags=_windows_detached_flags(),
                close_fds=True,
            )
        finally:
            log_file.close()
    except OSError as exc:
        log_warn(f"Could not start the Windows uninstall worker: {exc}")
        return False

    log_info(f"Windows cleanup worker started (PID {proc.pid})")
    log_info(f"Cleanup status: {status_path}")
    log_info("This command will now exit so Windows can release the venv files.")
    return True


def _wait_for_windows_pid_exit(pid: int, timeout: float = 60.0) -> bool:
    """Wait for the venv parent to exit, matching the existing desktop cleanup."""
    if pid <= 0:
        return True
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        try:
            result = subprocess.run(
                ["tasklist", "/NH", "/FI", f"PID eq {pid}"],
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                encoding="mbcs",
                errors="replace",
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return False
        rows = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        if not any(str(pid) in row.split() for row in rows):
            return True
        time.sleep(0.5)
    return False


def _stop_windows_install_processes(project_root: Path) -> None:
    """Stop process trees executing from this exact install root.

    Ported from the bounded CIM sweep in ``install.ps1``.  It disarms respawns
    through the normal gateway uninstall before this worker starts, then takes
    up to ten passes and requires three consecutive clean passes.
    """
    root = str(project_root.resolve()).rstrip("\\/") + "\\"
    quoted_root = root.replace("'", "''")
    worker_pid = os.getpid()
    script = rf"""
$root = '{quoted_root}'
$workerPid = {worker_pid}
$clean = 0
for ($sweep = 0; $sweep -lt 10 -and $clean -lt 3; $sweep++) {{
  $found = 0
  Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
    Where-Object {{
      $_.ProcessId -ne $PID -and $_.ProcessId -ne $workerPid -and (
        ($_.ExecutablePath -and $_.ExecutablePath.StartsWith($root, [System.StringComparison]::OrdinalIgnoreCase)) -or
        ($_.CommandLine -and $_.CommandLine.IndexOf($root, [System.StringComparison]::OrdinalIgnoreCase) -ge 0)
      )
    }} |
    ForEach-Object {{
      $found++
      Stop-Process -Id $_.ProcessId -Force -ErrorAction SilentlyContinue
    }}
  if ($found -eq 0) {{ $clean++ }} else {{ $clean = 0 }}
  Start-Sleep -Milliseconds 400
}}
"""
    try:
        subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=20,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        log_warn(f"Could not complete the Windows install-process sweep: {exc}")


def _make_tree_entry_writable(func, path, exc) -> None:
    """rmtree callback: clear Git/checkout read-only bits, then retry the call."""
    if isinstance(exc, tuple):
        exc = exc[1]
    if not isinstance(exc, OSError):
        raise exc
    try:
        os.chmod(path, os.stat(path).st_mode | stat.S_IWRITE | stat.S_IREAD)
    except OSError:
        pass
    parent = os.path.dirname(path)
    if parent:
        try:
            os.chmod(parent, os.stat(parent).st_mode | stat.S_IWRITE)
        except OSError:
            pass
    func(path)


def _remove_tree_with_retry(path: Path) -> bool:
    """Remove a tree with read-only recovery and bounded Windows lock retries."""
    if not path.exists():
        return True
    attempts = 10 if _is_windows() else 3
    for attempt in range(attempts):
        try:
            try:
                shutil.rmtree(path, onexc=_make_tree_entry_writable)
            except TypeError:
                shutil.rmtree(path, onerror=_make_tree_entry_writable)
            if not path.exists():
                return True
        except OSError:
            pass
        if not path.exists():
            return True
        if attempt < attempts - 1:
            time.sleep(min(1.0, 0.25 * (attempt + 1)))
    return not path.exists()

def get_project_root() -> Path:
    """Get the project installation directory."""
    return Path(__file__).parent.parent.resolve()


def find_shell_configs() -> list:
    """Find shell configuration files that might have PATH entries."""
    home = Path.home()
    configs = []
    
    candidates = [
        home / ".bashrc",
        home / ".bash_profile",
        home / ".profile",
        home / ".zshrc",
        home / ".zprofile",
    ]
    
    for config in candidates:
        if config.exists():
            configs.append(config)
    
    return configs


def remove_path_from_shell_configs():
    """Remove Hairball PATH entries from shell configuration files."""
    configs = find_shell_configs()
    removed_from = []
    
    for config_path in configs:
        try:
            content = config_path.read_text()
            original_content = content
            
            # Remove lines containing hairball-agent or hairball PATH entries
            new_lines = []
            skip_next = False
            
            for line in content.split('\n'):
                # Skip the "# Hairball Agent" comment and following line
                if '# Hairball Agent' in line or '# hairball-agent' in line:
                    skip_next = True
                    continue
                if skip_next and ('hairball' in line.lower() and 'PATH' in line):
                    skip_next = False
                    continue
                skip_next = False
                
                # Remove any PATH line containing hairball
                if 'hairball' in line.lower() and ('PATH=' in line or 'path=' in line.lower()):
                    continue
                    
                new_lines.append(line)
            
            new_content = '\n'.join(new_lines)
            
            # Clean up multiple blank lines
            while '\n\n\n' in new_content:
                new_content = new_content.replace('\n\n\n', '\n\n')
            
            if new_content != original_content:
                config_path.write_text(new_content)
                removed_from.append(config_path)
                
        except Exception as e:
            log_warn(f"Could not update {config_path}: {e}")
    
    return removed_from


def remove_wrapper_script():
    """Remove the hairball wrapper script if it exists."""
    wrapper_paths = [
        Path.home() / ".local" / "bin" / "hairball",
        Path("/usr/local/bin/hairball"),
    ]
    
    removed = []
    for wrapper in wrapper_paths:
        if wrapper.exists():
            try:
                # Check if it's our wrapper (contains hairball_cli reference)
                content = wrapper.read_text()
                if 'hairball_cli' in content or 'hairball-agent' in content:
                    wrapper.unlink()
                    removed.append(wrapper)
            except Exception as e:
                log_warn(f"Could not remove {wrapper}: {e}")
    
    return removed


def _node_symlink_candidate_dirs() -> "list[Path]":
    """Directories where the installer may have placed node/npm/npx symlinks."""
    dirs: list[Path] = [Path.home() / ".local" / "bin"]
    # Root FHS installs put links in /usr/local/bin.
    if sys.platform == "linux":
        dirs.append(Path("/usr/local/bin"))
    # Termux installs put links in $PREFIX/bin.
    prefix = os.environ.get("PREFIX", "")
    if prefix and "com.termux" in prefix:
        dirs.append(Path(prefix) / "bin")
    return dirs


def remove_node_symlinks(hairball_home: Path) -> list:
    """Remove the node/npm/npx symlinks the installer placed on PATH.

    The POSIX installer (``scripts/install.sh`` / ``scripts/lib/node-bootstrap.sh``)
    symlinks node/npm/npx into the same directory as the ``hairball`` command:

    - ``/usr/local/bin/`` on root FHS installs (Linux, uid 0)
    - ``$PREFIX/bin/`` on Termux
    - ``~/.local/bin/`` otherwise (the common non-root case)

    We check all candidate directories so that uninstall works regardless of
    how the install was done (e.g. a root FHS install that placed links in
    ``/usr/local/bin``, or an older install that used ``~/.local/bin`` before
    the FHS fix).  Only symlinks that resolve into this Hairball home's ``node``
    directory are removed — links the user has repointed elsewhere (nvm, fnm,
    etc.) are left untouched.
    """
    node_dir = (hairball_home / "node").resolve()
    removed = []

    for name in ("node", "npm", "npx"):
        for bin_dir in _node_symlink_candidate_dirs():
            link = bin_dir / name
            try:
                # Only act on symlinks — never delete a real binary the user put here.
                if not link.is_symlink():
                    continue

                # Resolve the link target and confirm it points into our node dir.
                # os.readlink + manual join handles broken (dangling) links too;
                # Path.resolve() on a dangling link still returns the target path.
                target = Path(os.readlink(link))
                if not target.is_absolute():
                    target = (link.parent / target)
                target = target.resolve()

                if target == node_dir or node_dir in target.parents:
                    link.unlink()
                    removed.append(link)
            except Exception as e:
                log_warn(f"Could not remove {link}: {e}")

    return removed


def uninstall_gateway_service():
    """Stop and uninstall the gateway service (systemd, launchd, Windows
    Scheduled Task / Startup folder) and kill any standalone gateway processes.

    Delegates to the gateway module which handles:
    - Linux: user + system systemd services (with proper DBUS env setup)
    - macOS: launchd plists
    - Windows: Scheduled Task + Startup-folder fallback, via ``gateway_windows``
    - All platforms: standalone ``hairball gateway run`` processes
    - Termux/Android: skips systemd (no systemd on Android), still kills standalone processes
    """
    import platform
    stopped_something = False

    # 1. Kill any standalone gateway processes (all platforms, including Termux)
    try:
        from hairball_cli.gateway import kill_gateway_processes, find_gateway_pids
        pids = find_gateway_pids()
        if pids:
            killed = kill_gateway_processes()
            if killed:
                log_success(f"Killed {killed} running gateway process(es)")
                stopped_something = True
    except Exception as e:
        log_warn(f"Could not check for gateway processes: {e}")

    system = platform.system()

    # Termux/Android has no systemd and no launchd — nothing left to do.
    prefix = os.getenv("PREFIX", "")
    is_termux = bool(os.getenv("TERMUX_VERSION") or "com.termux/files/usr" in prefix)
    if is_termux:
        return stopped_something

    # 2. Linux: uninstall systemd services (both user and system scopes)
    if system == "Linux":
        try:
            from hairball_cli.gateway import (
                get_systemd_unit_path,
                get_service_name,
                _systemctl_cmd,
            )
            svc_name = get_service_name()

            for is_system in (False, True):
                unit_path = get_systemd_unit_path(system=is_system)
                if not unit_path.exists():
                    continue

                scope = "system" if is_system else "user"
                try:
                    if is_system and os.geteuid() != 0:  # windows-footgun: ok — Linux systemd uninstall path, guarded by `if system == "Linux"` above
                        log_warn(f"System gateway service exists at {unit_path} "
                                 f"but needs sudo to remove")
                        continue

                    cmd = _systemctl_cmd(is_system)
                    subprocess.run(cmd + ["stop", svc_name],
                                   capture_output=True, check=False)
                    subprocess.run(cmd + ["disable", svc_name],
                                   capture_output=True, check=False)
                    unit_path.unlink()
                    subprocess.run(cmd + ["daemon-reload"],
                                   capture_output=True, check=False)
                    log_success(f"Removed {scope} gateway service ({unit_path})")
                    stopped_something = True
                except Exception as e:
                    log_warn(f"Could not remove {scope} gateway service: {e}")
        except Exception as e:
            log_warn(f"Could not check systemd gateway services: {e}")

    # 3. macOS: uninstall launchd plist
    elif system == "Darwin":
        try:
            from hairball_cli.gateway import get_launchd_plist_path
            plist_path = get_launchd_plist_path()
            if plist_path.exists():
                subprocess.run(["launchctl", "unload", str(plist_path)],
                               capture_output=True, check=False)
                plist_path.unlink()
                log_success(f"Removed macOS gateway service ({plist_path})")
                stopped_something = True
        except Exception as e:
            log_warn(f"Could not remove launchd gateway service: {e}")

    # 4. Windows: uninstall Scheduled Task + Startup-folder entry.  The
    #    gateway_windows module already knows how to locate and remove both
    #    code paths (schtasks /Delete + .cmd unlink) and how to stop any
    #    running detached pythonw gateway process.  We call into it so the
    #    uninstall logic stays in exactly one place.
    elif system == "Windows":
        try:
            from hairball_cli import gateway_windows
            if gateway_windows.is_installed() or gateway_windows.is_task_registered() \
                    or gateway_windows.is_startup_entry_installed():
                try:
                    gateway_windows.stop()
                except Exception as e:
                    log_warn(f"Could not stop Windows gateway cleanly: {e}")
                try:
                    gateway_windows.uninstall()
                    log_success("Removed Windows gateway (Scheduled Task + Startup entry)")
                    stopped_something = True
                except Exception as e:
                    log_warn(f"Could not fully uninstall Windows gateway: {e}")
        except Exception as e:
            log_warn(f"Could not check Windows gateway service: {e}")

    return stopped_something


# ============================================================================
# Windows-specific uninstall helpers
# ============================================================================
#
# The installer (``scripts/install.ps1``) does four Windows-only things that
# ``remove_path_from_shell_configs`` / ``remove_wrapper_script`` don't cover:
#
#   1. Sets User-scope env vars ``HAIRBALL_HOME`` and ``HAIRBALL_GIT_BASH_PATH``
#      via ``[Environment]::SetEnvironmentVariable(..., "User")``.  These
#      don't live in ~/.bashrc — they're in the Windows registry at
#      HKCU\Environment.
#   2. Prepends to User-scope ``PATH`` (same registry location) entries
#      like ``%LOCALAPPDATA%\hairball\git\cmd``, ``%LOCALAPPDATA%\hairball\git\bin``,
#      ``%LOCALAPPDATA%\hairball\git\usr\bin``, ``%LOCALAPPDATA%\hairball\node``.
#      Again not in any rc file — only accessible via the registry or the
#      .NET [Environment] API.
#   3. Downloads PortableGit to ``%LOCALAPPDATA%\hairball\git\`` and Node to
#      ``%LOCALAPPDATA%\hairball\node\`` as user-scoped, isolated copies.
#      These are ~200MB combined and serve no purpose after uninstall.
#   4. On the ``hairball dashboard`` + gateway paths, drops files into
#      ``%LOCALAPPDATA%\hairball\gateway-service\`` and sometimes
#      ``%APPDATA%\Microsoft\Windows\Start Menu\Programs\Startup\`` — the
#      latter is handled by ``gateway_windows.uninstall()`` already.
#
# Running a PowerShell one-liner per operation is overkill and fragile on
# locked-down machines (Constrained Language Mode, restricted ExecutionPolicy).
# Direct registry writes via ``winreg`` work without spawning any subprocess
# and apply immediately for new shells (SendMessage WM_SETTINGCHANGE would
# be nicer but requires ctypes and buys us nothing — the user will log out
# or open a new terminal anyway).


def _hairball_path_markers(hairball_home: Path) -> list[str]:
    """Path-entry substrings that identify Hairball-owned User-PATH entries."""
    root = str(hairball_home).rstrip("\\/")
    # Match on prefix so sub-entries (git\cmd, git\bin, git\usr\bin, node, etc.)
    # all get swept.  Also match the bare hairball-agent install dir.
    markers = [root + "\\hairball-agent", root + "\\git", root + "\\node", root + "\\venv"]
    # Also match if HAIRBALL_HOME was customised to somewhere else — find-and-nuke
    # any entry whose path component contains "hairball".  We don't want to catch
    # unrelated entries like "chairball-foo" or "ephermeral", so we look for
    # backslash-hairball as a word-ish boundary.
    return markers


def remove_path_from_windows_registry(hairball_home: Path) -> list[str]:
    """Strip Hairball-owned entries from User-scope PATH in the registry.

    Returns the list of removed path entries.  Operates on HKCU\\Environment,
    same key the installer wrote to via ``[Environment]::SetEnvironmentVariable``.
    """
    try:
        import winreg
    except ImportError:
        return []  # not on Windows, nothing to do

    removed: list[str] = []
    key_path = "Environment"
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key_path, 0,
                            winreg.KEY_READ | winreg.KEY_WRITE) as key:
            try:
                path_value, path_type = winreg.QueryValueEx(key, "Path")
            except FileNotFoundError:
                return []
            # Preserve REG_EXPAND_SZ vs REG_SZ so unexpanded %VARS% survive.
            entries = [e for e in path_value.split(";") if e]
            markers = _hairball_path_markers(hairball_home)
            kept: list[str] = []
            for entry in entries:
                entry_norm = entry.rstrip("\\/")
                matched = any(entry_norm.lower().startswith(m.lower()) for m in markers)
                if matched:
                    removed.append(entry)
                else:
                    kept.append(entry)
            if removed:
                new_value = ";".join(kept)
                winreg.SetValueEx(key, "Path", 0, path_type, new_value)
    except OSError as e:
        log_warn(f"Could not edit User PATH in registry: {e}")
    return removed


def remove_hairball_env_vars_windows() -> list[str]:
    """Delete HAIRBALL_HOME and HAIRBALL_GIT_BASH_PATH from User-scope env vars."""
    try:
        import winreg
    except ImportError:
        return []

    removed: list[str] = []
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, "Environment", 0,
                            winreg.KEY_READ | winreg.KEY_WRITE) as key:
            for name in ("HAIRBALL_HOME", "HAIRBALL_GIT_BASH_PATH"):
                try:
                    winreg.QueryValueEx(key, name)
                except FileNotFoundError:
                    continue
                try:
                    winreg.DeleteValue(key, name)
                    removed.append(name)
                except OSError as e:
                    log_warn(f"Could not delete {name} from User env: {e}")
    except OSError as e:
        log_warn(f"Could not open User Environment key: {e}")
    return removed


def remove_portable_tooling_windows(hairball_home: Path) -> list[Path]:
    """Delete PortableGit and Node installs the Windows installer created under
    ``%LOCALAPPDATA%\\hairball\\``.  Only called on full uninstall; they're
    isolated from any system Git / Node so they cannot break other tools."""
    removed: list[Path] = []
    for sub in ("git", "node", "gateway-service"):
        target = hairball_home / sub
        if target.exists():
            try:
                if not _remove_tree_with_retry(target):
                    raise OSError(f"directory still exists after bounded retries: {target}")
                removed.append(target)
            except Exception as e:
                log_warn(f"Could not remove {target}: {e}")
    return removed


def _is_windows() -> bool:
    import sys
    return sys.platform == "win32"


def _is_default_hairball_home(hairball_home: Path) -> bool:
    """Return True when ``hairball_home`` points at the default (non-profile) root."""
    try:
        from hairball_constants import get_default_hairball_root
        return hairball_home.resolve() == get_default_hairball_root().resolve()
    except Exception:
        return False


def _discover_named_profiles():
    """Return a list of ``ProfileInfo`` for every non-default profile, or ``[]``
    if profile support is unavailable or nothing is installed beyond the
    default root."""
    try:
        from hairball_cli.profiles import list_profiles
    except Exception:
        return []
    try:
        return [p for p in list_profiles() if not getattr(p, "is_default", False)]
    except Exception as e:
        log_warn(f"Could not enumerate profiles: {e}")
        return []


def _uninstall_profile(profile) -> None:
    """Fully uninstall a single named profile: stop its gateway service,
    remove its alias wrapper, and wipe its HAIRBALL_HOME directory.

    We shell out to ``hairball -p <name> gateway stop|uninstall`` because
    service names, unit paths, and plist paths are all derived from the
    current HAIRBALL_HOME and can't be easily switched in-process.
    """
    import sys as _sys
    name = profile.name
    profile_home = profile.path

    log_info(f"Uninstalling profile '{name}'...")

    # 1. Stop and remove this profile's gateway service.
    #    Use `python -m hairball_cli.main` so we don't depend on a `hairball`
    #    wrapper that may be half-removed mid-uninstall.
    hairball_invocation = [_sys.executable, "-m", "hairball_cli.main", "--profile", name]
    for subcmd in ("stop", "uninstall"):
        try:
            subprocess.run(
                hairball_invocation + ["gateway", subcmd],
                capture_output=True,
                text=True,
                timeout=60,
                check=False,
            )
        except subprocess.TimeoutExpired:
            log_warn(f"  Gateway {subcmd} timed out for '{name}'")
        except Exception as e:
            log_warn(f"  Could not run gateway {subcmd} for '{name}': {e}")

    # 2. Remove the wrapper alias script at ~/.local/bin/<name> (if any).
    alias_path = getattr(profile, "alias_path", None)
    if alias_path and alias_path.exists():
        try:
            alias_path.unlink()
            log_success(f"  Removed alias {alias_path}")
        except Exception as e:
            log_warn(f"  Could not remove alias {alias_path}: {e}")

    # 3. Wipe the profile's HAIRBALL_HOME directory.
    try:
        if profile_home.exists():
            shutil.rmtree(profile_home)
            log_success(f"  Removed {profile_home}")
    except Exception as e:
        log_warn(f"  Could not remove {profile_home}: {e}")


def run_gui_uninstall(args):
    """GUI-only uninstall: remove the Chat GUI, leave the agent + data intact.

    Mirrors ``hairball uninstall --gui``. Removes the desktop app's built
    artifacts, the packaged app bundle (best-effort), and the Electron
    userData dir — nothing under ``$HAIRBALL_HOME`` config/sessions/.env, and
    never the Python agent or its venv.
    """
    from hairball_cli.gui_uninstall import (
        agent_is_installed,
        gui_install_summary,
        uninstall_gui,
    )

    hairball_home = get_hairball_home()
    summary = gui_install_summary(hairball_home)
    skip_confirm = bool(getattr(args, "yes", False))

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA, Colors.BOLD))
    print(color("│         ฅ Hairball Chat GUI Uninstaller                  │", Colors.MAGENTA, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA, Colors.BOLD))
    print()

    if not summary["gui_installed"]:
        print("No Hairball Chat GUI installation was found.")
        print(f"  Checked: {hairball_home}, and the standard app locations for this OS.")
        return

    print(color("This removes the Chat GUI only. The Hairball agent stays installed.", Colors.CYAN))
    print()
    print(color("Will remove:", Colors.YELLOW, Colors.BOLD))
    for p in summary["source_built_artifacts"]:
        print(f"  • {p}")
    for p in summary["packaged_app_paths"]:
        print(f"  • {p}")
    if summary["userdata_exists"]:
        print(f"  • {summary['userdata_dir']}  (desktop app data)")
    print()
    if agent_is_installed(hairball_home):
        print(color("Kept intact:", Colors.GREEN, Colors.BOLD))
        print(f"  • The Hairball agent at {hairball_home / 'hairball-agent'}")
        print(f"  • Your config, sessions, and secrets under {hairball_home}")
        print()

    if not skip_confirm:
        try:
            confirm = input(f"Type '{color('yes', Colors.YELLOW)}' to remove the Chat GUI: ").strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            print("Cancelled.")
            return
        if confirm != "yes":
            print()
            print("Uninstall cancelled.")
            return

    print()
    print(color("Uninstalling Chat GUI...", Colors.CYAN, Colors.BOLD))
    print()
    uninstall_gui(hairball_home)

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.GREEN, Colors.BOLD))
    print(color("│            ✓ Chat GUI Uninstalled!                      │", Colors.GREEN, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.GREEN, Colors.BOLD))
    print()
    print("The Hairball agent is still installed. Run 'hairball' to use the CLI,")
    print("or 'hairball uninstall' to remove the agent too.")
    print()


def run_uninstall(args):
    """
    Run the uninstall process.
    
    Options:
    - Full uninstall: removes code + ~/.hairball/ (configs, data, logs)
    - Keep data: removes code but keeps ~/.hairball/ for future reinstall
    """
    project_root = get_project_root()
    hairball_home = get_hairball_home()

    if bool(getattr(args, "dry_run", False)):
        _print_uninstall_dry_run(
            project_root=project_root,
            hairball_home=hairball_home,
            full_uninstall=bool(getattr(args, "full", False)),
        )
        return

    # Detect named profiles when uninstalling from the default root —
    # offer to clean them up too instead of leaving zombie HAIRBALL_HOMEs
    # and systemd units behind.
    is_default_profile = _is_default_hairball_home(hairball_home)
    named_profiles = _discover_named_profiles() if is_default_profile else []

    # Non-interactive fast path (``--yes``): no prompts. ``--full`` selects a
    # full wipe (code + ~/.hairball data); otherwise keep-data. Named profiles
    # are NOT auto-removed here — that's a destructive, surprising default for
    # an unattended run, so it stays opt-in to the interactive flow. This is
    # the path the desktop app's detached cleanup script uses for its
    # lite/full modes.
    skip_confirm = bool(getattr(args, "yes", False))
    if skip_confirm:
        full_uninstall = bool(getattr(args, "full", False))
        if _should_handoff_windows_uninstall(project_root):
            log_info("Preparing Windows self-uninstall handoff...")
            return _launch_windows_uninstall_worker(
                project_root=project_root,
                hairball_home=hairball_home,
                full_uninstall=full_uninstall,
            )
        return _perform_uninstall(
            project_root=project_root,
            hairball_home=hairball_home,
            full_uninstall=full_uninstall,
            remove_profiles=False,
            named_profiles=named_profiles,
            gateway_prepared=bool(getattr(args, "worker", False)),
        )

    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.MAGENTA, Colors.BOLD))
    print(color("│            ฅ Hairball Agent Uninstaller                  │", Colors.MAGENTA, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.MAGENTA, Colors.BOLD))
    print()
    
    # Show what will be affected
    print(color("Current Installation:", Colors.CYAN, Colors.BOLD))
    print(f"  Code:    {project_root}")
    print(f"  Config:  {hairball_home / 'config.yaml'}")
    print(f"  Secrets: {hairball_home / '.env'}")
    print(f"  Data:    {hairball_home / 'cron/'}, {hairball_home / 'sessions/'}, {hairball_home / 'logs/'}")
    print()

    if named_profiles:
        print(color("Other profiles detected:", Colors.CYAN, Colors.BOLD))
        for p in named_profiles:
            running = " (gateway running)" if getattr(p, "gateway_running", False) else ""
            print(f"  • {p.name}{running}: {p.path}")
        print()
    
    # Ask for confirmation
    print(color("Uninstall Options:", Colors.YELLOW, Colors.BOLD))
    print()
    print("  1) " + color("Keep data", Colors.GREEN) + " - Remove code only, keep configs/sessions/logs")
    print("     (Recommended - you can reinstall later with your settings intact)")
    print()
    print("  2) " + color("Full uninstall", Colors.RED) + " - Remove everything including all data")
    print("     (Warning: This deletes all configs, sessions, and logs permanently)")
    print()
    print("  3) " + color("Cancel", Colors.CYAN) + " - Don't uninstall")
    print()
    
    try:
        choice = input(color("Select option [1/2/3]: ", Colors.BOLD)).strip()
    except (KeyboardInterrupt, EOFError):
        print()
        print("Cancelled.")
        return
    
    if choice == "3" or choice.lower() in {"c", "cancel", "q", "quit", "n", "no"}:
        print()
        print("Uninstall cancelled.")
        return
    
    full_uninstall = (choice == "2")

    # When doing a full uninstall from the default profile, also offer to
    # remove any named profiles — stopping their gateway services, unlinking
    # their alias wrappers, and wiping their HAIRBALL_HOME dirs. Otherwise
    # those leave zombie services and data behind.
    remove_profiles = False
    if full_uninstall and named_profiles:
        print()
        print(color("Other profiles will NOT be removed by default.", Colors.YELLOW))
        print(f"Found {len(named_profiles)} named profile(s): " +
              ", ".join(p.name for p in named_profiles))
        print()
        try:
            resp = input(color(
                f"Also stop and remove these {len(named_profiles)} profile(s)? [y/N]: ",
                Colors.BOLD
            )).strip().lower()
        except (KeyboardInterrupt, EOFError):
            print()
            print("Cancelled.")
            return
        remove_profiles = resp in {"y", "yes"}

    # Final confirmation
    print()
    if full_uninstall:
        print(color("⚠️  WARNING: This will permanently delete ALL Hairball data!", Colors.RED, Colors.BOLD))
        print(color("   Including: configs, API keys, sessions, scheduled jobs, logs", Colors.RED))
        if remove_profiles:
            print(color(
                f"   Plus {len(named_profiles)} profile(s): " +
                ", ".join(p.name for p in named_profiles),
                Colors.RED
            ))
    else:
        print("This will remove the Hairball code but keep your configuration and data.")
    
    print()
    try:
        confirm = input(f"Type '{color('yes', Colors.YELLOW)}' to confirm: ").strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        print("Cancelled.")
        return
    
    if confirm != "yes":
        print()
        print("Uninstall cancelled.")
        return

    if _should_handoff_windows_uninstall(project_root):
        log_info("Preparing Windows self-uninstall handoff...")
        return _launch_windows_uninstall_worker(
            project_root=project_root,
            hairball_home=hairball_home,
            full_uninstall=full_uninstall,
        )

    return _perform_uninstall(
        project_root=project_root,
        hairball_home=hairball_home,
        full_uninstall=full_uninstall,
        remove_profiles=remove_profiles,
        named_profiles=named_profiles,
        gateway_prepared=bool(getattr(args, "worker", False)),
    )


def _print_uninstall_dry_run(*, project_root: Path, hairball_home: Path, full_uninstall: bool) -> None:
    """Print the uninstall plan without stopping services or deleting files."""
    print()
    print(color("Dry run: no files, services, or environment entries will be changed.", Colors.CYAN, Colors.BOLD))
    print()
    print(color("Would inspect/remove:", Colors.YELLOW, Colors.BOLD))
    print("  • Gateway services and standalone gateway processes")
    print("  • Hairball PATH entries from shell configs / Windows User PATH")
    print("  • Hairball wrapper scripts and Hairball-managed node/npm/npx symlinks")
    print("  • Desktop Chat GUI artifacts")
    print(f"  • Code checkout: {project_root}")
    if full_uninstall:
        print(f"  • Hairball config/data: {hairball_home}")
        if _is_default_hairball_home(hairball_home):
            profiles = _discover_named_profiles()
            if profiles:
                print("  • Named profiles (interactive uninstall asks before removing):")
                for prof in profiles:
                    print(f"    - {prof.name}: {prof.path}")
    else:
        print(f"  • Keep Hairball config/data: {hairball_home}")
    print()


def _perform_uninstall(
    *,
    project_root: Path,
    hairball_home: Path,
    full_uninstall: bool,
    remove_profiles: bool,
    named_profiles: list,
    gateway_prepared: bool = False,
) -> bool:
    """Execute the uninstall steps. Shared by the interactive and ``--yes``
    paths so the destructive sequence lives in exactly one place.

    Steps: stop gateway → strip PATH (rc files + Windows registry) → remove the
    ``hairball`` wrapper + node symlinks → remove the desktop Chat GUI artifacts →
    delete the code checkout → (Windows) remove PortableGit/Node → optionally
    wipe ``$HAIRBALL_HOME`` data and named profiles on full uninstall.
    """
    print()
    print(color("Uninstalling...", Colors.CYAN, Colors.BOLD))
    print()
    
    # 1. Stop and uninstall gateway service + kill standalone processes
    if gateway_prepared:
        log_info("Gateway cleanup completed before Windows handoff")
    else:
        log_info("Checking for running gateway...")
        if not uninstall_gateway_service():
            log_info("No gateway service or processes found")
    
    # 2. Remove PATH entries from shell configs (POSIX) AND from the Windows
    #    User-scope registry.  Both helpers no-op on the wrong platform so we
    #    can safely call them unconditionally.
    log_info("Removing PATH entries from shell configs...")
    removed_configs = remove_path_from_shell_configs()
    if removed_configs:
        for config in removed_configs:
            log_success(f"Updated {config}")
    else:
        log_info("No PATH entries found to remove in shell rc files")

    if _is_windows():
        log_info("Removing PATH entries from Windows User environment...")
        # Expand %LOCALAPPDATA% etc. in hairball_home so the marker matching is
        # against fully resolved paths — installer writes literal strings
        # like C:\Users\<u>\AppData\Local\hairball\git\cmd, not %LOCALAPPDATA%.
        removed_path_entries = remove_path_from_windows_registry(Path(os.path.expandvars(str(hairball_home))))
        if removed_path_entries:
            for entry in removed_path_entries:
                log_success(f"Removed from User PATH: {entry}")
        else:
            log_info("No Hairball-owned PATH entries in User environment")

        log_info("Removing HAIRBALL_HOME / HAIRBALL_GIT_BASH_PATH User env vars...")
        removed_env = remove_hairball_env_vars_windows()
        if removed_env:
            for name in removed_env:
                log_success(f"Removed User env var: {name}")
        else:
            log_info("No Hairball-set User env vars to remove")
    
    # 3. Remove wrapper script
    log_info("Removing hairball command...")
    removed_wrappers = remove_wrapper_script()
    if removed_wrappers:
        for wrapper in removed_wrappers:
            log_success(f"Removed {wrapper}")
    else:
        log_info("No wrapper script found")

    # 3b. Remove node/npm/npx symlinks the installer left in ~/.local/bin
    #     (only when they still point into this Hairball home's node dir, so we
    #     never clobber an existing nvm / user-managed Node).
    log_info("Removing Hairball-managed node/npm/npx symlinks...")
    removed_node_links = remove_node_symlinks(hairball_home)
    if removed_node_links:
        for link in removed_node_links:
            log_success(f"Removed {link}")
    else:
        log_info("No Hairball-managed node/npm/npx symlinks found")

    # 3c. Remove the desktop Chat GUI's artifacts too (built renderer/release,
    #     node_modules, the packaged app bundle, and the Electron userData
    #     dir). Both the "keep data" and "full" CLI flows remove the agent
    #     code, so the GUI — which is just another consumer of the same
    #     checkout — should go with it. uninstall_gui() never touches config /
    #     sessions / .env, so it's safe in keep-data mode; on full uninstall the
    #     step-5 rmtree(hairball_home) would sweep the in-tree artifacts anyway,
    #     but the packaged app + Electron userData live OUTSIDE HAIRBALL_HOME and
    #     must be cleaned explicitly here.
    log_info("Removing desktop Chat GUI artifacts...")
    try:
        from hairball_cli.gui_uninstall import uninstall_gui
        gui_removed = uninstall_gui(hairball_home)
        if not gui_removed:
            log_info("No desktop GUI artifacts found")
    except Exception as e:
        log_warn(f"Could not remove desktop GUI artifacts: {e}")

    # 4. Remove installation directory (code)
    log_info("Removing installation directory...")
    
    # Check if we're running from within the install dir
    # We need to be careful here
    if _remove_tree_with_retry(project_root):
        log_success(f"Removed {project_root}")
    else:
        log_warn(f"Could not fully remove {project_root} after bounded retries")

    # 4b. Remove Windows-only installer artifacts that are NOT user data:
    #     PortableGit, bundled Node, gateway-service dir.  Installer put them
    #     under HAIRBALL_HOME but they're install tooling, not config — safe to
    #     remove even in "keep data" mode.  If we're doing a full uninstall
    #     the step-5 rmtree(hairball_home) would sweep them anyway; calling
    #     this helper there is a no-op since they'll already be gone.
    if _is_windows():
        log_info("Removing Windows installer artifacts (PortableGit, Node, gateway-service)...")
        removed_artifacts = remove_portable_tooling_windows(hairball_home)
        if removed_artifacts:
            for path in removed_artifacts:
                log_success(f"Removed {path}")
        else:
            log_info("No Windows installer artifacts to remove")
    
    # 5. Optionally remove ~/.hairball/ data directory (and named profiles)
    if full_uninstall:
        # 5a. Stop and remove each named profile's gateway service and
        #     alias wrapper. The profile HAIRBALL_HOME dirs live under
        #     ``<default>/profiles/<name>/`` and will be swept away by the
        #     rmtree below, but services + alias scripts live OUTSIDE the
        #     default root and have to be cleaned up explicitly.
        if remove_profiles and named_profiles:
            for prof in named_profiles:
                _uninstall_profile(prof)

        log_info("Removing configuration and data...")
        if _remove_tree_with_retry(hairball_home):
            log_success(f"Removed {hairball_home}")
        else:
            log_warn(f"Could not fully remove {hairball_home} after bounded retries")
    else:
        log_info(f"Keeping configuration and data in {hairball_home}")
    
    required_absent = [project_root]
    if _is_windows():
        required_absent.extend(
            hairball_home / sub for sub in ("git", "node", "gateway-service")
        )
    if full_uninstall:
        required_absent.append(hairball_home)
    remaining = [path for path in required_absent if path.exists()]
    if remaining:
        print()
        print(color("Uninstall incomplete.", Colors.RED, Colors.BOLD))
        for path in remaining:
            log_warn(f"Still present: {path}")
        print("Close any remaining Hairball process and run uninstall again.")
        return False

    # Done
    print()
    print(color("┌─────────────────────────────────────────────────────────┐", Colors.GREEN, Colors.BOLD))
    print(color("│              ✓ Uninstall Complete!                      │", Colors.GREEN, Colors.BOLD))
    print(color("└─────────────────────────────────────────────────────────┘", Colors.GREEN, Colors.BOLD))
    print()
    
    if not full_uninstall:
        print(color("Your configuration and data have been preserved:", Colors.CYAN))
        print(f"  {hairball_home}/")
        print()
        print("To reinstall later with your existing settings:")
        if _is_windows():
            print(color("  iex (irm ./website/install.ps1)", Colors.DIM))
        else:
            print(color("  curl -fsSL ./website/install.sh | bash", Colors.DIM))
        print()

    if _is_windows():
        print(color("Open a new terminal (PowerShell / Windows Terminal) to pick up", Colors.YELLOW))
        print(color("the updated User PATH and environment variables.", Colors.YELLOW))
    else:
        print(color("Reload your shell to complete the process:", Colors.YELLOW))
        print("  source ~/.bashrc  # or ~/.zshrc")
    print()
    print("Thank you for using Hairball Agent! ฅ")
    print()
    return True


class _UninstallArgs:
    """Lightweight args namespace for the module entrypoint below."""

    def __init__(self, *, mode: str, worker: bool = False):
        self.gui = mode == "gui"
        self.gui_summary = False
        self.full = mode == "full"
        self.yes = True  # the module entrypoint is always non-interactive
        self.dry_run = False
        self.worker = worker


def _write_uninstall_worker_status(
    status_file: "str | None", *, success: bool, error: "str | None" = None
) -> None:
    if not status_file:
        return
    path = Path(status_file)
    payload = {
        "schema_version": 1,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "success": bool(success),
        "error": error,
    }
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    except OSError as exc:
        log_warn(f"Could not write uninstall status {path}: {exc}")


def main(argv=None) -> int:
    """Module entrypoint: ``python -m hairball_cli.uninstall --mode <gui|lite|full>``.

    Exists so the desktop app can run the uninstall under a Python interpreter
    OUTSIDE the venv being deleted. On Windows, ``lite``/``full`` rmtree the
    venv that contains the running ``python.exe`` — and a running .exe is
    mandatory-locked, so doing that from the venv's own interpreter half-fails.
    The desktop launches this with the system Python + ``PYTHONPATH=<agentRoot>``
    so ``import hairball_cli`` resolves from source while the venv is torn down.

    This module imports only stdlib + ``hairball_constants`` + ``hairball_cli.colors``
    (and lazily ``hairball_cli.gui_uninstall``), so it runs fine under a bare
    system Python with no site-packages from the venv.
    """
    import argparse

    parser = argparse.ArgumentParser(prog="python -m hairball_cli.uninstall")
    parser.add_argument(
        "--mode",
        choices=["gui", "lite", "full"],
        required=True,
        help="gui = Chat GUI only; lite = GUI + agent, keep data; full = everything",
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--wait-pid", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--status-file", default=None, help=argparse.SUPPRESS)
    ns = parser.parse_args(argv)
    args = _UninstallArgs(mode=ns.mode, worker=bool(ns.worker))

    if ns.worker:
        if not _is_windows():
            _write_uninstall_worker_status(
                ns.status_file, success=False, error="Windows uninstall worker used off Windows"
            )
            return 2
        if not _wait_for_windows_pid_exit(int(ns.wait_pid or 0)):
            message = f"Timed out waiting for uninstall parent PID {ns.wait_pid}"
            log_warn(message)
            _write_uninstall_worker_status(ns.status_file, success=False, error=message)
            return 1
        _stop_windows_install_processes(get_project_root())

    if args.gui:
        run_gui_uninstall(args)
        ok = True
    else:
        ok = bool(run_uninstall(args))
    _write_uninstall_worker_status(
        ns.status_file,
        success=ok,
        error=None if ok else "Uninstall left required installation paths behind",
    )
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
