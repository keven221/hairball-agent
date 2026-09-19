"""DAP adapter discovery (W3+ — debugpy / js-debug / lldb / gdb).

Reference: Oh-My-Pi ``packages/coding-agent/src/dap/config.ts`` + ``defaults.json``.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Optional

JS_DEBUG_EXTS = frozenset({".js", ".jsx", ".ts", ".tsx", ".mjs", ".cjs"})
NATIVE_EXTS = frozenset(
    {
        ".c",
        ".cc",
        ".cpp",
        ".cxx",
        ".h",
        ".hh",
        ".hpp",
        ".m",
        ".mm",
        ".rs",
        ".swift",
        ".zig",
        # compiled binaries often have no suffix; callers pass adapter= explicitly
    }
)
ADAPTER_DEBUGPY = "debugpy"
ADAPTER_JS_DEBUG = "js-debug"
ADAPTER_LLDB = "lldb"
ADAPTER_GDB = "gdb"


def find_node() -> Optional[str]:
    return shutil.which("node")


def resolve_js_debug_server(cwd: Optional[str] = None) -> Optional[Path]:
    """Return path to ``dapDebugServer.js`` if present.

    Search order (OMP-aligned + Hairball env):
    1. ``HAIRBALL_JS_DEBUG_DAP_SERVER`` / ``JS_DEBUG_DAP_SERVER``
    2. ``$XDG_DATA_HOME/nvim/mason/packages/js-debug-adapter/js-debug/src/dapDebugServer.js``
    3. ``~/.local/opt/js-debug/src/dapDebugServer.js``
    4. nested ``js-debug/src/dapDebugServer.js`` under ``~/.local/opt/js-debug``
    """
    home = Path.home()
    data_home = Path(os.environ.get("XDG_DATA_HOME") or (home / ".local" / "share"))
    configured = (
        os.environ.get("HAIRBALL_JS_DEBUG_DAP_SERVER")
        or os.environ.get("JS_DEBUG_DAP_SERVER")
        or ""
    ).strip()
    candidates: list[Path] = []
    if configured:
        raw = Path(configured).expanduser()
        if not raw.is_absolute() and cwd:
            raw = (Path(cwd) / raw).resolve()
        else:
            raw = raw.resolve()
        candidates.append(raw)
    candidates.extend(
        [
            data_home
            / "nvim"
            / "mason"
            / "packages"
            / "js-debug-adapter"
            / "js-debug"
            / "src"
            / "dapDebugServer.js",
            home / ".local" / "opt" / "js-debug" / "src" / "dapDebugServer.js",
            home / ".local" / "opt" / "js-debug" / "js-debug" / "src" / "dapDebugServer.js",
        ]
    )
    for path in candidates:
        try:
            if path.is_file():
                return path
        except OSError:
            continue
    return None


def find_js_debug(cwd: Optional[str] = None) -> bool:
    return find_node() is not None and resolve_js_debug_server(cwd) is not None


def resolve_lldb_dap() -> Optional[str]:
    """Path to ``lldb-dap`` binary if available."""
    configured = (os.environ.get("HAIRBALL_LLDB_DAP") or "").strip()
    candidates: list[str] = []
    if configured:
        candidates.append(str(Path(configured).expanduser()))
    which = shutil.which("lldb-dap")
    if which:
        candidates.append(which)
    # Homebrew llvm (common on macOS when Xcode CLT has no lldb-dap)
    for prefix in (
        "/opt/homebrew/opt/llvm/bin/lldb-dap",
        "/usr/local/opt/llvm/bin/lldb-dap",
    ):
        candidates.append(prefix)
    brew = shutil.which("brew")
    if brew:
        try:
            import subprocess

            out = subprocess.check_output(
                [brew, "--prefix", "llvm"], text=True, stderr=subprocess.DEVNULL, timeout=3
            ).strip()
            if out:
                candidates.append(str(Path(out) / "bin" / "lldb-dap"))
        except Exception:
            pass
    for path in candidates:
        try:
            if path and Path(path).is_file() and os.access(path, os.X_OK):
                return path
        except OSError:
            continue
    return None


def find_lldb() -> bool:
    return resolve_lldb_dap() is not None


def resolve_gdb() -> Optional[str]:
    configured = (os.environ.get("HAIRBALL_GDB") or "").strip()
    if configured:
        p = Path(configured).expanduser()
        if p.is_file() and os.access(p, os.X_OK):
            return str(p)
    return shutil.which("gdb")


def find_gdb() -> bool:
    return resolve_gdb() is not None


def normalize_adapter_name(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    name = str(raw).strip().lower()
    if not name:
        return None
    if name in {"js-debug", "js-debug-adapter", "javascript", "typescript", "node", "pwa-node"}:
        return ADAPTER_JS_DEBUG
    if name in {"debugpy", "python"}:
        return ADAPTER_DEBUGPY
    if name in {"lldb", "lldb-dap", "codelldb"}:
        return ADAPTER_LLDB
    if name in {"gdb"}:
        return ADAPTER_GDB
    return name


def infer_adapter(
    program: Optional[str] = None,
    *,
    adapter: Optional[str] = None,
) -> str:
    """Pick adapter: explicit ``adapter`` wins, else file extension heuristics."""
    explicit = normalize_adapter_name(adapter)
    if explicit in {ADAPTER_JS_DEBUG, ADAPTER_DEBUGPY, ADAPTER_LLDB, ADAPTER_GDB}:
        return explicit
    if program:
        try:
            path = Path(program)
            ext = path.suffix.lower()
        except Exception:
            ext = ""
            path = None
        if ext in JS_DEBUG_EXTS:
            return ADAPTER_JS_DEBUG
        if ext == ".py":
            return ADAPTER_DEBUGPY
        if ext in NATIVE_EXTS or (path is not None and ext == "" and path.is_file()):
            # Prefer lldb on macOS toolchains; fall back to gdb; else lldb name
            # so the error message points at installing lldb-dap.
            if find_lldb():
                return ADAPTER_LLDB
            if find_gdb():
                return ADAPTER_GDB
            return ADAPTER_LLDB
    return ADAPTER_DEBUGPY


def any_adapter_available(cwd: Optional[str] = None) -> bool:
    return find_js_debug(cwd) or find_lldb() or find_gdb() or _debugpy_importable()


def _debugpy_importable() -> bool:
    try:
        import debugpy  # noqa: F401

        return True
    except Exception:
        return False
