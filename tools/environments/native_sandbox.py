"""Host-native isolation for local command execution.

The local backend keeps its existing shell/session semantics.  This module only
transforms the final argv at the process boundary, so terminal, file tools,
background jobs, and code execution share one filesystem policy.
"""

from __future__ import annotations

import os
import platform
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence


_MACOS_SANDBOX_EXECUTABLE = "/usr/bin/sandbox-exec"


_MACOS_BASE_POLICY = r"""
(version 1)
(deny default)

(allow process-exec)
(allow process-fork)
(allow signal (target same-sandbox))
(allow process-info* (target same-sandbox))

(allow file-write-data
  (require-all
    (path "/dev/null")
    (vnode-type CHARACTER-DEVICE)))

(allow sysctl-read
  (sysctl-name "hw.activecpu")
  (sysctl-name "hw.busfrequency_compat")
  (sysctl-name "hw.byteorder")
  (sysctl-name "hw.cacheconfig")
  (sysctl-name "hw.cachelinesize_compat")
  (sysctl-name "hw.cpufamily")
  (sysctl-name "hw.cpufrequency_compat")
  (sysctl-name "hw.cputype")
  (sysctl-name "hw.l1dcachesize_compat")
  (sysctl-name "hw.l1icachesize_compat")
  (sysctl-name "hw.l2cachesize_compat")
  (sysctl-name "hw.l3cachesize_compat")
  (sysctl-name "hw.logicalcpu_max")
  (sysctl-name "hw.machine")
  (sysctl-name "hw.model")
  (sysctl-name "hw.memsize")
  (sysctl-name "hw.ncpu")
  (sysctl-name "hw.nperflevels")
  (sysctl-name-prefix "hw.optional.arm.")
  (sysctl-name-prefix "hw.optional.armv8_")
  (sysctl-name "hw.packages")
  (sysctl-name "hw.pagesize_compat")
  (sysctl-name "hw.pagesize")
  (sysctl-name "hw.physicalcpu")
  (sysctl-name "hw.physicalcpu_max")
  (sysctl-name "hw.logicalcpu")
  (sysctl-name "hw.cpufrequency")
  (sysctl-name "hw.tbfrequency_compat")
  (sysctl-name "hw.vectorunit")
  (sysctl-name "machdep.cpu.brand_string")
  (sysctl-name "kern.argmax")
  (sysctl-name "kern.hostname")
  (sysctl-name "kern.maxfilesperproc")
  (sysctl-name "kern.maxproc")
  (sysctl-name "kern.osproductversion")
  (sysctl-name "kern.osrelease")
  (sysctl-name "kern.ostype")
  (sysctl-name "kern.osvariant_status")
  (sysctl-name "kern.osversion")
  (sysctl-name "kern.secure_kernel")
  (sysctl-name "kern.usrstack64")
  (sysctl-name "kern.version")
  (sysctl-name "sysctl.proc_cputype")
  (sysctl-name "vm.loadavg")
  (sysctl-name-prefix "hw.perflevel")
  (sysctl-name-prefix "kern.proc.pgrp.")
  (sysctl-name-prefix "kern.proc.pid.")
  (sysctl-name-prefix "net.routetable."))

(allow sysctl-write (sysctl-name "kern.grade_cputype"))
(allow iokit-open (iokit-registry-entry-class "RootDomainUserClient"))
(allow mach-lookup
  (global-name "com.apple.system.opendirectoryd.libinfo")
  (global-name "com.apple.PowerManagement.control"))
(allow ipc-posix-sem)
(allow ipc-posix-shm-read-data
  ipc-posix-shm-write-create
  ipc-posix-shm-write-unlink
  (ipc-posix-name-regex #"^/__KMP_REGISTERED_LIB_[0-9]+$"))

(allow pseudo-tty)
(allow file-read* file-write* file-ioctl (literal "/dev/ptmx"))
(allow file-read* file-write*
  (require-all
    (regex #"^/dev/ttys[0-9]+")
    (extension "com.apple.sandbox.pty")))
(allow file-ioctl (regex #"^/dev/ttys[0-9]+"))

(allow ipc-posix-shm-read* (ipc-posix-name-prefix "apple.cfprefs."))
(allow mach-lookup
  (global-name "com.apple.cfprefsd.daemon")
  (global-name "com.apple.cfprefsd.agent")
  (local-name "com.apple.cfprefsd.agent"))
(allow user-preference-read)

(allow file-read*)
""".strip()


_MACOS_NETWORK_POLICY = r"""
(allow network*)
(allow system-socket
  (require-all
    (socket-domain AF_SYSTEM)
    (socket-protocol 2)))
(allow mach-lookup
  (global-name "com.apple.bsd.dirhelper")
  (global-name "com.apple.system.opendirectoryd.membership")
  (global-name "com.apple.SecurityServer")
  (global-name "com.apple.networkd")
  (global-name "com.apple.ocspd")
  (global-name "com.apple.trustd.agent")
  (global-name "com.apple.SystemConfiguration.DNSConfiguration")
  (global-name "com.apple.SystemConfiguration.configd"))
(allow sysctl-read (sysctl-name-regex #"^net.routetable"))
""".strip()


def native_sandbox_available() -> bool:
    """Return whether this host has the supported native isolation boundary."""

    return (
        platform.system() == "Darwin"
        and os.path.isfile(_MACOS_SANDBOX_EXECUTABLE)
        and os.access(_MACOS_SANDBOX_EXECUTABLE, os.X_OK)
    )


def _absolute_root(value: object) -> str | None:
    raw = str(value or "").strip()
    if not raw:
        return None
    expanded = os.path.expanduser(os.path.expandvars(raw))
    if not os.path.isabs(expanded):
        return None
    return str(Path(expanded).resolve(strict=False))


def writable_roots_for_local_process(
    cwd: str,
    *,
    config: Mapping[str, object] | None = None,
    extra_roots: Iterable[str] = (),
    include_workspace: bool = True,
) -> tuple[str, ...]:
    """Resolve and de-duplicate the writable roots for one local process."""

    raw_config = config or {}
    # /tmp and tempfile.gettempdir() are both included because macOS commonly
    # exposes them through different resolved paths.  Plan's read-only mode
    # intentionally omits cwd and configured roots: only process scratch files
    # remain writable.
    candidates: list[object] = [tempfile.gettempdir(), "/tmp"]
    if include_workspace:
        candidates.insert(0, cwd)
        configured = raw_config.get("writable_roots", ())
        if isinstance(configured, str):
            candidates.append(configured)
        elif isinstance(configured, Sequence):
            candidates.extend(configured)
    candidates.extend(extra_roots)

    roots: list[str] = []
    seen: set[str] = set()
    for candidate in candidates:
        root = _absolute_root(candidate)
        if root is None or root in seen:
            continue
        seen.add(root)
        roots.append(root)
    return tuple(roots)


def build_macos_workspace_policy(
    writable_roots: Sequence[str],
    *,
    network: bool,
) -> tuple[str, tuple[tuple[str, str], ...]]:
    """Build a closed-by-default macOS policy and its path parameters."""

    params: list[tuple[str, str]] = []
    write_rules: list[str] = []
    for index, root in enumerate(writable_roots):
        key = f"WRITABLE_ROOT_{index}"
        params.append((key, root))
        write_rules.append(f'(allow file-write* (subpath (param "{key}")))')
    sections = [_MACOS_BASE_POLICY, *write_rules]
    if network:
        sections.append(_MACOS_NETWORK_POLICY)
    return "\n".join(sections), tuple(params)


def wrap_local_process_args(
    args: Sequence[str],
    *,
    cwd: str,
    config: Mapping[str, object] | None = None,
    extra_writable_roots: Iterable[str] = (),
    read_only_workspace: bool = False,
) -> list[str]:
    """Transform argv at the host spawn boundary.

    Unsupported platforms keep their established argv unchanged.  On macOS the
    transform is mandatory for the local backend; there is no legacy/new runtime
    selector inside the product.
    """

    argv = [str(item) for item in args]
    if not native_sandbox_available():
        return argv
    raw_config = config or {}
    roots = writable_roots_for_local_process(
        cwd,
        config=raw_config,
        extra_roots=extra_writable_roots,
        include_workspace=not read_only_workspace,
    )
    policy, params = build_macos_workspace_policy(
        roots,
        network=bool(raw_config.get("network", True)),
    )
    wrapped = [_MACOS_SANDBOX_EXECUTABLE, "-p", policy]
    wrapped.extend(f"-D{key}={value}" for key, value in params)
    wrapped.append("--")
    wrapped.extend(argv)
    return wrapped


def is_likely_native_sandbox_denial(returncode: int, output: str) -> bool:
    """Conservatively identify a failed command as a native-policy denial."""

    if returncode == 0:
        return False
    lower = str(output or "").casefold()
    return any(
        marker in lower
        for marker in (
            "operation not permitted",
            "permission denied",
            "read-only file system",
            "sandbox-exec",
            "failed to write file",
        )
    )


__all__ = [
    "build_macos_workspace_policy",
    "is_likely_native_sandbox_denial",
    "native_sandbox_available",
    "wrap_local_process_args",
    "writable_roots_for_local_process",
]
