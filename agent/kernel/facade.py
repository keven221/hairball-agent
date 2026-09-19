"""The kernel facade: one entry point, two backends, session-pinned routing.

Product surfaces call the facade instead of reaching into ``AIAgent``'s
transient fields to guess execution state. Backend selection is the
``kernel.default_version`` switch (pinned per session):

- **v1** — existing conversation loop, observed through
  :class:`~agent.kernel.v1_adapter.V1EventTap`.
- **v2** — :class:`~agent.kernel.engine.KernelEngine` (OpenCode-shaped turn
  loop). Prompt/tools stay borrowed from the agent so A/B varies only the loop.

Two rules the facade enforces on behalf of the whole design:

- **A session never switches kernels mid-life.** The version is pinned in
  ``state.db`` on first use and honoured afterwards, so a rollout toggle can
  only affect *new* sessions. Rolling back never requires touching user data.
- **The ledger is optional, not load-bearing.** With ``kernel.enabled: false``
  (the default) ``run_turn`` is a thin passthrough to ``run_conversation`` with
  no tap installed and no database work, so an experiment that goes wrong cannot
  degrade anyone who did not opt in.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Iterable, Optional

from agent.kernel.contracts import (
    KERNEL_V1,
    KERNEL_V2,
    EventKind,
    KernelEvent,
    TurnRequest,
    new_turn_id,
)
from agent.kernel.event_ledger import EventLedger
from agent.kernel.projectors import (
    AgentTreeProjector,
    ChatProjector,
    MetricsProjector,
    Projector,
    SidebarProjector,
    WorkspaceProjector,
    check_invariants,
    replay,
)
from agent.kernel.v1_adapter import V1EventTap

logger = logging.getLogger(__name__)

_PROJECTOR_TYPES: dict[str, type[Projector]] = {
    "chat": ChatProjector,
    "workspace": WorkspaceProjector,
    "agent_tree": AgentTreeProjector,
    "sidebar": SidebarProjector,
    "metrics": MetricsProjector,
}


class KernelSettings:
    """Resolved ``kernel:`` section of ``config.yaml``.

    Every knob here is behavioral configuration, so it lives in ``config.yaml``
    rather than an environment variable — ``.env`` is for secrets only.
    """

    def __init__(self, raw: Optional[dict[str, Any]] = None) -> None:
        data = dict(raw or {})
        self.enabled = bool(data.get("enabled", False))
        self.default_version = str(data.get("default_version", KERNEL_V1) or KERNEL_V1)
        self.observe_v1 = bool(data.get("observe_v1", True))
        self.persist_stream_deltas = bool(data.get("persist_stream_deltas", False))
        self.enforce_invariants = bool(data.get("enforce_invariants", True))
        self.projectors: tuple[str, ...] = tuple(
            data.get("projectors") or ("chat", "workspace", "sidebar", "agent_tree")
        )

    @classmethod
    def from_config(cls, config: Optional[dict[str, Any]] = None) -> "KernelSettings":
        """Read settings from *config*, falling back to the on-disk config.

        Import of the config loader is deferred so that ``agent.kernel`` stays
        importable in a bare unit test without pulling in the CLI package.
        """
        if config is not None:
            return cls(config.get("kernel") if isinstance(config, dict) else None)
        try:
            from hairball_cli.config import load_config

            return cls((load_config() or {}).get("kernel"))
        except Exception:
            logger.debug("kernel settings fell back to defaults", exc_info=True)
            return cls(None)


class TurnHandle:
    """What a caller gets back from :meth:`Kernel.run_turn`."""

    def __init__(
        self,
        request: TurnRequest,
        result: Any,
        *,
        kernel_version: str,
        first_seq: int,
        last_seq: int,
    ) -> None:
        self.request = request
        self.result = result
        self.kernel_version = kernel_version
        self.first_seq = first_seq
        self.last_seq = last_seq

    @property
    def turn_id(self) -> str:
        return self.request.turn_id

    @property
    def final_response(self) -> str:
        if isinstance(self.result, dict):
            return str(self.result.get("final_response") or "")
        return str(self.result or "")


class Kernel:
    """The narrow waist: start turns, read events, query projections."""

    def __init__(
        self,
        *,
        db: Any = None,
        settings: Optional[KernelSettings] = None,
        ledger: Optional[EventLedger] = None,
    ) -> None:
        self.settings = settings or KernelSettings.from_config()
        self._db = db
        self._ledger = ledger
        self._taps: dict[int, V1EventTap] = {}

    # ── ledger access ──

    @property
    def ledger(self) -> Optional[EventLedger]:
        """The durable ledger, lazily opened. ``None`` when the kernel is off."""
        if not self.settings.enabled:
            return None
        if self._ledger is None:
            db = self._db
            if db is None:
                try:
                    from hairball_state import SessionDB

                    db = SessionDB()
                except Exception:
                    logger.warning("kernel ledger unavailable: SessionDB failed to open",
                                   exc_info=True)
                    return None
                self._db = db
            self._ledger = EventLedger(
                db,
                persist_ephemeral=self.settings.persist_stream_deltas,
                enforce_invariants=self.settings.enforce_invariants,
            )
        return self._ledger

    # ── version pinning ──

    def resolve_version(self, session_id: str) -> str:
        """Return the kernel version this session must run under.

        A session already pinned keeps its pin. An unpinned session gets — and
        is pinned to — the configured default.
        """
        ledger = self.ledger
        if ledger is None:
            return KERNEL_V1
        try:
            existing = ledger.session_version(session_id)
            if existing:
                return str(existing.get("kernel_version") or KERNEL_V1)
            row = ledger.open_session(
                session_id, kernel_version=self.settings.default_version
            )
            return str(row.get("kernel_version") or KERNEL_V1)
        except Exception:
            logger.warning("kernel version resolution failed for %s", session_id,
                           exc_info=True)
            return KERNEL_V1

    # ── running a turn ──

    def run_turn(
        self,
        agent: Any,
        user_message: Any,
        *,
        request: Optional[TurnRequest] = None,
        run: Optional[Callable[[], Any]] = None,
        **run_kwargs: Any,
    ) -> TurnHandle:
        """Run one turn through the pinned backend, recording kernel events.

        *run* lets a caller supply its own invocation (for example
        ``agent.chat``). When omitted:

        - **v1** → ``agent.run_conversation`` for explicit legacy facade callers
        - **v2** → ``KernelEngine.run_turn`` (real second loop, not a fallback)
        """
        session_id = str(getattr(agent, "session_id", "") or "")
        ledger = self.ledger
        # Product surfaces no longer come through this compatibility facade:
        # native events originate in ``conversation_loop``.  Keep the explicit
        # facade API for integrations that intentionally invoke it.
        invoke = run or (lambda: agent.run_conversation(user_message, **run_kwargs))

        if ledger is None or not session_id or not self.settings.observe_v1:
            result = invoke()
            req = request or TurnRequest(
                session_id=session_id, turn_id=new_turn_id(), user_input=user_message
            )
            return TurnHandle(req, result, kernel_version=KERNEL_V1,
                              first_seq=0, last_seq=0)

        version = self.resolve_version(session_id)
        before = ledger.max_seq(session_id)

        if version == KERNEL_V2 and run is None:
            from agent.kernel.engine import KernelEngine

            engine = KernelEngine(agent, ledger=ledger, session_id=session_id)
            req = request or TurnRequest(
                session_id=session_id,
                turn_id=new_turn_id(),
                user_input=user_message,
                provider_ref=str(getattr(agent, "provider", "") or ""),
                model_ref=str(getattr(agent, "model", "") or ""),
            )
            result = engine.run_turn(
                user_message,
                request=req,
                conversation_history=run_kwargs.get("conversation_history"),
                system_message=run_kwargs.get("system_message"),
                max_iterations=int(run_kwargs.get("max_iterations") or 0),
            )
            return TurnHandle(
                req,
                result,
                kernel_version=KERNEL_V2,
                first_seq=before + 1,
                last_seq=ledger.max_seq(session_id),
            )

        tap = V1EventTap(agent, ledger, session_id=session_id)
        tap.install()
        try:
            with tap.turn(request, user_input=user_message) as req:
                result = invoke()
                tap.finish_turn_with_result(result)
        finally:
            tap.uninstall()

        return TurnHandle(
            req,
            result,
            kernel_version=version,
            first_seq=before + 1,
            last_seq=ledger.max_seq(session_id),
        )

    # ── queries ──

    def events(
        self,
        session_id: str,
        *,
        after_seq: int = 0,
        limit: Optional[int] = None,
        turn_id: Optional[str] = None,
        kinds: Optional[Iterable[EventKind | str]] = None,
    ) -> list[KernelEvent]:
        """Read durable events after *after_seq* — the reconnect path."""
        ledger = self.ledger
        if ledger is None:
            return []
        return ledger.read(
            session_id, after_seq=after_seq, limit=limit, turn_id=turn_id, kinds=kinds
        )

    def project(
        self,
        session_id: str,
        *,
        names: Optional[Iterable[str]] = None,
        after_seq: int = 0,
    ) -> dict[str, dict[str, Any]]:
        """Replay the ledger into the requested projections.

        Because each projector is an independent function of the log, asking for
        ``["chat"]`` cannot mutate or invalidate what ``["workspace"]`` would
        return.

        Returns ``{}`` when no ledger is available, so a caller can distinguish
        "the kernel is not recording" from "the kernel recorded an empty
        session" — handing back empty snapshots would let a surface render
        "no messages" for a session that has plenty.
        """
        if self.ledger is None:
            return {}
        selected = tuple(names) if names else self.settings.projectors
        projectors = [
            _PROJECTOR_TYPES[name]() for name in selected if name in _PROJECTOR_TYPES
        ]
        if not projectors:
            return {}
        return replay(self.events(session_id, after_seq=after_seq), projectors)

    def verify(self, session_id: str) -> Any:
        """Check the session's ledger against the durable invariants."""
        return check_invariants(self.events(session_id))

    def subscribe(self, callback: Callable[[KernelEvent], None]) -> Callable[[], None]:
        """Register a live post-commit listener. Returns an unsubscribe callable."""
        ledger = self.ledger
        if ledger is None:
            return lambda: None
        return ledger.subscribe(callback)


_DEFAULT_KERNEL: Optional[Kernel] = None


def get_kernel(*, refresh: bool = False) -> Kernel:
    """Process-wide kernel instance.

    Cached because opening ``SessionDB`` is not free and the ledger keeps
    per-session invariant state that should be shared by every caller in the
    process. Pass ``refresh=True`` after changing configuration.
    """
    global _DEFAULT_KERNEL
    if _DEFAULT_KERNEL is None or refresh:
        _DEFAULT_KERNEL = Kernel()
    return _DEFAULT_KERNEL
