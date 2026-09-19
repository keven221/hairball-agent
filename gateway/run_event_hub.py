"""Bounded multicast and replay for Hairball App Server run events.

The HTTP adapter owns agent execution and approval state.  This module owns
only the transport-facing event history: one canonical sequence per run,
independent subscriber queues, reconnect replay, and bounded retention.

All mutating methods are called on the owning asyncio event-loop thread.  Worker
threads must schedule ``publish`` with ``loop.call_soon_threadsafe`` rather than
calling it directly.
"""

from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Optional


class RunEventCursorExpired(ValueError):
    """The requested cursor predates the bounded replay journal."""

    def __init__(self, *, requested: int, first_available: int, last_available: int):
        super().__init__(
            f"Run event cursor {requested} expired; first available event is "
            f"{first_available}"
        )
        self.requested = requested
        self.first_available = first_available
        self.last_available = last_available


@dataclass(frozen=True)
class RunEventLagged:
    """Signal that a subscriber must reconnect from its last received cursor."""

    last_available: int


@dataclass
class _RunChannel:
    created_at: float
    max_events: int
    events: Deque[Dict[str, Any]] = field(init=False)
    next_sequence: int = 1
    subscribers: Dict[int, "asyncio.Queue[object]"] = field(default_factory=dict)
    next_subscriber_id: int = 1
    closed: bool = False

    def __post_init__(self) -> None:
        self.events = deque(maxlen=self.max_events)


class RunEventSubscription:
    """One independent replay/live cursor over a run's canonical event stream."""

    def __init__(
        self,
        *,
        hub: "RunEventHub",
        run_id: str,
        subscriber_id: Optional[int],
        replay: tuple[Dict[str, Any], ...],
        queue: "asyncio.Queue[object]",
        closed: bool,
    ) -> None:
        self._hub = hub
        self.run_id = run_id
        self._subscriber_id = subscriber_id
        self._replay = deque(replay)
        self._queue = queue
        self._closed = closed

    async def get(self, *, timeout: Optional[float] = None) -> object:
        """Return the next event, ``RunEventLagged``, or ``None`` at EOF."""
        if self._replay:
            return self._replay.popleft()
        if self._closed:
            return None
        if timeout is None:
            item = await self._queue.get()
        else:
            item = await asyncio.wait_for(self._queue.get(), timeout=timeout)
        if item is None:
            self._closed = True
        return item

    def close(self) -> None:
        if self._subscriber_id is not None:
            self._hub.unsubscribe(self.run_id, self._subscriber_id)
            self._subscriber_id = None
        self._closed = True


class RunEventHub:
    """Canonical per-run event journal with bounded multicast subscribers."""

    def __init__(self, *, max_events: int = 2048, subscriber_queue_size: int = 512):
        if max_events < 1:
            raise ValueError("max_events must be positive")
        if subscriber_queue_size < 2:
            raise ValueError("subscriber_queue_size must be at least 2")
        self._max_events = int(max_events)
        self._subscriber_queue_size = int(subscriber_queue_size)
        self._runs: Dict[str, _RunChannel] = {}

    def open(self, run_id: str, *, created_at: float) -> None:
        if not run_id:
            raise ValueError("run_id is required")
        if run_id in self._runs:
            raise ValueError(f"run already exists: {run_id}")
        self._runs[run_id] = _RunChannel(
            created_at=float(created_at),
            max_events=self._max_events,
        )

    def has_run(self, run_id: str) -> bool:
        return run_id in self._runs

    def run_count(self) -> int:
        return len(self._runs)

    def is_closed(self, run_id: str) -> bool:
        channel = self._runs.get(run_id)
        return bool(channel and channel.closed)

    def publish(self, run_id: str, event: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Publish once to the journal and every live subscriber."""
        channel = self._runs.get(run_id)
        if channel is None or channel.closed:
            return None

        canonical = dict(event)
        sequence = channel.next_sequence
        channel.next_sequence += 1
        canonical["run_id"] = run_id
        canonical["seq"] = sequence
        channel.events.append(canonical)

        lagged: list[int] = []
        for subscriber_id, queue in tuple(channel.subscribers.items()):
            if queue.full():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                queue.put_nowait(RunEventLagged(last_available=sequence))
                queue.put_nowait(None)
                lagged.append(subscriber_id)
            else:
                queue.put_nowait(canonical)
        for subscriber_id in lagged:
            channel.subscribers.pop(subscriber_id, None)
        return canonical

    def close(self, run_id: str) -> None:
        channel = self._runs.get(run_id)
        if channel is None or channel.closed:
            return
        channel.closed = True
        for queue in tuple(channel.subscribers.values()):
            if queue.full():
                while not queue.empty():
                    try:
                        queue.get_nowait()
                    except asyncio.QueueEmpty:
                        break
                queue.put_nowait(
                    RunEventLagged(last_available=channel.next_sequence - 1)
                )
            queue.put_nowait(None)
        channel.subscribers.clear()

    def subscribe(self, run_id: str, *, after: int = 0) -> Optional[RunEventSubscription]:
        channel = self._runs.get(run_id)
        if channel is None:
            return None
        if after < 0:
            raise ValueError("after must be non-negative")

        first_available = (
            int(channel.events[0]["seq"])
            if channel.events
            else channel.next_sequence
        )
        last_available = channel.next_sequence - 1
        if channel.events and after < first_available - 1:
            raise RunEventCursorExpired(
                requested=after,
                first_available=first_available,
                last_available=last_available,
            )

        replay = tuple(event for event in channel.events if int(event["seq"]) > after)
        queue: "asyncio.Queue[object]" = asyncio.Queue(
            maxsize=self._subscriber_queue_size
        )
        subscriber_id: Optional[int] = None
        if not channel.closed:
            subscriber_id = channel.next_subscriber_id
            channel.next_subscriber_id += 1
            channel.subscribers[subscriber_id] = queue
        return RunEventSubscription(
            hub=self,
            run_id=run_id,
            subscriber_id=subscriber_id,
            replay=replay,
            queue=queue,
            closed=channel.closed,
        )

    def unsubscribe(self, run_id: str, subscriber_id: int) -> None:
        channel = self._runs.get(run_id)
        if channel is not None:
            channel.subscribers.pop(subscriber_id, None)

    def subscriber_count(self, run_id: str) -> int:
        channel = self._runs.get(run_id)
        return len(channel.subscribers) if channel is not None else 0

    def created_at(self, run_id: str) -> Optional[float]:
        channel = self._runs.get(run_id)
        return channel.created_at if channel is not None else None

    def expire(self, *, now: float, ttl: float) -> list[str]:
        """Drop transport journals older than ``ttl`` with no live subscriber."""
        expired = [
            run_id
            for run_id, channel in self._runs.items()
            if now - channel.created_at > ttl and not channel.subscribers
        ]
        for run_id in expired:
            self._runs.pop(run_id, None)
        return expired

    def discard(self, run_id: str) -> None:
        channel = self._runs.pop(run_id, None)
        if channel is None:
            return
        for queue in tuple(channel.subscribers.values()):
            if queue.full():
                try:
                    queue.get_nowait()
                except asyncio.QueueEmpty:
                    pass
            queue.put_nowait(None)

    def clear(self) -> None:
        for run_id in tuple(self._runs):
            self.discard(run_id)
