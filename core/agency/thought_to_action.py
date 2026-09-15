"""From a thought she had on her own to an action, through the Will.

The neural intent router was written to close the gap between an internal
thought that carries an action and the action itself. It gates on trusted
internal sources, matches a small whitelist of action schemas, asks the Will
and fails closed without one, and records every outcome in the life trace.
Nothing ever called it. Volition and the drive engine announce their goals on
the "thoughts" topic, and the only listener was the interface.

This is the listener. It hands the router every thought whose `source` names
one of the router's trusted internal sources, and ignores the rest: a thought
that does not say where it came from is not trusted to act. The router decides
the rest, so a goal that matches no action schema is recorded as a thought and
nothing more.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from collections.abc import Mapping
from typing import Any

from core.runtime.errors import record_degradation
from core.utils.task_tracker import get_task_tracker

logger = logging.getLogger(__name__)

__all__ = ["ThoughtRouter", "activate_thought_router", "get_thought_router"]

TOPIC = "thoughts"


class ThoughtRouter:
    """Listens on the thoughts topic and routes the trusted ones."""

    def __init__(self, router: Any = None) -> None:
        self._router = router
        self._task: asyncio.Task | None = None
        #: Advanced by every start and stop. A loop runs only while the value it
        #: started with is current, so a stopped or replaced listener leaves on
        #: its next wake even if its cancellation never lands.
        self._generation = 0
        self.routed = 0
        self.ignored = 0
        self.failed = 0

    def _resolve_router(self) -> Any:
        if self._router is None:
            from core.agency.neural_intent_router import get_neural_intent_router

            self._router = get_neural_intent_router()
        return self._router

    async def handle(self, message: Any) -> bool:
        """Route one thought. Returns whether it reached the router."""
        from core.agency.neural_intent_router import TRUSTED_INTERNAL_SOURCES

        if not isinstance(message, Mapping):
            self.ignored += 1
            return False
        source = str(message.get("source") or "").strip().lower()
        content = str(message.get("content") or "").strip()
        if source not in TRUSTED_INTERNAL_SOURCES or not content:
            self.ignored += 1
            return False
        try:
            await self._resolve_router().route(source, content)
        except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
            self.failed += 1
            record_degradation(
                "thought_to_action",
                exc,
                action="left a trusted thought unrouted; the router raised",
                severity="warning",
            )
            return False
        self.routed += 1
        return True

    async def _run(self, generation: int | None = None) -> None:
        from core.event_bus import get_event_bus
        from core.runtime.reconcile import IDLE_POLL_S

        mine = self._generation if generation is None else generation
        bus = get_event_bus()
        queue = await bus.subscribe(TOPIC)
        try:
            while mine == self._generation:
                # Bounded, the way every consumer loop on the bus is: a quiet
                # topic otherwise holds this listener past the router it belongs
                # to, and a queue get is cancel-safe, so an idle timeout drops no
                # thought.
                try:
                    item = await asyncio.wait_for(queue.get(), timeout=IDLE_POLL_S)
                except TimeoutError:
                    continue
                # The bus queues (priority, sequence, {"topic": ..., "data": message}),
                # so the thought is two layers in.
                event = item[-1] if isinstance(item, tuple) and item else item
                message = event.get("data") if isinstance(event, Mapping) and "data" in event else event
                await self.handle(message)
        finally:
            # It subscribed, so it leaves. The queue was never released before,
            # and every restart added another subscriber nobody drained.
            with contextlib.suppress(Exception):
                await bus.unsubscribe(TOPIC, queue)

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return
        self._generation += 1
        self._task = get_task_tracker().create_task(self._run(self._generation), name="agency.thought_to_action")

    async def stop(self) -> None:
        self._generation += 1
        task, self._task = self._task, None
        if task is None:
            return
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            logger.debug("thought router stopped")

    def as_dict(self) -> dict[str, Any]:
        return {"routed": self.routed, "ignored": self.ignored, "failed": self.failed}


_instance: ThoughtRouter | None = None


def get_thought_router() -> ThoughtRouter:
    global _instance
    if _instance is None:
        _instance = ThoughtRouter()
    return _instance


async def activate_thought_router(*, foreground_only: bool = False) -> Any:
    """Start the listener at boot, as a foundations activator.

    A foreground-only runtime runs no autonomous life, so nothing it thinks on
    its own is waiting to act and the listener is not started.
    """
    from core.runtime.foundations import ActivationResult

    if foreground_only:
        return ActivationResult(
            name="thought_to_action", ok=True, detail="not started: foreground-only runtime"
        )
    await get_thought_router().start()
    return ActivationResult(
        name="thought_to_action", ok=True, detail=f"listening on the {TOPIC} topic"
    )
