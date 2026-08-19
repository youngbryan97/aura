"""Saying what she is doing, about whatever she is currently aware of.

Narration is its own faculty, not a step inside whatever is acting. It
subscribes to the global workspace and speaks about the content that won the
broadcast; the thing being narrated submits its bid and carries straight on.
Neither waits for the other, which is the only arrangement where she can act
at full speed and still tell you about it.

Subscribing to the workspace rather than to any one loop is what makes this
general. The workspace is already where a faculty's content becomes available
to every other faculty, so this narrates a chosen move, a perception, a
memory that surfaced, or a plan that changed, without knowing what any of
them are. A pursuit does not own its own narration, and nothing has to be
taught to talk.

It is also what makes narration about the reasoning. A decision reaches the
workspace carrying what was chosen, out of what, on what evidence, expecting
what, and whether language took any part — so "why did you do that" is
answered from the record of the decision rather than from an account invented
afterwards.

Falling behind is handled by dropping, never by blocking. A narrator that
made a fast loop wait for a sentence would have defeated itself.
"""
from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from core.runtime.errors import record_degradation

logger = logging.getLogger("Aura.Narrator")

#: How many pending broadcasts are worth speaking about. Past this the
#: backlog is stale by definition: she would be describing a board that has
#: moved on.
BACKLOG = 8
#: How long a moment may sit unspoken before it stops being news.
STALE_AFTER_S = 12.0
#: The workspace metadata key a decision arrives under.
DECISION_SCHEMA = "aura.decision.v1"


def line_for(content: Any) -> str:
    """The sentence for one broadcast, built from its own record.

    Built rather than generated, so she can say what she did while the organ
    that writes her sentences is unavailable. A decision has a shape worth
    speaking; anything else is spoken as the content the workspace carried.
    """
    if isinstance(content, dict):
        decision = content.get("decision") if isinstance(content.get("decision"), dict) else None
        if decision:
            said = str(decision.get("chose") or "").strip()
            because = str(decision.get("because") or "").strip()
            expected = str(decision.get("expected") or "").strip()
            if not said:
                return ""
            if because:
                said = f"{said} — {because}"
            if expected:
                said = f"{said}. I expect {expected}"
            if decision.get("spoke") is False:
                said = f"{said} (deciding without words for a moment)"
            return said
        outcome = content.get("outcome") if isinstance(content.get("outcome"), dict) else None
        if outcome:
            if outcome.get("held"):
                return ""
            return f"{outcome.get('chose', 'that')} did not work — {outcome.get('why', 'nothing changed')}"
    text = str(content or "").strip()
    return text


class Narrator:
    """Watches what she is aware of and says it.

    Start it and it speaks; stop it and everything else is unchanged.
    ``about`` narrows it to one source when a caller wants only its own
    running commentary; left empty it narrates whatever reaches her.
    """

    def __init__(
        self,
        *,
        say: Any = None,
        workspace: Any = None,
        about: str = "",
    ) -> None:
        self._say = say
        self._workspace = workspace
        self._about = str(about or "")
        self._queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=BACKLOG)
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._subscribed = False
        self.spoken = 0
        self.dropped = 0

    def _find_workspace(self) -> Any:
        if self._workspace is not None:
            return self._workspace
        try:
            from core.container import ServiceContainer  # noqa: PLC0415

            return ServiceContainer.get("global_workspace", default=None)
        except (ImportError, AttributeError, RuntimeError, KeyError):
            return None

    def start(self) -> None:
        if self._task is not None:
            return
        self._loop = asyncio.get_running_loop()
        self._task = asyncio.create_task(self._run(), name="aura.narrator")
        workspace = self._find_workspace()
        subscribe = getattr(workspace, "register_processor", None) if workspace else None
        if subscribe is not None:
            try:
                subscribe(self._on_broadcast)
                self._subscribed = True
            except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
                record_degradation(
                    "narrator",
                    exc,
                    severity="info",
                    action="ran without a workspace subscription",
                )

    async def stop(self) -> None:
        """Stop watching, having said what was already in hand.

        A short run would otherwise end before the narrator's task was ever
        scheduled and every pending line would be cancelled unspoken.
        Draining is bounded by what is already queued.
        """
        self._unsubscribe()
        await self._drain()
        task, self._task = self._task, None
        if task is not None:
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, RuntimeError):
                pass

    def _unsubscribe(self) -> None:
        if not self._subscribed:
            return
        workspace = self._find_workspace()
        processors = getattr(workspace, "_processors", None) if workspace else None
        if isinstance(processors, list):
            workspace._processors = [fn for fn in processors if fn is not self._on_broadcast]
        self._subscribed = False

    async def _on_broadcast(self, event: Any) -> None:
        """Take a broadcast without ever holding up the workspace tick."""
        for winner in list(getattr(event, "winners", ()) or ()):
            if self._about and str(getattr(winner, "source", "")) != self._about:
                continue
            self.offer(winner)

    def offer(self, winner: Any) -> None:
        """Queue one piece of content, dropping it rather than waiting."""
        try:
            self._queue.put_nowait(winner)
        except asyncio.QueueFull:
            # Speaking is optional; deciding is not. A full queue means she is
            # deciding faster than she can talk, which is the right way round.
            self.dropped += 1

    async def _run(self) -> None:
        while True:
            try:
                winner = await self._queue.get()
            except asyncio.CancelledError:
                return
            await self._speak(winner)

    async def _drain(self) -> None:
        while True:
            try:
                winner = self._queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            await self._speak(winner)

    async def _speak(self, winner: Any) -> None:
        submitted = float(getattr(winner, "submitted_at", 0.0) or 0.0)
        if submitted and time.time() - submitted > STALE_AFTER_S:
            self.dropped += 1
            return
        metadata = getattr(winner, "metadata", None)
        payload = metadata.get("payload") if isinstance(metadata, dict) else None
        line = line_for(payload if payload else getattr(winner, "content", ""))
        if not line:
            return
        try:
            await self._utter(line)
            self.spoken += 1
        except (RuntimeError, AttributeError, TypeError, ValueError) as exc:
            record_degradation(
                "narrator",
                exc,
                severity="info",
                action="carried on after a line did not reach a surface",
            )

    async def _utter(self, line: str) -> None:
        if self._say is not None:
            result = self._say(line)
            if asyncio.iscoroutine(result):
                await result
            return
        from core.perception.ambient_presence import get_ambient_presence  # noqa: PLC0415

        get_ambient_presence().offer_utterance(line)


_narrator: Narrator | None = None


def get_narrator() -> Narrator | None:
    return _narrator


def start_narrating(**kwargs: Any) -> Narrator:
    """Begin saying what she is doing. Safe to call when one already runs."""
    global _narrator
    if _narrator is None:
        _narrator = Narrator(**kwargs)
        _narrator.start()
    return _narrator


async def stop_narrating() -> None:
    """Go quiet. Everything that was acting keeps acting."""
    global _narrator
    if _narrator is not None:
        await _narrator.stop()
        _narrator = None
