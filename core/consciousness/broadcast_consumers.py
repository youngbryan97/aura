"""What a broadcast reaches, other than a log line.

The workspace runs a real competition: candidates carry perceptual, affective,
memorial, intentional and metacognitive content, priority is shaped by affect
and free energy, there is a winner per tick, an ignition threshold and winner
fatigue. Then the winner was handed to an attention schema, an event emitter,
and a list of registered processors that nothing ever registered anything on.

Global availability is the whole content of the workspace idea. A broadcast
that no specialised process consumes has not been broadcast; it has been
logged. So these are the consumers, one per domain the winner is supposed to
become available to, and each one does the smallest real thing that domain
would do with it.

The recurrent cognition consumer is the one worth pointing at:
`LiquidSubstrate.encode_text_to_stimulus` was written to turn content into a
stimulus vector for the substrate and was called from nowhere. The winning
content now drives it, which is the reentrant half of global access — what wins
changes the substrate, the substrate shapes the next competition.

Every consumer is cheap, total, and returns rather than raises. The workspace
wraps them anyway; failing quietly inside one of these would still cost the
tick that called it.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["register_broadcast_consumers"]

logger = logging.getLogger("Aura.Consciousness.Broadcast")

#: How many broadcasts the memory trace keeps. Bounded because it is written on
#: every tick and read as recent context, not as a history.
TRACE_LIMIT: int = 32


def _winner(event: Any) -> Any:
    winners = getattr(event, "winners", None) or []
    return winners[0] if winners else None


def register_broadcast_consumers(workspace: Any, *, substrate: Any = None) -> list[str]:
    """Wire the winner to the domains it is supposed to become available to.

    Returns the names of the consumers that were registered, so a caller can
    record which ones this runtime actually has rather than assuming all of
    them.
    """
    if workspace is None or not hasattr(workspace, "register_processor"):
        return []

    registered: list[str] = []

    async def to_recurrent_cognition(event: Any) -> None:
        """The winner stimulates the substrate. The reentrant half of access."""
        winner = _winner(event)
        if winner is None or substrate is None:
            return
        try:
            stimulus = substrate.encode_text_to_stimulus(str(winner.content)[:512])
            # `encode_text_to_stimulus` and `inject_stimulus` were written for
            # each other and nothing connected them. The weight is the winner's
            # own effective priority, so a broadcast that barely won barely
            # moves the substrate.
            await substrate.inject_stimulus(
                stimulus, weight=min(1.0, max(0.0, float(winner.effective_priority)))
            )
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "broadcast_consumers",
                exc,
                severity="debug",
                action="the substrate did not take the broadcast stimulus",
            )

    async def to_self_model(event: Any) -> None:
        """What she is attending to is part of what she knows about herself."""
        winner = _winner(event)
        if winner is None:
            return
        try:
            from core.container import ServiceContainer

            model = ServiceContainer.get("self_model", default=None)
            beliefs = getattr(model, "beliefs", None)
            if beliefs is None:
                return
            beliefs["attending:source"] = str(winner.source)[:64]
            beliefs["attending:priority"] = round(float(winner.effective_priority), 4)
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="the self model did not record the broadcast")

    async def to_affect(event: Any) -> None:
        """Ignition is arousal. A broadcast that ignited should raise it."""
        winner = _winner(event)
        if winner is None:
            return
        try:
            from core.container import ServiceContainer

            engine = ServiceContainer.get("affect_engine", default=None)
            level = float(getattr(workspace, "ignition_level", 0.0) or 0.0)
            if engine is None or level <= 0.0:
                return
            # The blend weight is the ignition level itself: a competition that
            # barely ignited moves arousal barely, a full ignition moves it most
            # of the way to the winner's priority.
            current = float(getattr(engine, "arousal", 0.5) or 0.5)
            target = min(1.0, max(0.0, float(winner.effective_priority)))
            if hasattr(engine, "arousal"):
                engine.arousal = (1.0 - level) * current + level * target
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="affect did not take the broadcast")

    async def to_memory(event: Any) -> None:
        """A bounded trace of what won, so recall can see what she attended to."""
        winner = _winner(event)
        if winner is None:
            return
        try:
            trace = getattr(workspace, "broadcast_trace", None)
            if trace is None:
                trace = []
                workspace.broadcast_trace = trace
            trace.append(
                {
                    "source": str(winner.source)[:64],
                    "content": str(winner.content)[:240],
                    "priority": round(float(winner.effective_priority), 4),
                }
            )
            del trace[:-TRACE_LIMIT]
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="the broadcast trace was not written")

    async def to_deliberation(event: Any) -> None:
        """A drive that keeps winning is a drive that is asking for something."""
        winner = _winner(event)
        if winner is None:
            return
        try:
            from core.container import ServiceContainer

            source = str(getattr(winner, "source", ""))
            if not source.startswith("drive_"):
                return
            goals = ServiceContainer.get("goal_engine", default=None)
            note = getattr(goals, "note_pressure", None)
            if callable(note):
                note(source[len("drive_"):], float(winner.effective_priority))
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="deliberation did not take the broadcast")

    for name, consumer in (
        ("recurrent_cognition", to_recurrent_cognition),
        ("self_model", to_self_model),
        ("affect", to_affect),
        ("memory", to_memory),
        ("deliberation", to_deliberation),
    ):
        try:
            workspace.register_processor(consumer)
            registered.append(name)
        except (AttributeError, TypeError) as exc:
            logger.warning("broadcast consumer %s not registered: %s", name, exc)
    return registered
