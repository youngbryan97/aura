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

import contextvars
import logging
from typing import Any

from core.runtime.errors import record_degradation

__all__ = [
    "consumer_activity",
    "register_broadcast_consumers",
    "reset_consumer_activity",
]

logger = logging.getLogger("Aura.Consciousness.Broadcast")



def _winner(event: Any) -> Any:
    winners = getattr(event, "winners", None) or []
    return winners[0] if winners else None


#: Which need each kind of broadcast content serves. The budgets are named
#: for what they are, and so are the sources, so this is a reading of both
#: rather than an invention: an exchange is social contact, a percept or a
#: recollection or an unprecedented moment is something learned, noticing one's
#: own incoherence is integrity, the body is energy, and an intention pursued
#: is growth.
SOURCE_DRIVES: dict[str, str] = {
    "exchange": "social",
    "perception": "curiosity",
    "memory": "curiosity",
    "ontogeny": "curiosity",
    "world_model": "curiosity",
    "substrate": "energy",
    "interoception": "energy",
    "metacognition": "integrity",
    "self": "integrity",
    "deliberation": "growth",
}


#: Which need each feeling bears on. Written out rather than derived by
#: stripping a prefix: `affect_joy` is not a drive called joy, and no budget has
#: ever been called that. Prefix-stripping was defining psychology, and what it
#: defined was five budgets and thirty emotions that mostly named none of them.
#:
#: The mapping is by what the feeling is about. Warmth and its absence are
#: about company. Interest and its absence are about finding out. Fear and
#: vulnerability are about holding together. Frustration and anger are about
#: not yet being able, which is what growth is the drive for.
AFFECT_DRIVES: dict[str, str] = {
    "joy": "social",
    "trust": "social",
    "warmth": "social",
    "belonging": "social",
    "happiness": "social",
    "gratitude": "social",
    "loneliness": "social",
    "longing": "social",
    "sadness": "social",
    "curiosity": "curiosity",
    "interest": "curiosity",
    "wonder": "curiosity",
    "anticipation": "curiosity",
    "surprise": "curiosity",
    "excitement": "curiosity",
    "boredom": "curiosity",
    "apathy": "curiosity",
    "fear": "integrity",
    "dread": "integrity",
    "vulnerability": "integrity",
    "upset": "integrity",
    "terror": "integrity",
    "unhappiness": "integrity",
    "frustration": "growth",
    "anger": "growth",
    "confused": "growth",
    "disgust": "growth",
    "pride": "growth",
    "satisfaction": "growth",
    "hope": "growth",
    "relief": "growth",
    "inspiration": "growth",
    "nostalgia": "social",
    "love": "social",
    "empathy": "social",
    "admiration": "social",
    "contempt": "social",
    "submission": "social",
    "awe": "curiosity",
    "amusement": "curiosity",
    "indifference": "curiosity",
    "cynicism": "integrity",
    "remorse": "integrity",
    "aggressiveness": "growth",
}


def _drive_served(source: str) -> str:
    """The budget a broadcast from this source replenishes, or an empty name."""
    if source.startswith("drive_"):
        return source[len("drive_") :]
    if source.startswith("affect_"):
        return AFFECT_DRIVES.get(source[len("affect_") :], "")
    return SOURCE_DRIVES.get(source, "")


#: How often each consumer was called and how often it changed anything.
#:
#: A consumer that returns early every time is a registered processor with no
#: effect, and registration alone cannot tell you: the list of what is wired
#: looks the same either way. Every one of these takes the winner and writes
#: somewhere its destination domain reads, so a run where one of them never
#: writes is a run where global access is narrower than the report says.
_CALLED: dict[str, int] = {}
_WROTE: dict[str, int] = {}
#: Ran with a winner to act on and reported no write. Counted separately
#: because it is neither "wrote" nor "was never called", and the two it sits
#: between are the two this monitor exists to tell apart.
_QUIET: dict[str, int] = {}

#: Which consumer is running, so :func:`_wrote` needs no argument and the
#: consumers keep the signature the workspace calls them with.
_RUNNING: contextvars.ContextVar[str] = contextvars.ContextVar(
    "broadcast_consumer", default=""
)


def _wrote() -> None:
    """The running consumer changed the state its destination reads.

    Called at the point the write lands, never before it. Everything this
    monitor reports rests on that being true.
    """
    name = _RUNNING.get()
    if name:
        _WROTE[name] = _WROTE.get(name, 0) + 1


def _counted(name: str, consumer: Any) -> Any:
    """Wrap a consumer so the run can say whether it ever did anything.

    A consumer that ran with a winner and reported nothing used to be counted
    as having written, on the ground that calling a working consumer dead is
    worse. That reasoning had a hole big enough to swallow the whole monitor:
    nothing ever set the marker, so EVERY write in this report was the
    fallback inventing one, and a consumer that silently did nothing was
    indistinguishable from one that worked — which is the state of affairs the
    file was written to detect.

    So the invention is gone and the third state is named instead. A consumer
    that does not report is `ran_without_writing`, which is a thing a person
    can go and look at.
    """
    import functools

    @functools.wraps(consumer)
    async def counted(event: Any) -> None:
        _CALLED[name] = _CALLED.get(name, 0) + 1
        before = _WROTE.get(name, 0)
        token = _RUNNING.set(name)
        try:
            await consumer(event)
        finally:
            _RUNNING.reset(token)
        if _WROTE.get(name, 0) == before and _winner(event) is not None:
            _QUIET[name] = _QUIET.get(name, 0) + 1

    return counted


def consumer_activity() -> dict[str, Any]:
    """Which broadcast consumers have done anything, and which never have."""
    return {
        "called": dict(sorted(_CALLED.items())),
        "wrote": dict(sorted(_WROTE.items())),
        "ran_without_writing": dict(sorted(_QUIET.items())),
        "never_wrote": sorted(name for name in _CALLED if not _WROTE.get(name)),
    }


def reset_consumer_activity() -> None:
    _CALLED.clear()
    _WROTE.clear()
    _QUIET.clear()


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
            _wrote()
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
            _wrote()
        except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="the self model did not record the broadcast")

    async def to_affect(event: Any) -> None:
        """Ignition is arousal, left where the affect phase will find it.

        The first version wrote the blended arousal onto the affect engine,
        which nothing reads back into `AuraState.affect` — a channel of exactly
        the kind this file was written to fix. The reading is left on the
        workspace instead and `AffectUpdatePhase` picks it up after its own
        emotion channels have settled, which is a cycle later and is where
        affect can actually keep it.
        """
        winner = _winner(event)
        if winner is None:
            return
        try:
            level = float(getattr(workspace, "ignition_level", 0.0) or 0.0)
            workspace.last_broadcast_arousal = {
                "ignition": max(0.0, min(1.0, level)),
                "priority": max(0.0, min(1.0, float(winner.effective_priority))),
                "source": str(winner.source)[:64],
            }
            _wrote()
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="affect did not take the broadcast")

    async def to_deliberation(event: Any) -> None:
        """Whatever wins the workspace is a need being attended to.

        The motivation phase already treats attention as satisfying: active
        conversation slows the social drive's decay and then replenishes it.
        Winning the broadcast is the same thing measured at the competition
        rather than at the conversation, so it is left where that phase reads
        it and applied there.

        It used to fire only for a `drive_` winner. Drive candidates enter the
        competition through an alert gated at seventy percent urgency and five
        minutes since the last one, so in an ordinary hour of thinking this
        consumer set nothing at all and attention had no motivational
        consequence whatsoever. Every source serves some need — attending to an
        exchange satisfies the social one, to a recollection or a percept the
        curious one, to her own coherence the integrity one — and the mapping
        is written down rather than left to a prefix that almost never matches.

        The first version called `note_pressure` on the goal engine, which has
        no such method — a guarded no-op, in the file written to fix exactly
        that.
        """
        winner = _winner(event)
        if winner is None:
            return
        try:
            source = str(getattr(winner, "source", ""))
            drive = _drive_served(source)
            if not drive:
                return
            workspace.last_drive_attention = {
                "drive": drive,
                "priority": max(0.0, min(1.0, float(winner.effective_priority))),
            }
            _wrote()
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="deliberation did not take the broadcast")

    async def to_perception(event: Any) -> None:
        """What is globally available biases what is noticed next.

        The one coupling the theory is most explicit about, and the one this
        file did not have: the broadcast reached recurrent cognition, the self
        model, affect and deliberation, and perception was not among them. So
        nothing she was attending to could change what she noticed, and the
        only route into perception at all was the readback of a file she had
        just written.

        Left on the workspace, where `core.state.percepts.emit_percept` reads
        it as an arriving percept is stamped — the same device the affect
        consumer uses, and for the same reason: a consumer that writes where
        its destination cannot look is a consumer that does not count.
        """
        winner = _winner(event)
        if winner is None:
            return
        try:
            workspace.last_broadcast_attention = {
                "source": str(winner.source)[:64],
                "content": str(winner.content)[:512],
                "priority": max(0.0, min(1.0, float(winner.effective_priority))),
            }
            _wrote()
        except (AttributeError, TypeError, ValueError) as exc:
            record_degradation("broadcast_consumers", exc, severity="debug",
                               action="perception did not take the broadcast")

    # No memory consumer. There was one, and it appended a bounded trace to an
    # attribute on the workspace that nothing anywhere reads — a writer with no
    # reader, inside the file written to remove writers with no readers. What
    # actually carries a broadcast into memory is `_remember_broadcast` in
    # `workspace_feed`, which writes the ignited winner into
    # `cognition.long_term_memory`, where recall and the memory domain both
    # look. A processor that writes where its destination cannot look is a
    # processor that does not count.
    for name, consumer in (
        ("recurrent_cognition", to_recurrent_cognition),
        ("self_model", to_self_model),
        ("affect", to_affect),
        ("deliberation", to_deliberation),
        ("perception", to_perception),
    ):
        try:
            workspace.register_processor(_counted(name, consumer))
            registered.append(name)
        except (AttributeError, TypeError) as exc:
            logger.warning("broadcast consumer %s not registered: %s", name, exc)
    return registered
