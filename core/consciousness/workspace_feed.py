"""Giving the global workspace something to compete over.

The workspace is built as a real arbiter: candidates carry perceptual,
affective, memorial, intentional, somatic and metacognitive content, priority
is shaped by affect and free energy, there is one winner per tick, an ignition
threshold, winner fatigue to force rotation, and an inhibition gate. Then the
candidate list was empty on every tick of the cognitive cycle. Two places in
the tree submit anything at all, both inside rare branches of the soul's drive
handling, so the competition was almost always between nothing and nothing and
the winner was None.

A workspace with no candidates is not a workspace. This turns the cycle's own
contents into bids — one per domain that has something to say — and lets them
compete.

Every priority here is read off the state rather than chosen: a percept bids
with its own salience, a feeling with its own intensity, a goal with its own
urgency, a coherence problem with the size of the problem. A domain with
nothing to say does not bid, which is what keeps the competition a competition
rather than a round-robin over ten fixed sources.
"""

from __future__ import annotations

import logging
from typing import Any

from core.runtime.errors import record_degradation

__all__ = ["build_candidates", "feed_workspace"]

logger = logging.getLogger("Aura.Consciousness.WorkspaceFeed")

#: Below this a bid is not worth the competition's time. It is the same number
#: the workspace already uses to decide a candidate is worth keeping, read from
#: there rather than repeated here if it is available.
FLOOR: float = 0.05


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, out))


def build_candidates(state: Any) -> list[Any]:
    """One bid per domain that has something to say, priced by the state."""
    from core.consciousness.global_workspace import CognitiveCandidate, ContentType

    bids: list[Any] = []
    affect = getattr(state, "affect", None)
    cognition = getattr(state, "cognition", None)
    world = getattr(state, "world", None)
    identity = getattr(state, "identity", None)
    soma = getattr(state, "soma", None)

    percepts = list(getattr(world, "recent_percepts", []) or []) if world else []
    if percepts:
        latest = percepts[-1]
        salience = _clamp(latest.get("salience") if isinstance(latest, dict) else 0.0)
        if salience > FLOOR:
            bids.append(
                CognitiveCandidate(
                    content=str(latest.get("content", ""))[:240] if isinstance(latest, dict) else str(latest)[:240],
                    source="perception",
                    priority=salience,
                    content_type=ContentType.PERCEPTUAL,
                )
            )

    if affect is not None:
        emotions = getattr(affect, "emotions", {}) or {}
        if emotions:
            name = max(emotions, key=lambda key: _clamp(emotions[key]))
            intensity = _clamp(emotions[name])
            if intensity > FLOOR:
                bids.append(
                    CognitiveCandidate(
                        content=f"feeling {name}",
                        source=f"affect_{name}",
                        priority=intensity,
                        content_type=ContentType.AFFECTIVE,
                        # The affect weight the competition was designed around
                        # and almost never received: arousal is how urgent a
                        # feeling is, which is exactly what this field is for.
                        affect_weight=_clamp(getattr(affect, "arousal", 0.0)),
                    )
                )

    working = list(getattr(cognition, "working_memory", []) or []) if cognition else []
    if working:
        last = working[-1]
        stamp = 0.0
        if isinstance(last, dict):
            try:
                stamp = float(last.get("timestamp", 0.0) or 0.0)
            except (TypeError, ValueError):
                stamp = 0.0
        memory_bid = CognitiveCandidate(
            content=str(last.get("content", ""))[:240] if isinstance(last, dict) else str(last)[:240],
            source="memory",
            priority=1.0,
            content_type=ContentType.MEMORIAL,
        )
        # Entered as of when the memory was formed, not when the bid was made.
        # The workspace already decays priority with arrival age, so dating the
        # bid correctly is what makes a stale recollection lose to a live
        # feeling instead of winning every tick on a flat 1.0.
        if stamp > 0.0:
            memory_bid.submitted_at = stamp
        bids.append(memory_bid)

    goals = list(getattr(cognition, "active_goals", []) or []) if cognition else []
    for goal in goals[-2:]:
        urgency = _clamp(goal.get("urgency", 0.0) if isinstance(goal, dict) else 0.0)
        if urgency > FLOOR:
            bids.append(
                CognitiveCandidate(
                    content=str(goal.get("goal", goal))[:240] if isinstance(goal, dict) else str(goal)[:240],
                    source="deliberation",
                    priority=urgency,
                    content_type=ContentType.INTENTIONAL,
                )
            )

    if cognition is not None:
        trouble = max(
            _clamp(getattr(cognition, "fragmentation_score", 0.0)),
            1.0 - _clamp(getattr(cognition, "coherence_score", 1.0), 1.0),
        )
        if trouble > FLOOR:
            bids.append(
                CognitiveCandidate(
                    content=f"coherence is {1.0 - trouble:.2f}",
                    source="metacognition",
                    priority=trouble,
                    content_type=ContentType.META,
                )
            )

    if identity is not None:
        instability = 1.0 - _clamp(getattr(identity, "stability", 1.0), 1.0)
        if instability > FLOOR:
            bids.append(
                CognitiveCandidate(
                    content=f"identity stability {1.0 - instability:.2f}",
                    source="self",
                    priority=instability,
                    content_type=ContentType.META,
                )
            )

    # An unprecedented moment deserves attention. The lifetime state computes
    # exactly that number every cycle and nothing competed on it, so a life
    # that had never seen anything like this bid the same as one on a familiar
    # afternoon.
    #
    # It bids at whatever the novelty is, with no threshold of its own. A first
    # version only entered above 0.5 and therefore never entered at all: the
    # reservoir puts an ordinary moment near 0.2, so the gate excluded every
    # real reading and admitted only the 0.5 it returns before it has a
    # distribution to compare against. Deciding in advance which bids are worth
    # hearing is the workspace's job, and it is better at it than a constant.
    try:
        from core.ontogeny.lifetime import last_reading

        reading = last_reading()
        novelty = _clamp(getattr(reading, "novelty", 0.0)) if reading is not None else 0.0
        if novelty > FLOOR:
            bids.append(
                CognitiveCandidate(
                    content=f"this is unlike the ordinary run of things ({novelty:.2f})",
                    source="ontogeny",
                    priority=novelty,
                    content_type=ContentType.META,
                )
            )
    except (ImportError, AttributeError, TypeError, ValueError):
        pass

    if soma is not None:
        hardware = getattr(soma, "hardware", {}) or {}
        pressure = max(
            _clamp(float(hardware.get("cpu_usage", 0.0) or 0.0) / 100.0),
            _clamp(float(hardware.get("temperature", 0.0) or 0.0) / 100.0),
        )
        if pressure > 0.5:
            bids.append(
                CognitiveCandidate(
                    content=f"body under load {pressure:.2f}",
                    source="interoception",
                    priority=pressure,
                    content_type=ContentType.SOMATIC,
                )
            )

    return bids


#: How many broadcast lines the cycle's context keeps. One per tick, bounded,
#: because this is what the response is generated from and a growing list of
#: them would crowd out the retrieval it sits beside.
CONTEXT_LIMIT: int = 4


def _remember_broadcast(state: Any, winner: Any, ignited: bool) -> None:
    """Where she is looking, and — if it ignited — what the cycle is working from.

    This is the claim global workspace theory actually makes: what wins the
    competition becomes available to the specialised processes, and the largest
    specialised process here is the one that answers. Without this the winner
    reached the substrate, the self model and affect, and never reached the
    thing that speaks.

    Only on ignition, and bounded. A broadcast that did not ignite did not
    become globally available, and saying so in the context would be asserting
    the opposite of what the competition decided.
    """
    cognition = getattr(state, "cognition", None)
    if winner is None or cognition is None:
        return
    # Where she is looking, which is a different question from whether it went
    # global. `cognition.attention_focus` is read by the mind-moment
    # reconstruction and by the being runtime and was written by nobody, so
    # both of them saw None for the life of the process while the attention
    # schema beside them held the answer.
    cognition.attention_focus = f"{winner.source}: {str(winner.content)[:120]}"
    if not ignited:
        return
    line = f"[broadcast: {winner.source}] {str(winner.content)[:180]}"
    context = list(getattr(cognition, "long_term_memory", []) or [])
    context = [item for item in context if not str(item).startswith("[broadcast: ")]
    context.append(line)
    cognition.long_term_memory = context[-CONTEXT_LIMIT:]


async def feed_workspace(state: Any, workspace: Any) -> Any:
    """Submit this cycle's bids and run the competition. Returns the winner."""
    if workspace is None:
        return None
    try:
        bids = build_candidates(state)
    except (AttributeError, TypeError, ValueError, ImportError) as exc:
        record_degradation(
            "workspace_feed", exc, severity="warning",
            action="the workspace competed over whatever was already pending",
        )
        bids = []
    for bid in bids:
        try:
            await workspace.submit(bid)
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "workspace_feed", exc, severity="debug",
                action=f"bid from {bid.source} was not submitted",
            )
    try:
        winner = await workspace.run_competition()
    except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
        record_degradation(
            "workspace_feed", exc, severity="warning",
            action="no broadcast this cycle",
        )
        return None
    try:
        _remember_broadcast(state, winner, bool(getattr(workspace, "ignited", False)))
    except (AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "workspace_feed", exc, severity="debug",
            action="the broadcast did not reach this cycle's context",
        )
    return winner
