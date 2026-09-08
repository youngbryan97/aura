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

    # What recall put in mind, not what was just said. Bidding the last
    # working-memory item bids the turn that has only this moment finished —
    # which is fresh on every cycle, so it entered at full priority every time
    # and won almost every competition, and the other domains' bids never
    # decided anything. A memory bid should be a recollection.
    retrieved = list(getattr(cognition, "long_term_memory", []) or []) if cognition else []
    if retrieved:
        # At the middle, not the maximum. A recollection's claim on attention
        # is how well it matched what was asked, and that score is not carried
        # into the context the retriever writes — so there is no reading here
        # to price it by. The honest default for an unavailable reading is
        # neutral: entered at 1.0 it won almost every competition and no other
        # domain's bid decided anything, which is the same defect the exchange
        # bid had. Plumbing the retrieval score through would replace this.
        bids.append(
            CognitiveCandidate(
                content=str(retrieved[-1])[:240],
                source="memory",
                priority=0.5,
                content_type=ContentType.MEMORIAL,
            )
        )

    # The exchange that has just finished is current content and belongs in the
    # competition — but at the energy of the conversation, which is a reading,
    # not at a flat maximum. Entered at 1.0 it was fresh on every cycle and won
    # almost everything, and no other domain's bid decided anything.
    working = list(getattr(cognition, "working_memory", []) or []) if cognition else []
    if working:
        energy = _clamp(getattr(cognition, "conversation_energy", 0.5), 0.5)
        if energy > FLOOR:
            last = working[-1]
            bids.append(
                CognitiveCandidate(
                    content=str(last.get("content", ""))[:240] if isinstance(last, dict) else str(last)[:240],
                    source="exchange",
                    priority=energy,
                    content_type=ContentType.LINGUISTIC,
                )
            )

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
    except (ImportError, AttributeError, TypeError, ValueError) as exc:
        # An organ that is not loaded offers nothing to the competition, which
        # is a smaller candidate list rather than a fault.
        logger.debug("ontogeny had nothing to offer the workspace: %s", exc)

    # A surprising world is the oldest thing there is a competition for. The
    # world model computes its own prediction error every cycle and the only
    # route it had into the workspace was a heartbeat branch gated at a free
    # energy above 0.35, which is a different and much rarer event.
    try:
        from core.container import ServiceContainer

        model = ServiceContainer.get("unified_world_model", default=None)
        surprise = model.surprise() if model is not None else None
        if surprise is not None:
            level = _clamp(float(surprise))
            if level > FLOOR:
                bids.append(
                    CognitiveCandidate(
                        content=f"the world did not do what was predicted ({level:.2f})",
                        source="world_model",
                        priority=level,
                        content_type=ContentType.META,
                    )
                )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the world model had nothing to offer the workspace: %s", exc)

    # And a substrate that is moving fast. Volatility is the continuous
    # substrate's own reading of how much it is changing, which is what makes a
    # moment worth attending to before anything has named why.
    try:
        from core.runtime.service_registry import get_runtime_service

        substrate = get_runtime_service("conscious_substrate", default=None)
        reading = substrate.get_state_summary_nowait() if substrate is not None else None
        if isinstance(reading, dict) and not reading.get("snapshot_stale"):
            level = _clamp(float(reading.get("volatility", 0.0)) / 100.0)
            if level > FLOOR:
                bids.append(
                    CognitiveCandidate(
                        content=f"the substrate is moving ({level:.2f})",
                        source="substrate",
                        priority=level,
                        content_type=ContentType.SOMATIC,
                    )
                )
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("the substrate had nothing to offer the workspace: %s", exc)

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


#: Ticks the workspace may go without competing before the caller runs the
#: competition itself. One: if the heartbeat took a beat and did not arbitrate,
#: it is not going to.
STALE_TICKS: int = 1


async def feed_workspace(state: Any, workspace: Any) -> Any:
    """Submit this cycle's bids, and arbitrate only if nothing else will.

    Submission is the cycle's job and arbitration is the heartbeat's — one
    winner per cognitive tick is the whole design. The first version competed
    here as well, which emptied the candidate list before the heartbeat reached
    it: the heartbeat's own competition then found nothing, its winner was
    None, and the focus it hands the self-prediction loop was the string
    "none" on every beat. Two mechanisms both correct, and between them a
    self-model that never learned what she had been attending to.

    So this submits, and competes only when the workspace's tick has not moved
    since the last time it looked — which is what happens when there is no
    heartbeat running, and nothing else.
    """
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
    tick = int(getattr(workspace, "_tick", 0) or 0)
    seen = getattr(workspace, "_feed_last_tick", None)
    workspace._feed_last_tick = tick
    winner = getattr(workspace, "last_winner", None)
    # No evidence yet that anything else arbitrates, so arbitrate. A caller
    # with no heartbeat behind it would otherwise lose its first cycle, and
    # from the second call on the tick count says who is doing the work.
    if seen is None or tick - seen <= STALE_TICKS - 1:
        try:
            winner = await workspace.run_competition()
        except (AttributeError, RuntimeError, TypeError, ValueError) as exc:
            record_degradation(
                "workspace_feed", exc, severity="warning",
                action="no broadcast this cycle",
            )
            return None
        workspace._feed_last_tick = int(getattr(workspace, "_tick", tick) or tick)

    # What last won reaches this cycle's context whether this call arbitrated
    # or the heartbeat did. A broadcast the cycle cannot see has not been
    # broadcast to the part of her that answers.
    try:
        _remember_broadcast(state, winner, bool(getattr(workspace, "ignited", False)))
    except (AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "workspace_feed", exc, severity="debug",
            action="the broadcast did not reach this cycle's context",
        )
    return winner
