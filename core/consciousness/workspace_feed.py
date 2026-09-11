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
import math
from typing import Any

from core.runtime.errors import record_degradation
from core.soma.effort import note_effort
from core.state.percepts import read_percept

__all__ = ["build_candidates", "feed_workspace"]

logger = logging.getLogger("Aura.Consciousness.WorkspaceFeed")

#: Below this a bid is not worth the competition's time. It is the same number
#: the workspace already uses to decide a candidate is worth keeping, read from
#: there rather than repeated here if it is available.
FLOOR: float = 0.05

#: How a line that reached the cycle by winning the competition is marked, so
#: the memory bid can tell a recollection from something that has just been
#: attended to. One string, used by the writer and the reader, because two
#: spellings of it is how the pairing comes apart.
_BROADCAST_MARK: str = "[broadcast: "

#: Statuses that mean a goal is no longer an intention. A finished goal is a
#: record of what happened and has no claim on attention.
_FINISHED: frozenset[str] = frozenset({"done", "failed", "complete", "completed", "cancelled"})


def _clamp(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return max(0.0, min(1.0, out))


def _surprise_ratio(model: Any, surprise: Any) -> float:
    """How surprising this moment is against how surprising they usually are."""
    typical = 0.0
    try:
        status = model.status() if hasattr(model, "status") else {}
        detail = ((status or {}).get("facets", {}).get("learned", {}) or {}).get("detail", {}) or {}
        typical = max(0.0, float(detail.get("mean_surprise", 0.0) or 0.0))
    except (AttributeError, TypeError, ValueError):
        typical = 0.0
    now = max(0.0, float(surprise or 0.0))
    total = now + typical
    return 0.0 if total <= 1e-9 else now / total


def _intention_text(item: Any) -> str:
    """What this goal or initiative is, however the producer spelled it."""
    if not isinstance(item, dict):
        return str(item or "").strip()
    for key in ("goal", "description", "objective", "name", "content"):
        text = item.get(key)
        if isinstance(text, str) and text.strip():
            return text.strip()
    return ""


def _is_open_goal(goal: Any) -> bool:
    """Whether this is an intention at all.

    A goal marked done is a record of what happened. A goal with no text is not
    a goal — the goal engine writes one when the objective it was built from
    has neither an objective nor a name, and it arrives carrying a priority of
    one, which is a flat maximum that wins every competition and crowds every
    other domain out of the workspace.
    """
    if not isinstance(goal, dict):
        return bool(str(goal).strip())
    if str(goal.get("status", "")) in _FINISHED:
        return False
    return bool(_intention_text(goal))


def _goal_priority(goal: Any) -> float:
    """What an intention is asking to be thought about, if it has said.

    `urgency` only. A goal also carries a `priority`, and the two are not the
    same thing: priority is how important the work is once it has been chosen,
    urgency is how much it is asking to be thought about now. Read as a claim
    on attention, priority arrived as a flat one from the goal engine's own
    projection and won every competition in the workspace — deliberation
    beating perception, memory, affect and the body on every single turn, not
    because anything was pressing but because a default had been read as a
    demand.

    An intention that states no urgency has made no claim, so it enters at
    neutral and the competition decides.
    """
    if not isinstance(goal, dict):
        return 0.5
    stated = goal.get("urgency")
    if stated is None:
        return 0.5
    if isinstance(stated, str):
        named = {"critical": 1.0, "high": 0.8, "medium": 0.5, "normal": 0.5, "low": 0.25}
        return named.get(stated.strip().lower(), 0.5)
    return _clamp(stated, 0.5)


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
        # Read through the shared reading rather than off the raw dict. This
        # bid was priced from a `salience` key that no producer in the tree has
        # ever written, so every real percept bid zero and perception never
        # once reached the workspace. Producers state `intensity`; a stated
        # strength is a stated claim on attention.
        latest = read_percept(percepts[-1])
        if latest.salience > FLOOR:
            bids.append(
                CognitiveCandidate(
                    content=latest.content or latest.kind,
                    source="perception",
                    priority=latest.salience,
                    content_type=ContentType.PERCEPTUAL,
                )
            )

    if affect is not None:
        emotions = getattr(affect, "emotions", {}) or {}
        if emotions:
            name = max(emotions, key=lambda key: _clamp(emotions[key]))
            intensity = _clamp(emotions[name])
            if intensity > FLOOR:
                # How far this feeling is above where it usually sits, which is
                # a property of the bid. It was the moment's global arousal —
                # and `priority_at` adds three tenths of the affect weight to
                # every candidate's claim, so a quantity belonging to the whole
                # moment became one competitor's private advantage. Nothing
                # else in the tree sets the field, so affect carried a bonus no
                # other domain could earn and won the competition on ninety-six
                # turns in a hundred: attention was `affect_*` whatever else
                # was happening, and the action chosen by what she was
                # attending to was the same action every turn.
                baselines = getattr(affect, "mood_baselines", {}) or {}
                charge = _clamp(intensity - _clamp(baselines.get(name, 0.0)))
                bids.append(
                    CognitiveCandidate(
                        content=f"feeling {name}",
                        source=f"affect_{name}",
                        priority=intensity,
                        content_type=ContentType.AFFECTIVE,
                        affect_weight=charge,
                    )
                )

    # What recall put in mind, not what was just said. Bidding the last
    # working-memory item bids the turn that has only this moment finished —
    # which is fresh on every cycle, so it entered at full priority every time
    # and won almost every competition, and the other domains' bids never
    # decided anything. A memory bid should be a recollection.
    # A recollection's claim on attention is how well it matched what was
    # asked. Retrieval ranks its candidates by exactly that and used to discard
    # the number, so this bid entered at a flat neutral 0.5 — nothing about
    # what was recalled could change what won, and active memory had no route
    # into attention at all.
    #
    # What is in mind because it just won the competition is excluded. It
    # enters this list at the strength it won with, so bidding it back made a
    # loop: the winner became the strongest recollection, which won again, and
    # the competition settled on whatever had won once. Something that has just
    # had attention has no claim on it — the same reason the workspace fatigues
    # a repeated winner, applied to the channel that was going round it. It
    # stays in mind for the reply; it stops competing.
    retrieved = list(getattr(cognition, "long_term_memory", []) or []) if cognition else []
    scores = list(getattr(cognition, "memory_scores", []) or []) if cognition else []
    recalled = [
        (str(text), scores[index] if index < len(scores) else 0.5)
        for index, text in enumerate(retrieved)
        if not str(text).startswith(_BROADCAST_MARK)
    ]
    if recalled:
        strength = _clamp(max(score for _, score in recalled), 0.5)
        bids.append(
            CognitiveCandidate(
                content=recalled[-1][0][:240],
                source="memory",
                priority=strength,
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

    # Only the ones still open. A goal marked done is a record of what
    # happened, not an intention — and the action arm appends one on every turn
    # it acts, at the priority of a completed thing, which is a flat maximum
    # that won every competition and crowded every other domain out of the
    # workspace. The same defect the memory and exchange bids had.
    goals = [
        goal
        for goal in (list(getattr(cognition, "active_goals", []) or []) if cognition else [])
        if _is_open_goal(goal)
    ]
    # One bid per intention, however many lists it appears in. The same
    # intention is projected into both `active_goals` and
    # `pending_initiatives`, and bid twice it competes with itself.
    spoken: set[str] = set()
    # The two that are asking hardest, not the two that happen to be last in
    # the list. Position in `active_goals` is the order things were written,
    # and a goal appended this turn because a need just became pressing sat at
    # the end behind five older ones — so the slice read whichever two the
    # bookkeeping had left there. An intention's claim on attention is what it
    # states, and that is what decides which two get to make it.
    for goal in sorted(goals, key=_goal_priority, reverse=True)[:2]:
        # `priority` is what the goal engine writes; `urgency` is what this bid
        # read, and no producer in the tree has ever written it. So every real
        # goal bid zero and deliberation never once reached the workspace. A
        # goal that is active and carries no stated priority still has a claim
        # — it is on the list — so the fallback is neutral rather than silence.
        urgency = _goal_priority(goal)
        text = _intention_text(goal)
        if urgency > FLOOR and text not in spoken:
            spoken.add(text)
            bids.append(
                CognitiveCandidate(
                    content=text[:240],
                    source="deliberation",
                    priority=urgency,
                    content_type=ContentType.INTENTIONAL,
                )
            )

    # And what she has decided to do about it. An initiative is an intention
    # competing for attention, which is what this competition is for — and the
    # motivation phase writes them to `pending_initiatives` while this bid read
    # `active_goals`, so the one thing deliberation actually produces within a
    # turn never reached the workspace at all.
    initiatives = list(getattr(cognition, "pending_initiatives", []) or []) if cognition else []
    for initiative in initiatives[:2]:
        # An initiative's claim on attention is its `urgency` — what the
        # intention itself asks for. Several producers also carry a `priority`,
        # which is how important the work is once chosen, not how much it is
        # asking to be thought about; read as a claim it arrived as a flat
        # maximum and won every competition. A record that states no urgency
        # has not made a claim, so it enters at neutral.
        text = _intention_text(initiative)
        if not text or text in spoken:
            continue
        spoken.add(text)
        urgency = (
            _clamp(initiative.get("urgency"), 0.5)
            if isinstance(initiative, dict) and initiative.get("urgency") is not None
            else 0.5
        )
        if urgency > FLOOR:
            bids.append(
                CognitiveCandidate(
                    content=text[:240],
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
            # Against the model's own running mean, not squashed and not
            # clipped. Prediction error is unbounded above, so a clip turned
            # every surprise past one into the same maximum and a squash went
            # flat not far after that — either way this bid sat at its ceiling
            # on every turn and stopped being a reading of anything. What
            # matters is whether the moment is more surprising than this
            # model's moments usually are.
            level = _clamp(_surprise_ratio(model, surprise))
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
            # How fast the substrate is moving against how far it is from rest,
            # which is a reading of the same kind as the world model's surprise
            # beside it: scale-free, and sensitive wherever this substrate's
            # own amplitude happens to sit.
            #
            # It was the summary's `volatility` divided by a hundred — and the
            # summary multiplies the mean velocity by a hundred to make it
            # readable, so the two cancelled and the bid was the raw mean
            # velocity, about two thousandths. The floor every bid has to clear
            # is five hundredths. Recurrent cognition could not reach attention
            # by any route, on any turn, at any speed.
            speed = max(0.0, float(reading.get("volatility", 0.0)) / 100.0)
            amplitude = max(0.0, float(reading.get("global_energy", 0.0)))
            total = speed + amplitude
            level = _clamp(0.0 if total <= 1e-9 else speed / total)
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
        # Her own exertion counts, and it is the part of this reading she
        # causes. The rest is the machine, most of whose load is not hers.
        pressure = max(
            _clamp(float(hardware.get("cpu_usage", 0.0) or 0.0) / 100.0),
            _clamp(float(hardware.get("temperature", 0.0) or 0.0) / 100.0),
            _clamp(getattr(soma, "exertion", 0.0)),
        )
        # At the same floor as every other bid. This one alone was gated at a
        # half, so the body could only speak when the machine was already at
        # fifty percent — and deciding in advance which bids are worth hearing
        # is the competition's job, which it is better at than a constant.
        if pressure > FLOOR:
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
    line = f"{_BROADCAST_MARK}{winner.source}] {str(winner.content)[:180]}"
    context = list(getattr(cognition, "long_term_memory", []) or [])
    scores = list(getattr(cognition, "memory_scores", []) or [])
    # The two lists are read side by side — the workspace prices its memory bid
    # from the score at the same index — so anything that writes one has to
    # write the other or the pairing silently comes apart. What is in mind
    # because it won the competition is in mind at the strength it won with.
    if len(scores) != len(context):
        scores = scores[: len(context)] + [0.5] * max(0, len(context) - len(scores))
    keep = [
        index for index, item in enumerate(context) if not str(item).startswith("[broadcast: ")
    ]
    context = [context[index] for index in keep]
    scores = [scores[index] for index in keep]
    context.append(line)
    scores.append(_clamp(getattr(winner, "effective_priority", 0.5), 0.5))
    cognition.long_term_memory = context[-CONTEXT_LIMIT:]
    cognition.memory_scores = scores[-CONTEXT_LIMIT:]


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
        # Weighing them costs something, and what it costs varies with how
        # much the rest of her had to say. Without a reporter here the body's
        # sense of its own exertion moved only with recall, which finds nothing
        # offline — so the one channel an experiment can leave free was a
        # constant.
        note_effort("candidates", len(bids))
        _remember_broadcast(state, winner, bool(getattr(workspace, "ignited", False)))
    except (AttributeError, TypeError, ValueError) as exc:
        record_degradation(
            "workspace_feed", exc, severity="debug",
            action="the broadcast did not reach this cycle's context",
        )
    return winner
