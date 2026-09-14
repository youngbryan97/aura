"""Displacing one domain of the core state: do(X_i = x_i + delta).

Each writer here touches the same attributes its reader in
`core.subject.state` reads, so an intervention that shows nothing is a fact
about the coupling and not about the instrument. `perturbable` names the
domains that have a writer at all, and the battery refuses to score a domain
that has none. `perturb_organs` displaces the half of a domain that lives in an
organ rather than in the state object, through the organ's own public path.

The readers stay in `state`; this module is the writers. Both are one
instrument, and `state` re-exports what is here so every caller keeps its
import.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np

from core.state.percepts import emit_percept
from core.subject import state as _state


def _bump(obj: Any, path: str, delta: float, lo: float, hi: float) -> bool:
    parts = path.split(".")
    node = obj
    for part in parts[:-1]:
        node = node.get(part) if isinstance(node, Mapping) else getattr(node, part, None)
        if node is None:
            return False
    name = parts[-1]
    if isinstance(node, dict):
        current = _state._f(node.get(name))
        node[name] = min(hi, max(lo, current + delta))
        return True
    if not hasattr(node, name):
        return False
    current = _state._f(getattr(node, name))
    setattr(node, name, min(hi, max(lo, current + delta)))
    return True


def _perturb_P(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    world = _state._dig(state, "world", None)
    if world is None or not isinstance(getattr(world, "recent_percepts", None), list):
        return False
    # In the shape a percept actually arrives in. The first version wrote a
    # dict with `source` and `salience` and no `type`, which is not a percept
    # any part of the tree produces: the affect phase keys on the type and
    # dropped it, the workspace read a salience nobody writes, and the
    # displacement of perception was a displacement of a list's length.
    #
    # `novel_stimulus` because that is what a probe is — something arrived that
    # was not predicted — and the strength is the displacement itself, in the
    # same units every other domain is displaced in.
    #
    # `world.spatial_context` is deliberately not written. It is a field with
    # no reader anywhere in the tree, so displacing it would move this domain's
    # own vector and could not move anything else: it would inflate the
    # coupling gain with a number that means nothing.
    emit_percept(
        world,
        "novel_stimulus",
        content=f"probe delta {delta:+.4f}",
        intensity=min(1.0, abs(delta)),
    )
    return True


def _perturb_I(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    # These fields are percentages and milliseconds, not fractions. The first
    # version clamped cpu_usage to [0, 1] while the runtime writes 0..100, so
    # displacing the body by +0.15 set it to one percent — a large move in the
    # wrong direction, dressed as a small one in the right one.
    hit = _bump(state, "soma.hardware.temperature", delta * 20.0, 0.0, 110.0)
    hit |= _bump(state, "soma.hardware.cpu_usage", delta * 100.0, 0.0, 100.0)
    hit |= _bump(state, "soma.hardware.vram_usage", delta * 100.0, 0.0, 100.0)
    hit |= _bump(state, "soma.hardware.ram_usage", delta * 100.0, 0.0, 100.0)
    hit |= _bump(state, "soma.latency.last_thought_ms", delta * 500.0, 0.0, 60_000.0)
    hit |= _bump(state, "vitality", -abs(delta), 0.0, 1.0)
    # And her own exertion, which is the half of the body that is hers rather
    # than the machine's. An experiment that holds the host still to keep two
    # arms comparable holds still every host channel, so without this the body
    # has no channel left that an intervention can move.
    hit |= _bump(state, "soma.exertion", delta, 0.0, 1.0)
    return hit


def _perturb_A(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "affect.valence", delta, -1.0, 1.0)
    hit |= _bump(state, "affect.arousal", delta, 0.0, 1.0)
    hit |= _bump(state, "affect.curiosity", delta, 0.0, 1.0)
    # And the feeling itself. Ten of this domain's twenty-two columns are
    # emotion channels, and the workspace prices its affect bid by the
    # strongest of them — so a displacement that moved valence and arousal and
    # left the channels alone was a displacement of affect that affect's own
    # consumers could not see. A writer has to move what the reader reads.
    emotions = _state._dig(state, "affect.emotions", None)
    if isinstance(emotions, dict):
        for name in _state._EMOTIONS:
            if name in emotions:
                emotions[name] = min(1.0, max(0.0, _state._f(emotions[name]) + delta))
        hit = True
    return hit


def _perturb_G(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "cognition.conversation_energy", delta, 0.0, 1.0)
    hit |= _bump(state, "cognition.coherence_score", -abs(delta), 0.0, 1.0)
    node = getattr(state, "cognition", None)
    if node is not None:
        node.attention_focus = f"probe:{delta:+.4f}"
        hit = True
    return hit


def _perturb_C(state: Any, delta: float, ontogeny: Any) -> bool:
    """The recurrent estimate, and nothing else in the state.

    This also rewrote `cognition.phenomenal_state`'s valence, arousal and
    latent snapshot. That field is rebuilt once a turn from `affect.*`, the
    energy budget and the coherence score, and nothing in the runtime reads any
    of its numbers — so those writes were erased before anything could have
    used them and would have reached nobody if they had survived. Recurrent
    cognition's own state is the liquid substrate and the closed loop, and
    `perturb_organs` is where it is displaced.
    """
    del ontogeny
    return _bump(state, "phi_estimate", delta, -10.0, 10.0)


def _perturb_S(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = _bump(state, "identity.stability", -abs(delta), 0.0, 1.0)
    hit |= _bump(state, "identity.bonding_level", delta, 0.0, 1.0)
    hit |= _bump(state, "identity.evolution_score", delta, -10.0, 10.0)
    growth = _state._dig(state, "identity.personality_growth", None)
    if isinstance(growth, dict):
        growth["openness"] = _state._f(growth.get("openness")) + delta
        hit = True
    return hit


def _perturb_M(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    hit = False
    working = _state._dig(state, "cognition.working_memory", None)
    if isinstance(working, list):
        working.append(
            {
                "role": "probe",
                "content": f"subject-core memory probe {delta:+.4f}",
                "timestamp": time.time(),
            }
        )
        hit = True
    # And what is in mind. The retrieved set is what recall put there and what
    # the workspace bids a recollection from; displacing active memory without
    # touching it displaces the conversation buffer and calls it memory.
    cognition = getattr(state, "cognition", None)
    if cognition is not None:
        retrieved = list(getattr(cognition, "long_term_memory", []) or [])
        scores = list(getattr(cognition, "memory_scores", []) or [])
        retrieved.append(f"probe recollection {delta:+.4f}")
        # And how strongly it is in mind. Recall writes a match score beside
        # every recollection and the workspace prices its memory bid from it,
        # so a displacement that added text and no score displaced what was
        # recalled without displacing its claim on attention — which is the
        # half of active memory that anything downstream can act on.
        scores.append(min(1.0, max(0.0, 0.5 + delta)))
        cognition.long_term_memory = retrieved[-8:]
        cognition.memory_scores = scores[-8:]
        hit = True
    return hit


def _perturb_W(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    facts = _state._dig(state, "world.facts", None)
    if not isinstance(facts, dict):
        return False
    facts["subject_core_probe"] = {"delta": delta, "at": time.time()}
    # And the entities she takes to be there. A world model displaced only in a
    # scratch dict is displaced in the one part of the world state that nothing
    # consults; who and what is in the room is read by the workspace, the
    # social layer and the schema alike.
    entities = _state._dig(state, "world.known_entities", None)
    if isinstance(entities, dict):
        entities["subject_core_probe"] = {
            "kind": "probe",
            "salience": min(1.0, max(0.0, 0.5 + delta)),
            "at": time.time(),
        }
    return True


def _perturb_D(state: Any, delta: float, ontogeny: Any) -> bool:
    del ontogeny
    goals = _state._dig(state, "cognition.active_goals", None)
    hit = False
    if isinstance(goals, list):
        # In the shape the goal engine writes, `priority` included. A goal with
        # no stated priority is a goal the workspace cannot price, and an
        # intention nothing can attend to is not an intention.
        goals.append(
            {
                "id": "subject_core_probe",
                "goal": f"probe intention {delta:+.4f}",
                "description": f"probe intention {delta:+.4f}",
                "origin": "probe",
                "status": "pending",
                "priority": min(1.0, max(0.0, 0.5 + delta)),
                # And what it is asking to be thought about now. The workspace
                # prices deliberation's bid on `urgency` and drops a bid that
                # states none, so a probe goal carrying only a priority was a
                # displacement of deliberation that attention could not see.
                "urgency": min(1.0, max(0.0, 0.5 + delta)),
            }
        )
        hit = True
    budgets = _state._dig(state, "motivation.budgets", None)
    if isinstance(budgets, dict):
        for name in _state._DRIVES:
            entry = budgets.get(name)
            if not isinstance(entry, dict):
                continue
            # In units of the budget's own capacity. A drive level runs 0..100,
            # so adding the raw displacement moved it by fifteen hundredths of
            # one percent — a displacement of deliberation that deliberation
            # could not have noticed, in the same way the body's was a
            # displacement of one percent of a CPU.
            span = _state._f(entry.get("capacity"), 100.0) or 100.0
            for key in ("current", "level"):
                if key in entry:
                    # Toward capacity, by a share of what is missing, rather
                    # than by the same amount everywhere. What deliberation
                    # decides on is which drive is most depleted, and adding
                    # one number to every drive leaves that comparison exactly
                    # as it was: the intention generator dispatched on the same
                    # name in both arms, so the displaced arm did the same
                    # thing as the sham and deliberation reached nothing
                    # through action. Meeting the most pressing need most is
                    # what a displacement of motivation is.
                    level = _state._f(entry[key])
                    room = max(0.0, span - level)
                    entry[key] = max(0.0, min(span, level + delta * room))
                    hit = True
    return hit


def _perturb_N(state: Any, delta: float, ontogeny: Any) -> bool:
    del state
    if ontogeny is None or not hasattr(ontogeny, "h"):
        return False
    hidden = np.asarray(ontogeny.h, dtype=np.float64)
    ontogeny.h = np.clip(hidden + delta, -1.0, 1.0)
    return True


_WRITERS: dict[str, Callable[[Any, float, Any], bool]] = {
    "P": _perturb_P,
    "I": _perturb_I,
    "A": _perturb_A,
    "G": _perturb_G,
    "C": _perturb_C,
    "S": _perturb_S,
    "M": _perturb_M,
    "W": _perturb_W,
    "D": _perturb_D,
    "N": _perturb_N,
}


def perturbable() -> tuple[str, ...]:
    """The domains an intervention can actually reach."""
    return tuple(key for key in _state.DOMAINS if key in _WRITERS)


def perturb(state: Any, domain: str, delta: float, *, ontogeny: Any = None) -> bool:
    """Displace one domain by delta. False means the write found nothing.

    A False is the useful answer: it says this domain is not writable in this
    runtime, which makes every edge out of it unmeasured rather than absent.
    """
    writer = _WRITERS.get(domain)
    if writer is None:
        return False
    return bool(writer(state, float(delta), ontogeny))


async def perturb_organs(
    organs: _state.Organs, domain: str, delta: float, *, state: Any = None
) -> bool:
    """Displace the part of a domain that lives in an organ rather than in state.

    Each write here goes through the organ's own public path — the substrate's
    gated update, the world model's observe, the self model's belief update —
    rather than reaching past it into an attribute. A perturbation that a
    subsystem's own authority would refuse is not a perturbation of that
    subsystem, and forcing it would measure a state the runtime can never
    reach.
    """
    hit = False
    if domain == "C" and organs.substrate is not None:
        try:
            # The recurrent state first, then the readouts on top of it.
            #
            # The five named psychological dimensions are five of five hundred
            # and twelve, and they are readouts: `get_substrate_affect` takes
            # x[0], x[1], x[2] and two means over the whole vector. Writing
            # only those moved recurrent cognition's own reading by ten
            # standard deviations — the clipping ceiling — and moved the state
            # the dynamics carry by almost nothing, which is why C could be
            # displaced hardest of all ten domains and reach a tenth of a
            # standard deviation anywhere else. An intervention on the readout
            # of a recurrent system is not an intervention on the system.
            #
            # `inject_stimulus` is the organ's own gated path into the state
            # vector, and it applies a tenth of what it is given, so the vector
            # carries ten times the displacement to land on `delta`. The
            # authority may constrain the weight, which makes the intervention
            # smaller and is a refusal the measurement has to live with.
            await organs.substrate.inject_stimulus(
                np.full(
                    int(getattr(getattr(organs.substrate, "config", None), "neuron_count", 512)),
                    delta * 10.0,
                    dtype=np.float64,
                ),
                weight=1.0,
            )
            # And the dimensions the named consumers read, which the state
            # injection moves too but which the blend and the modifiers take
            # from the readout rather than from x.
            reading = organs.substrate.get_substrate_affect() or {}
            await organs.substrate.update(
                delta_frustration=delta,
                delta_curiosity=delta,
                valence=min(1.0, max(-1.0, _state._f(reading.get("valence")) + delta)),
                arousal=min(1.0, max(0.0, _state._f(reading.get("arousal"), 0.5) + delta)),
                dominance=min(1.0, max(-1.0, _state._f(reading.get("dominance")) + delta)),
                source="subject_core_probe",
            )
            hit = True
        except Exception:  # noqa: BLE001 - a refused write is not a write
            hit = False
    elif domain == "G" and organs.workspace is not None:
        workspace = organs.workspace
        # Displacing the workspace means changing what wins, not nudging a
        # readout. Writing `ignition_level` moved the number the schema reads
        # and nothing downstream, because the consumers fire on a broadcast and
        # a broadcast comes from a competition. This enters a bid strong enough
        # to change the outcome, which is the workspace intervention the
        # specification asks for: perturb one workspace content.
        try:
            from core.consciousness.global_workspace import CognitiveCandidate, ContentType

            # The runner-up, raised until it wins. A displacement of attention
            # is a change in what wins the competition among what is actually
            # competing — and the first version submitted a candidate of its
            # own under a probe's name, which nothing downstream can interpret:
            # the action she takes is chosen by the source of what she is
            # attending to, and a source no action table knows falls through to
            # the same action the sham took. So displacing the workspace could
            # not change what she did, and the whole route from attention
            # through action to the world and back to perception was closed to
            # the one domain that should open it.
            # Read off what has already been submitted rather than rebuilding
            # the bids: building them reports the work of building them, and a
            # displacement that costs the body something the sham did not pay
            # would manufacture the very edge it is measuring.
            runner_up = None
            lead = 0.0
            pending = list(getattr(workspace, "_candidates", ()) or ())
            if len(pending) >= 2:
                ranked = sorted(
                    pending, key=lambda bid: bid.effective_priority, reverse=True
                )
                runner_up = ranked[1]
                lead = ranked[0].effective_priority
            if runner_up is not None:
                await workspace.submit(
                    CognitiveCandidate(
                        content=runner_up.content,
                        source=runner_up.source,
                        priority=min(1.0, max(0.0, lead + abs(delta))),
                        content_type=runner_up.content_type,
                        affect_weight=runner_up.affect_weight,
                    )
                )
            else:
                await workspace.submit(
                    CognitiveCandidate(
                        content=f"subject core probe {delta:+.4f}",
                        source="subject_core_probe",
                        priority=min(1.0, max(0.0, 0.5 + delta * 3.0)),
                        content_type=ContentType.META,
                        affect_weight=abs(delta),
                    )
                )
            hit = True
        except Exception:  # noqa: BLE001
            hit = False
    elif domain == "S" and organs.self_model is not None:
        try:
            await organs.self_model.update_belief(
                "subject_core_probe", round(delta, 4), note="displacement probe"
            )
            hit = True
        except Exception:  # noqa: BLE001
            hit = False
    elif domain == "I":
        # The effort ledger, which is where her own exertion is kept.
        #
        # `_perturb_I` writes `soma.exertion`, and the proprioceptive loop
        # derives that from the ledger at the top of every turn — so the write
        # lasted until the next turn began. Worse, the one consumer that prices
        # anything on how hard she has been working reads the ledger rather
        # than the readout, so the displacement reached it never. The host is
        # held still for the duration of a trial, which leaves this as the only
        # channel the body has, and it was going nowhere.
        #
        # A share of what she has already spent, not a number: being a fifth
        # more tired than you are is a displacement, and being a fifth of some
        # absolute quantity more tired is a fact about the units.
        try:
            from core.soma.effort import get_effort_ledger

            ledger = get_effort_ledger()
            spent = dict(ledger.peek())
            for kind, amount in spent.items():
                step = float(amount) * delta
                if abs(step) > 1e-9:
                    ledger.note(kind, step)
                    hit = True
        except Exception:  # noqa: BLE001 - an absent ledger is an absent ledger
            hit = False
    elif domain == "N":
        # The reservoir the cognitive cycle steps, when that is not the one the
        # state writer already moved. `start_organism` binds the two together,
        # so in an assembled organism this is the same object and displacing it
        # twice would double the intervention; before the organs are up they
        # are different, and displacing only the readout would move what N
        # reports without moving what novelty is computed from.
        try:
            from core.ontogeny.service import get_ontogeny

            reservoir = getattr(get_ontogeny(), "_state", None)
            if (
                reservoir is not None
                and hasattr(reservoir, "h")
                and reservoir is not organs.ontogeny
            ):
                reservoir.h = np.clip(
                    np.asarray(reservoir.h, dtype=np.float64) + delta, -1.0, 1.0
                )
                hit = True
        except Exception:  # noqa: BLE001 - an absent organ is an absent organ
            hit = False
    elif domain == "W" and organs.world_model is not None:
        # An observation vector, which is what the forward model takes. A dict
        # went in and was padded to zeros, so the displacement of the world
        # model was a call that changed nothing: this domain moved itself by
        # seventeen thousandths of a standard deviation while every other
        # domain moved itself by two to ten, and a domain that cannot be
        # displaced cannot be shown to influence anything.
        #
        # The displacement is applied to the situation the cycle actually
        # reports, so the model is shown a world slightly other than the one it
        # is in — which is what displacing a model of the world means.
        try:
            from core.world_model.observe_cycle import action_of, observation_of

            if state is None:
                raise ValueError("displacing the world model needs the situation")
            observation = observation_of(state) + float(delta)
            organs.world_model.observe(observation, action_of(state), learn=True)
            hit = True
        except Exception:  # noqa: BLE001
            hit = False
    return hit
