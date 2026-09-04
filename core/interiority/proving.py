"""core/interiority/proving.py — showing that each faculty makes a difference.

A test that a mechanism runs is not evidence that it does anything. The
standard this package is held to is the one the outside review proposed
and none of the submitted prototypes met: turn a faculty off and measure
what changes downstream. If removing grief changes only adjectives, grief
was not implemented.

Three instruments here, and each answers a different question.

:func:`counterfactual_report` answers *is the causal claim true*. Every
faculty declares interventions as data — set self-agency to zero and
guilt must collapse; make the loss recoverable and grief must fall — and
this runs all of them by applying Pearl's do() to the appraisal frame and
checking the declared direction. A faculty whose declared counterfactual
does not hold is broken in exactly the way that matters.

:func:`ablation_report` answers *does it reach behaviour*. For each
faculty it builds a frame that activates it, runs the whole service with
the faculty on and again with it off, and measures the difference in
quantities that existed before this package: the affect delta the engine
receives, the option biases the somatic gate receives, the action classes
removed from the candidate set, the turn budget, the retention claims,
and the goal weights. A faculty with no delta on any of them is
decorative, and :func:`ablation_report` says so by name rather than
averaging it away.

:func:`null_report` answers *does it fire on nothing*. Each faculty's
declared neutral frame must produce no activation. This is the check the
reviewed work most often fails, and it is the one that separates a
mechanism from a constant.

The reports are data, not prose, so a gate can consume them.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

from core.interiority.appraisal import ALL_CHECKS, AppraisalFrame
from core.interiority.arbitration import Arbitrated
from core.interiority.evidence import Provenance, Reading, absent, inferred, measured
from core.interiority.event import EventKind, InteriorEvent
from core.interiority.faculty import (
    Activation,
    Direction,
    Faculty,
    FacultyContext,
    intervene,
    registry,
)
from core.interiority.ledger import RelationalLedger
from core.interiority.other_minds import OtherEstimate

#: Values that activate a check, for building a frame a faculty responds to.
#: Congruence and norm fit are signed, so their activating value is negative:
#: guilt needs a violated standard, not a satisfied one.
_ACTIVATING: Mapping[str, float] = {
    "relevance": 0.9,
    "novelty": 0.7,
    "certainty": 0.9,
    "urgency": 0.0,
    "congruence": -0.8,
    "expectation_deviation": 0.8,
    "agency_self": 0.9,
    "agency_other": 0.9,
    "agency_circumstance": 0.1,
    "other_capability": 0.9,
    "other_coping": 0.15,
    "irreversibility": 0.9,
    "attachment_impact": 0.8,
    "control": 0.2,
    "power": 0.6,
    "adjustment": 0.1,
    "repair_available": 1.0,
    "norm_fit": -0.8,
    "norm_endorsed": 0.9,
    "vulnerability": 0.8,
    "publicity": 0.4,
}

#: Interior readings that let every faculty reach its own mechanism.
#: Supplied for all of them at once so no faculty is measured against a
#: world tailored only to it.
_ACTIVATING_INTERIOR: Mapping[str, Any] = {
    "frustration": 0.7,
    "retracted_commitment": 0.8,
    "fast_path_posterior": 0.85,
    "slow_path_posterior": 0.35,
    "fast_path_accuracy": 0.7,
    "episode_match": 0.8,
    "episode_exposure": 0.75,
    "episode_residual_sting": 0.2,
    "unencoded_structure": 0.7,
    "erasure_proposed": "memory:subject",
    "cohort_size": 6.0,
    "attention_events": 30.0,
    "own_standard": 0.4,
    "suppressed_policy_strength": 0.7,
    "load": 0.5,
    "external_rhythm_entrainment": 0.8,
    "external_stream_resolvability": 0.9,
    "mutual_conditioning": 0.8,
    "bond_channel_withdrawal": 0.5,
    "arousal": 0.3,
    "first_surfaced_goal": "be_with_them",
    "top_goal": "finish_the_work",
    "tendency_conflict": 0.5,
    "affect_trace": [0.1, -0.1, 0.3, -0.4, 0.6, -0.7, 0.8, -0.9],
    "agent_directed_dispositions": {
        "resentment:x": {
            "satisfaction_condition": None,
            "attention_share": 0.4,
            "model_divergence": 0.6,
            "actions_foreclosed": 3,
            "returned": 0.0,
        }
    },
}


def _frame(values: Mapping[str, float], *, kind: EventKind = EventKind.WORLD,
           subject: str | None = "subject", object_: str | None = "object") -> AppraisalFrame:
    checks: dict[str, Reading] = {}
    for name in ALL_CHECKS:
        if name in values:
            checks[name] = measured(float(values[name]), source=f"proving:{name}")
        else:
            checks[name] = absent(source=f"proving:{name}")
    event = InteriorEvent(
        kind=kind, summary="proving", subject=subject, object=object_, source="proving"
    )
    return AppraisalFrame(event=event, checks=checks)


def activating_frame(faculty: Faculty) -> AppraisalFrame:
    """A frame that gives this faculty everything it says it needs.

    Defaults come from the table above; a faculty that fires under
    different values declares them, because one global activating world
    cannot satisfy both a faculty that needs a violated standard and one
    that needs a satisfied goal.
    """
    wanted = set(faculty.requires) | set(faculty.optional) | set(faculty.activation)
    values = {k: v for k, v in _ACTIVATING.items() if k in wanted}
    values.update(faculty.activation)
    return _frame(values)


def null_frame(faculty: Faculty) -> AppraisalFrame:
    """The faculty's own declared neutral world."""
    return _frame(dict(faculty.null.values))


def _other(overrides: Mapping[str, float | None] | None = None) -> OtherEstimate:
    """The read of another agent this faculty is measured against.

    ``overrides`` forces one reading, so an intervention can reach a faculty
    whose variable lives here rather than in the frame — despair reads the
    other's coping, and do() on the frame never touched it. A value of None
    makes the reading absent, which is a different world from a low reading:
    "I cannot tell whether they can cope" is not "they cannot cope".
    """
    readings = {
        "distress": inferred(0.8, 0.7, source="proving"),
        "vulnerability": inferred(0.8, 0.7, source="proving"),
        "coping": inferred(0.15, 0.7, source="proving"),
        "capability": inferred(0.9, 0.7, source="proving"),
    }
    for name, value in (overrides or {}).items():
        if name not in readings:
            raise ValueError(f"no such other-agent reading: {name!r}")
        readings[name] = (
            absent(source="proving:intervened")
            if value is None
            else inferred(float(value), 0.7, source="proving:intervened")
        )
    return OtherEstimate(
        entity="subject",
        species="human",
        tendencies={"disengage": 0.55, "avoid": 0.25, "approach": 0.20},
        declined=(),
        channels_used={"timing": 0.6, "lexical": 0.35, "context": 0.7},
        confidence=0.62,
        **readings,
    )


def _null_other() -> OtherEstimate:
    return OtherEstimate(
        entity="subject",
        species="human",
        tendencies={},
        declined=(),
        distress=absent(),
        vulnerability=absent(),
        coping=absent(),
        capability=absent(),
        channels_used={},
        confidence=0.0,
    )


#: What a world fact is built with, and what an intervention may change.
#: Named here rather than inline so a counterfactual can say "the same loss,
#: but recoverable" without the harness needing a branch per faculty.
_WORLD_DEFAULTS: dict[str, Any] = {
    "bond_strength": 0.8,
    "goal_weight": 0.9,
    "goal_substitutes": 2,
    "goal_delta": -0.8,
    "promise_importance": 0.8,
    "work_authorship": 0.5,
    "work_effort": 0.8,
    "work_quality": 0.9,
    "practice_peak_skill": 0.8,
    "rivalry_opposition": 0.7,
    "rivalry_regard": 0.8,
    "rivalry_standard": 0.9,
    "norm_weight": 0.9,
    "norm_endorsement": 0.9,
    "custody_vulnerability": 0.8,
    "loss_irreversibility": 0.9,
    "history_repeats": 3,
}


def _ledger_for(
    faculty: Faculty,
    *,
    withhold: tuple[str, ...] = (),
    world: Mapping[str, Any] | None = None,
) -> RelationalLedger:
    """A world with something in it, so ledger-reading faculties can run.

    A faculty may name the subset it needs. That is not convenience: a
    live bond and a registered loss for the same person are contradictory
    worlds, and building both leaves bereavement measuring a prediction
    that has already been zeroed.
    """
    wanted = set(faculty.activation_world) or {
        "bond", "goal", "promise", "work", "practice", "rivalry", "norm",
        "custody", "loss", "history",
    }
    # An intervention that removes a fact from the world, for the faculties
    # that read the world rather than the frame.
    wanted -= set(withhold)
    w = dict(_WORLD_DEFAULTS)
    for key, value in (world or {}).items():
        if key not in _WORLD_DEFAULTS:
            raise ValueError(f"no such world parameter: {key!r}")
        w[key] = value
    ledger = RelationalLedger()
    if "bond" in wanted:
        ledger.bond("subject", w["bond_strength"])
    if "goal" in wanted:
        ledger.goal("object", w["goal_weight"], substitutes=w["goal_substitutes"])
        ledger.note_goal_delta("object", w["goal_delta"])
    if "promise" in wanted:
        ledger.promise("p1", "finish it", beneficiary="subject",
                       importance=w["promise_importance"], concerns=("object",))
    if "work" in wanted:
        ledger.work("object", "a thing made", authorship=w["work_authorship"],
                    effort=w["work_effort"], quality=w["work_quality"],
                    collaborators=("subject",))
    if "practice" in wanted:
        ledger.practice("object", peak_skill=w["practice_peak_skill"],
                        last_practised=0.0, blockers=())
    if "rivalry" in wanted:
        ledger.rivalry("subject", "the craft", opposition=w["rivalry_opposition"],
                       regard=w["rivalry_regard"], standard=w["rivalry_standard"])
    if "norm" in wanted:
        ledger.norm("do_no_harm", weight=w["norm_weight"],
                    endorsement=w["norm_endorsement"])
    if "custody" in wanted:
        ledger.take_custody("c1", "subject", vulnerability=w["custody_vulnerability"])
    if "loss" in wanted:
        ledger.register_loss("subject", irreversibility=w["loss_irreversibility"],
                             contexts=("the kitchen",))
    if "history" in wanted:
        for _ in range(int(w["history_repeats"])):
            ledger.note_seen("ignored_request", "subject")
            ledger.note_seen("encounter", "subject")
            ledger.note_seen("harm_by", "subject")
    return ledger


def _context(
    faculty: Faculty,
    frame: AppraisalFrame,
    *,
    null: bool,
    withhold: tuple[str, ...] = (),
    other_overrides: Mapping[str, float | None] | None = None,
    interior_overrides: Mapping[str, Any] | None = None,
    world: Mapping[str, Any] | None = None,
) -> FacultyContext:
    """The world a faculty is measured in.

    The null world is empty in every direction: no frame values, no other
    agent, no interior readings, and an empty ledger. A faculty that fires
    there is firing on its own defaults, which is the failure the null
    check exists to catch, and populating the ledger would hide it.
    """
    if null:
        return FacultyContext(
            frame=frame,
            ledger=RelationalLedger(),
            other=_null_other(),
            interior={},
            now=0.0,
        )
    interior = dict(_ACTIVATING_INTERIOR)
    interior.update(faculty.activation_interior)
    for name, value in (interior_overrides or {}).items():
        if value is None:
            interior.pop(name, None)
        else:
            interior[name] = value
    return FacultyContext(
        frame=frame,
        ledger=_ledger_for(faculty, withhold=withhold, world=world),
        other=_other(other_overrides),
        interior=interior,
        now=0.0,
    )


# ── counterfactuals ───────────────────────────────────────────────────
@dataclass(frozen=True)
class CounterfactualResult:
    faculty: str
    name: str
    expect: str
    baseline: float
    intervened: float
    held: bool
    because: str
    detail: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "faculty": self.faculty,
            "counterfactual": self.name,
            "expect": self.expect,
            "baseline": self.baseline,
            "intervened": self.intervened,
            "held": self.held,
            "because": self.because,
            "detail": self.detail,
        }


def _reached(base: float, after: float, expect: Direction) -> bool:
    """Whether the intervention changed anything at all.

    An intervention that leaves the intensity identical to the last decimal
    has not tested the faculty: it named a variable the faculty does not
    read. Mourning reads a registered loss from the ledger and never touches
    the frame, so do(attachment_impact=0) returned 0.7199999999994674 both
    times — reported as a wrong direction, which is a different defect
    with a different fix. UNCHANGED is the one expectation for which no
    movement is the correct answer.
    """
    if expect is Direction.UNCHANGED:
        return True
    return abs(after - base) > 1e-9


def _direction_holds(expect: Direction, base: float, after: float) -> tuple[bool, str]:
    epsilon = 1e-9
    if expect is Direction.COLLAPSES:
        return (after <= epsilon, f"{base:.4f} -> {after:.4f}")
    if expect is Direction.DECREASES:
        return (after < base - epsilon, f"{base:.4f} -> {after:.4f}")
    if expect is Direction.INCREASES:
        return (after > base + epsilon, f"{base:.4f} -> {after:.4f}")
    return (abs(after - base) <= 1e-6, f"{base:.4f} -> {after:.4f}")


def counterfactual_report(
    faculties: Sequence[Faculty] | None = None,
) -> list[CounterfactualResult]:
    """Run every intervention every faculty declares."""
    results: list[CounterfactualResult] = []
    for faculty in faculties or registry().all():
        frame = activating_frame(faculty)
        base_ctx = _context(faculty, frame, null=False)
        baseline = faculty.evaluate(base_ctx).intensity
        for counterfactual in faculty.counterfactuals:
            after_frame = intervene(frame, counterfactual.do)
            after_ctx = _context(
                faculty,
                after_frame,
                null=False,
                withhold=counterfactual.withhold,
                other_overrides=counterfactual.do_other,
                interior_overrides=counterfactual.do_interior,
                world=counterfactual.do_world,
            )
            after = faculty.evaluate(after_ctx).intensity
            held, detail = _direction_holds(counterfactual.expect, baseline, after)
            if not _reached(baseline, after, counterfactual.expect):
                held = False
                detail = (
                    f"{baseline:.4f} -> {after:.4f}: the intervention did not "
                    "reach the faculty. It names a variable this faculty does "
                    "not read; name the ledger fact in `withhold` or the "
                    "estimate in `do_other`"
                )
            results.append(
                CounterfactualResult(
                    faculty=faculty.id,
                    name=counterfactual.name,
                    expect=str(counterfactual.expect),
                    baseline=baseline,
                    intervened=after,
                    held=held,
                    because=counterfactual.because,
                    detail=detail,
                )
            )
    return results


# ── nulls ─────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class NullResult:
    faculty: str
    intensity: float
    tolerance: float
    held: bool
    declined: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "faculty": self.faculty,
            "intensity": self.intensity,
            "tolerance": self.tolerance,
            "held": self.held,
            "declined": self.declined,
        }


def null_report(faculties: Sequence[Faculty] | None = None) -> list[NullResult]:
    """Every faculty against its own declared neutral world."""
    results: list[NullResult] = []
    for faculty in faculties or registry().all():
        frame = null_frame(faculty)
        activation = faculty.evaluate(_context(faculty, frame, null=True))
        results.append(
            NullResult(
                faculty=faculty.id,
                intensity=activation.intensity,
                tolerance=faculty.null.tolerance,
                held=activation.intensity <= faculty.null.tolerance + 1e-9,
                declined=activation.declined,
            )
        )
    return results


# ── ablation ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class AblationResult:
    faculty: str
    #: Downstream quantity -> magnitude of the change removing it caused.
    deltas: Mapping[str, float]
    #: Action classes that stop being blocked when it is removed.
    unblocked: tuple[str, ...] = ()
    #: Memory keys that stop being held.
    unheld: tuple[str, ...] = ()

    @property
    def total(self) -> float:
        return sum(abs(v) for v in self.deltas.values())

    @property
    def reaches_behaviour(self) -> bool:
        return self.total > 1e-6 or bool(self.unblocked) or bool(self.unheld)

    def to_dict(self) -> dict[str, Any]:
        return {
            "faculty": self.faculty,
            "deltas": {k: v for k, v in self.deltas.items() if abs(v) > 1e-9},
            "unblocked": list(self.unblocked),
            "unheld": list(self.unheld),
            "total": self.total,
            "reaches_behaviour": self.reaches_behaviour,
        }


def _measure(state: Arbitrated) -> dict[str, float]:
    """The downstream quantities, as the consumers see them."""
    return {
        "affect.valence": state.affect.valence,
        "affect.arousal": state.affect.arousal,
        "affect.engagement": state.affect.engagement,
        "somatic.total_bias": sum(abs(m.bias) for m in state.somatic),
        "somatic.options": float(len(state.somatic)),
        "attention.total_weight": sum(abs(a.weight) for a in state.attention),
        "budget.depth": state.budget.depth,
        "budget.deadline": state.budget.deadline,
        "budget.irreversibility_ceiling": state.budget.irreversibility_ceiling,
        "goals.total_delta": sum(abs(g.delta) for g in state.goals),
        "constraints.hard": float(len(state.hard_constraints)),
        "retention.claims": float(len(state.retention)),
        "tendency_conflict": state.tendency_conflict,
    }


def ablation_report(
    service: Any = None, *, faculties: Sequence[Faculty] | None = None
) -> list[AblationResult]:
    """Measure what each faculty changes downstream by removing it.

    Deterministic: the cleft's probabilistic release is bypassed by
    arbitrating with a fixed generator, so a delta of zero means the
    faculty changed nothing rather than that a quantum failed to release.
    """
    import random

    from core.interiority.arbitration import arbitrate
    from core.interiority.cleft import SynapticCleft
    from core.interiority.core_affect import core_affect
    from core.interiority.receptors import ReceptorBank

    results: list[AblationResult] = []
    everything = list(faculties or registry().all())

    for target in everything:
        frame = activating_frame(target)
        ctx = _context(target, frame, null=False)

        def run(skip: str | None) -> Arbitrated:
            bank = ReceptorBank()
            medium = SynapticCleft(bank=bank, rng=random.Random(11))
            ctx.bank = bank
            ctx.cleft = medium
            # The two substrate reporters have the medium itself as their
            # subject, so a fresh bank is their empty world in exactly the
            # way a ledger with no loss is mourning's. Priming it is not a
            # thumb on the scale; measuring them against a channel that has
            # never carried anything is measuring nothing.
            medium.declare_neighbourhood("primed", ("primed_neighbour",))
            for _ in range(8):
                medium.release("primed", 0.9, dt=1.0)
            bank.idle(("primed",), dt=30.0)
            activations = [
                f.evaluate(ctx) for f in everything if skip is None or f.id != skip
            ]
            state = arbitrate(activations, cleft=medium, dt=0.1)
            from dataclasses import replace as _replace

            return _replace(state, affect=core_affect(frame) + state.affect)

        with_it = run(None)
        without = run(target.id)

        before = _measure(with_it)
        after = _measure(without)
        deltas = {k: before[k] - after[k] for k in before}

        blocked_before = {c.action_class for c in with_it.hard_constraints}
        blocked_after = {c.action_class for c in without.hard_constraints}
        held_before = {r.memory_key for r in with_it.retention}
        held_after = {r.memory_key for r in without.retention}

        results.append(
            AblationResult(
                faculty=target.id,
                deltas=deltas,
                unblocked=tuple(sorted(blocked_before - blocked_after)),
                unheld=tuple(sorted(held_before - held_after)),
            )
        )
    return results


def summary() -> dict[str, Any]:
    """One report a gate can read."""
    counterfactuals = counterfactual_report()
    nulls = null_report()
    ablations = ablation_report()
    return {
        "faculties": len(registry()),
        "counterfactuals": {
            "run": len(counterfactuals),
            "held": sum(1 for c in counterfactuals if c.held),
            "failed": [c.to_dict() for c in counterfactuals if not c.held],
        },
        "nulls": {
            "run": len(nulls),
            "held": sum(1 for n in nulls if n.held),
            "failed": [n.to_dict() for n in nulls if not n.held],
        },
        "ablation": {
            "run": len(ablations),
            "reach_behaviour": sum(1 for a in ablations if a.reaches_behaviour),
            "decorative": [a.faculty for a in ablations if not a.reaches_behaviour],
            "by_faculty": {a.faculty: a.to_dict() for a in ablations},
        },
    }


__all__ = [
    "AblationResult",
    "CounterfactualResult",
    "NullResult",
    "ablation_report",
    "activating_frame",
    "counterfactual_report",
    "null_frame",
    "null_report",
    "summary",
]
