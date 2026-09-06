"""core/runtime/the_shape_of_one_turn.py — the cognition order, compiled and sealed.

A reviewer comparing Aura against Generative Agents, LangGraph and Soar made
the same complaint from three directions, and it is the same complaint each
time: the order in which Aura thinks is real, and it cannot be read off
anything. Generative Agents puts the whole cycle in one function. Soar's
decision cycle is a state machine anyone can enumerate. LangGraph compiles its
graph before it runs and refuses to start when the topology does not resolve.

Aura already has more than any of them and less than all of them. The pieces:

* ``pipeline_blueprint`` holds the order and which phases a priority tick
  suppresses — so the sequence and the frequency are both declared.
* ``cognitive_contract`` holds what each phase reads, writes, needs authority
  for and does outside state — eleven of twenty-nine phases so far.
* ``pass_manager`` records what actually ran, in order, per turn.

What was missing is the join. Nothing put the three together, checked that the
result is coherent, sealed it, or compared what was declared against what ran.
So a phase could read a field nothing writes, two phases could write the same
field with no rule for combining them, and the only way to find out was to
watch it happen.

This compiles it. The plan is a value: an order, per mode, with each phase's
reads and writes, and a seal over the whole thing. A plan that does not resolve
is a refusal with the reasons attached rather than a runtime that starts and
finds out later.

The seal is the point of the exercise as much as the check is. Two runs of the
same commit produce the same seal; a phase added, removed, reordered or
re-declared produces a different one. A receipt carrying a seal says which
cognition produced it.
"""

from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("Aura.TheShapeOfOneTurn")

__all__ = [
    "APhaseInThePlan",
    "LAST_IN_THE_ORDER",
    "THE_WRITE_MODES",
    "declare_write_mode",
    "write_mode_for",
    "TheShapeOfOneTurn",
    "compile_the_cognition",
    "the_declared_and_the_realised",
    "the_seal",
]

#: Modes a turn can run in. A priority tick suppresses the background-only
#: phases; a degraded one is the foreground set with whatever is still up.
THE_MODES: tuple[str, ...] = ("foreground", "background", "degraded")

#: The default way several writers of one field are combined: the phases run
#: one after another in a declared order, so the last one to write is the
#: value. This is a claim about the pipeline being sequential, and it is why
#: it is written down rather than assumed — the day two phases run together it
#: stops being true and the field needs a real reducer.
LAST_IN_THE_ORDER = "last in the order"

#: Ways of combining concurrent writes this build knows. A field declared with
#: anything else is a refusal rather than a guess.
THE_WRITE_MODES: frozenset[str] = frozenset(
    {LAST_IN_THE_ORDER, "single writer", "highest", "lowest", "union", "sum"}
)

#: Fields whose several writers are combined by something other than order.
#: Empty is the honest starting state: every shared field in this pipeline is
#: settled by the order today, and this is where a field goes when that stops
#: being the right answer for it.
_HOW_A_FIELD_IS_COMBINED: dict[str, str] = {}


def declare_write_mode(path: str, mode: str) -> None:
    """Say how several writers of one field are combined.

    LangGraph declares this per key in the state schema and rejects
    unspecified concurrent writes, which is the right shape: the alternative
    is that the answer depends on which phase happened to run last, and
    nothing records that it does.
    """

    if mode not in THE_WRITE_MODES:
        raise ValueError(f"no such write mode: {mode!r}")
    _HOW_A_FIELD_IS_COMBINED[str(path)] = mode


def write_mode_for(path: str) -> str | None:
    """How this field's writers combine, or None when it says something unknown."""

    mode = _HOW_A_FIELD_IS_COMBINED.get(str(path), LAST_IN_THE_ORDER)
    return mode if mode in THE_WRITE_MODES else None


@dataclass(frozen=True)
class APhaseInThePlan:
    """One phase, as the plan sees it."""

    order: int
    attribute: str
    phase: str
    #: Whether this phase runs in the mode the plan was compiled for.
    runs: bool
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ()
    authority: str = ""
    side_effects: tuple[str, ...] = ()
    #: True when nothing declares what this phase reads or writes. Not an
    #: error — eighteen of twenty-nine are like this — but the plan says so
    #: rather than reporting an empty set as if it were a claim.
    undeclared: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "order": self.order,
            "attribute": self.attribute,
            "phase": self.phase,
            "runs": self.runs,
            "reads": list(self.reads),
            "writes": list(self.writes),
            "authority": self.authority,
            "side_effects": list(self.side_effects),
            "undeclared": self.undeclared,
        }


@dataclass(frozen=True)
class TheShapeOfOneTurn:
    """The compiled plan for one mode, and whether it holds together."""

    mode: str
    phases: tuple[APhaseInThePlan, ...]
    #: Every reason the plan does not hold, in the order they were found. A
    #: plan with any of these must not be run.
    refusals: tuple[str, ...] = ()
    #: Things worth saying that do not stop it running.
    remarks: tuple[str, ...] = ()
    seal: str = ""

    @property
    def holds(self) -> bool:
        return not self.refusals

    @property
    def runs(self) -> tuple[str, ...]:
        return tuple(one.phase for one in self.phases if one.runs)

    def to_dict(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "seal": self.seal,
            "holds": self.holds,
            "phases": [one.to_dict() for one in self.phases],
            "runs": list(self.runs),
            "refusals": list(self.refusals),
            "remarks": list(self.remarks),
            "declared": sum(1 for one in self.phases if not one.undeclared),
        }


def _contracts() -> dict[str, Any]:
    try:
        from core.runtime.cognitive_contract import all_contracts

        return dict(all_contracts())
    except (ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("no contracts to compile against: %s", exc)
        return {}


def _order_and_frequency() -> tuple[tuple[str, ...], frozenset[str], dict[str, str]]:
    from core.runtime.pipeline_blueprint import (
        foreground_phase_attributes,
        kernel_phase_attribute_order,
        phase_class_for_attribute,
    )

    order = tuple(kernel_phase_attribute_order())
    foreground = frozenset(foreground_phase_attributes())
    classes = {one: phase_class_for_attribute(one) for one in order}
    return order, foreground, classes


def compile_the_cognition(mode: str = "foreground") -> TheShapeOfOneTurn:
    """The order for this mode, checked, with a seal over the result.

    Four things stop a plan. An attribute appearing twice is an order that does
    not have one answer. An attribute with no phase class is a name that
    resolves to nothing. A field two running phases both write, with nothing
    saying how to combine them, is a race the order does not settle. And a
    field a running phase reads that no earlier running phase writes is a read
    of whatever happened to be there — reported as a remark rather than a
    refusal, because the world writes some of these and the contracts do not
    say which.
    """

    if mode not in THE_MODES:
        return TheShapeOfOneTurn(
            mode=mode, phases=(), refusals=(f"no such mode: {mode!r}",)
        )
    order, foreground, classes = _order_and_frequency()
    contracts = _contracts()
    refusals: list[str] = []
    remarks: list[str] = []

    seen: set[str] = set()
    phases: list[APhaseInThePlan] = []
    for at, attribute in enumerate(order):
        if attribute in seen:
            refusals.append(f"{attribute} appears twice in the order")
            continue
        seen.add(attribute)
        phase = classes.get(attribute) or ""
        if not phase:
            refusals.append(f"{attribute} names no phase class")
            continue
        contract = contracts.get(phase)
        phases.append(
            APhaseInThePlan(
                order=at,
                attribute=attribute,
                phase=phase,
                # Background runs everything. Foreground and degraded run the
                # set a priority tick does not suppress — they differ in what
                # is allowed to fail, not in what is scheduled, so they share a
                # membership and will diverge when degradation is declared per
                # phase rather than per subsystem.
                runs=(mode == "background") or (attribute in foreground),
                reads=tuple(getattr(contract, "reads", ()) or ()),
                writes=tuple(getattr(contract, "writes", ()) or ()),
                authority=str(getattr(contract, "authority", "") or ""),
                side_effects=tuple(getattr(contract, "side_effects", ()) or ()),
                undeclared=contract is None,
            )
        )

    running = [one for one in phases if one.runs]
    written_by: dict[str, list[str]] = {}
    for one in running:
        for path in one.writes:
            written_by.setdefault(path, []).append(one.phase)
    for path, writers in sorted(written_by.items()):
        if len(writers) < 2:
            continue
        mode = write_mode_for(path)
        if mode == LAST_IN_THE_ORDER:
            # Several writers in a sequence that has one order is not a race:
            # the order settles it, and the last writer is the value. Said out
            # loud anyway, because "the order settles it" is a claim about the
            # phases running one after another, and the day any of them run
            # together it stops being true.
            remarks.append(
                f"{path} is written by {', '.join(writers)}; "
                f"the order settles it and {writers[-1]} is the value"
            )
            continue
        if mode is None:
            refusals.append(
                f"{path} is written by {len(writers)} running phases "
                f"({', '.join(sorted(writers))}) and its write mode is declared "
                f"as something this build does not know"
            )
            continue
        remarks.append(
            f"{path} is written by {', '.join(writers)} and combined by {mode}"
        )

    produced: set[str] = set()
    for one in running:
        for path in one.reads:
            if path not in produced and path not in written_by:
                remarks.append(f"{one.phase} reads {path}, which no phase in this plan writes")
        produced.update(one.writes)

    made = TheShapeOfOneTurn(
        mode=mode,
        phases=tuple(phases),
        refusals=tuple(refusals),
        remarks=tuple(remarks),
    )
    return TheShapeOfOneTurn(
        mode=made.mode,
        phases=made.phases,
        refusals=made.refusals,
        remarks=made.remarks,
        seal=the_seal(made),
    )


def the_seal(plan: TheShapeOfOneTurn) -> str:
    """A digest of the plan, so a receipt can say which cognition made it.

    Over the order, the phases, what they declare and whether they run. Not
    over the remarks, which are about the plan rather than part of it, and not
    over the refusals, because a refused plan does not get a name.
    """

    body = json.dumps(
        [
            [one.order, one.attribute, one.phase, one.runs, list(one.reads), list(one.writes)]
            for one in plan.phases
        ],
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(f"{plan.mode}\n{body}".encode()).hexdigest()[:16]


def the_declared_and_the_realised(mode: str = "foreground") -> dict[str, Any]:
    """What the plan says will run, beside what actually did.

    The finding asks for the realised order per turn, and the pass manager has
    been recording it the whole time — what was missing is anything comparing
    the two. A phase that runs and is not in the plan is the more interesting
    direction: it means cognition is happening somewhere the order does not
    mention.
    """

    plan = compile_the_cognition(mode)
    ran: list[str] = []
    try:
        from core.pipeline.pass_manager import get_instrumentation

        report = get_instrumentation().report()
        for record in report.get("recent") or ():
            name = str(record.get("name") or "")
            # Records are prefixed by which loop produced them.
            ran.append(name.split("/")[-1])
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError) as exc:
        logger.debug("no realised order to compare: %s", exc)
    declared = {one.attribute for one in plan.phases if one.runs} | set(plan.runs)
    unplanned = sorted({one for one in ran if one and one not in declared})
    return {
        "mode": mode,
        "seal": plan.seal,
        "holds": plan.holds,
        "declared": list(plan.runs),
        "realised": ran[-24:],
        "ran_outside_the_plan": unplanned,
        "refusals": list(plan.refusals),
        "remarks": list(plan.remarks)[:8],
    }
