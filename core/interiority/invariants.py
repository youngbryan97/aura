"""core/interiority/invariants.py — structural facts that must stay true.

An invariant here sits next to the thing it protects and yields a
violation when the shape of the package stops matching what it claims.
These are not tests of behaviour; the proving harness does that. These
are checks that the *declarations* have not drifted — that every
mechanism still has a home, that no number has lost its reason, that a
faculty has not quietly acquired a way to read another agent's true
state.

The last one is the important one. The single defect running through
every prototype this package was written against is a mechanism handed
the answer it claims to infer, and it is invisible in a test suite
because the arithmetic is correct. It is visible here, as a rule about
what a faculty is allowed to touch.
"""

from __future__ import annotations

import inspect

from core.verify import Severity, Violation, invariant

_OWNER = "core/interiority/"


@invariant(
    "interiority.every_faculty_has_a_home",
    scope="interiority",
    severity=Severity.ERROR,
    description="A mechanism with no declared place in the runtime is a folder.",
    owner=_OWNER,
)
def _every_faculty_has_a_home():
    from core.interiority.faculties import load_all
    from core.interiority.faculty import registry
    from core.interiority.homes import HOMES

    load_all()
    for faculty in registry().all():
        home = HOMES.get(faculty.id)
        if home is None:
            yield Violation(
                subject=faculty.id,
                message="declares no organ it belongs to and no consumer it feeds",
                remedy="add an entry to core/interiority/homes.py, or delete it",
            )
        elif not home.feeds:
            yield Violation(
                subject=faculty.id,
                message="has a home but feeds nothing that existed before it",
                remedy="name the existing consumer and the quantity it moves",
            )


@invariant(
    "interiority.every_number_states_its_reason",
    scope="interiority",
    severity=Severity.ERROR,
    description="A coefficient with no origin is an opinion wearing a decimal point.",
    owner=_OWNER,
)
def _every_number_states_its_reason():
    from core.interiority.faculties import load_all
    from core.interiority.params import ParamKind, registry

    load_all()
    for param in registry().all():
        if len(param.basis.strip()) < 40:
            yield Violation(
                subject=param.name,
                message=f"basis is {len(param.basis)} characters, too thin to check",
                remedy="cite it, derive it, measure it, or mark it CALIBRATION",
            )
        if param.kind is ParamKind.CALIBRATION and param.sweep_range is None:
            yield Violation(
                subject=param.name,
                message="a guess with no declared plausible range",
                remedy="declare sweep_range so the ordering can be shown invariant",
            )


@invariant(
    "interiority.no_faculty_reads_ground_truth",
    scope="interiority",
    severity=Severity.ERROR,
    description=(
        "A faculty may read a posterior about another agent, never their "
        "actual state."
    ),
    owner=_OWNER,
)
def _no_faculty_reads_ground_truth():
    from core.interiority.faculties import load_all
    from core.interiority.faculty import registry

    load_all()
    # The reviewed prototypes all take the other agent's real affect as an
    # argument and return it. There is no such channel here, and these are
    # the shapes that would reintroduce one.
    forbidden = (
        ".true_state",
        ".actual_affect",
        ".ground_truth",
        "other.affect",
    )
    for faculty in registry().all():
        try:
            source = inspect.getsource(type(faculty))
        except (OSError, TypeError):
            continue
        for marker in forbidden:
            if marker in source:
                yield Violation(
                    subject=faculty.id,
                    message=(
                        f"reads {marker}, which would hand it the answer it "
                        "claims to infer"
                    ),
                    remedy="read the OtherEstimate posterior and its confidence",
                )


@invariant(
    "interiority.constraints_are_not_weights",
    scope="interiority",
    severity=Severity.ERROR,
    description="A value that a large enough number can buy is a price.",
    owner=_OWNER,
)
def _constraints_are_not_weights():
    from core.interiority.arbitration import arbitrate, permitted
    from core.interiority.effects import (
        ActionConstraint,
        ConstraintForce,
        Effects,
        SomaticMarker,
    )
    from core.interiority.faculty import Activation

    held = Activation(
        "invariant.probe",
        1.0,
        "protect",
        Effects(
            constraints=(
                ActionConstraint(
                    "probe_action", ConstraintForce.HARD, "held", "invariant.probe"
                ),
            )
        ),
    )
    # Something wanting the forbidden action as hard as anything can.
    wanting = Activation(
        "invariant.tempter",
        1.0,
        "approach",
        Effects(somatic=(SomaticMarker("probe_action", 1.0, "wants it"),)),
    )
    state = arbitrate([held, wanting], dt=0.1)
    kept, blocked = permitted(["probe_action", "other_action"], state)
    if "probe_action" in kept or "probe_action" not in blocked:
        yield Violation(
            subject="core/interiority/arbitration.py",
            message=(
                "a hard constraint did not remove the action from the set "
                "when something else wanted it at full strength"
            ),
            remedy="filter constraints before scoring, never add them to it",
        )


__all__ = []
