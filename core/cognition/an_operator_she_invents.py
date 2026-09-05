"""The operator kernel, given a caller at last.

`operator_invention` has been in this tree with a complete gate and nobody to
open it. Its seven refusals are real — not persistent, computes nothing, raised,
no compression, fails an adversarial probe — and every one of them was reachable
only from a test. A gate with no caller is a claim about what would happen if
something ever asked, and this is the something.

What asks is the ranking. Inventing an operator is a developmental action like
any other: it is priced off what operators have saved before, it competes with
widening the language and with changing the order, and it runs only when the
record says it is worth the candidates. What it is NOT is a rung that fires
because something else returned nothing.

The residual is the evidence. A family is a candidate for an operator when it
has been attempted enough times and solved none of them, and that count comes
from the answering path rather than from a test — so a family that fails once
is not persistent and the kernel says so.
"""

from __future__ import annotations

import itertools
import logging
from typing import Any, Callable, Sequence

from core.cognition.operator_invention import Candidate, OperatorKernel

__all__ = [
    "how_it_has_gone",
    "note_how_it_went",
    "offer_inventing_an_operator",
    "the_kernel",
]

logger = logging.getLogger("Aura.AnOperatorSheInvents")

_KERNEL = OperatorKernel()


def the_kernel() -> OperatorKernel:
    return _KERNEL


def note_how_it_went(family: str, *, solved: bool, probes: Sequence[Any] = ()) -> Any:
    """Tell the kernel what happened. Called from the answering path.

    Without this every residual is empty, every family is transient, and the
    kernel refuses everything for the right reason and the wrong cause.
    """
    return _KERNEL.attempt(family, solved=solved, probes=probes)


def how_it_has_gone() -> list[dict[str, Any]]:
    """Every family the kernel is watching, and whether it has stuck.

    All of them, not only the persistent ones. `residuals()` returns what is
    ripe for an operator, which is the right thing for the kernel to offer and
    the wrong thing for a diagnostic: a reading that shows nothing until
    something is already wrong cannot be used to see that nothing is.
    """
    return [
        {
            "family": one.family,
            "attempts": one.attempts,
            "solved": one.solved,
            "persistent": one.persistent,
        }
        for one in _KERNEL._residuals.values()  # noqa: PLC2701, SLF001
    ]


def _computes_a_number(body: Any, probes: Sequence[Any]) -> bool:
    """The cheap probe: does this term give a number at all?

    Run once, on one probe, with a small fuel. Two hundred candidates offered
    to the kernel were all refused as "raised", which is the gate working and
    the proposer wasting it: most short terms over two variables are not
    arithmetic and cannot be. Asking the cheap question first leaves the dear
    machinery — bounded execution on every probe, compression, adversarial
    behaviour — for candidates that could pass it.
    """
    from core.cognition.operator_invention import (
        _the_term_as_a_function,  # noqa: PLC2701
    )

    if not probes:
        return False
    run = _the_term_as_a_function(body)
    try:
        return isinstance(run(probes[0]), int)
    except Exception:  # noqa: BLE001 - anything else is not arithmetic
        return False


def _a_candidate_for(family: str, probes: Sequence[Any], *, how_many: int = 4000):
    """Terms to offer, shortest first, over the floor.

    The proposer here is enumeration, and that is the honest description: the
    kernel's job is to refuse, and what it refuses has to come from somewhere
    before a better proposer exists. What enumeration offers is filtered by the
    cheap probe above, so the kernel spends its refusals on candidates that at
    least compute.
    """
    from core.cognition.the_floor_she_stands_on import every_code, how_long

    offered = 0
    for at, body in enumerate(
        itertools.islice(
            every_code(deepest=3, variables=1, constants=(0, 1, 2), also=()),
            how_many,
        )
    ):
        if how_long(body) < 2 or not _computes_a_number(body, probes):
            continue
        offered += 1
        yield Candidate(
            name=f"an operator for {family} ({at})",
            body=f"a term of {how_long(body)} symbols",
            # One, because that is what the kernel's wrapper builds: a term
            # applied to a single value. Offering a two-place candidate to a
            # one-place caller is how two hundred perfectly good terms came
            # back as "raised".
            arity=1,
            term=body,
        )
        if offered >= 64:
            return


def offer_inventing_an_operator(
    *, solves: Callable[[Callable[..., Any], str], bool] | None = None
) -> None:
    """Put it in the registry, so the ranking can reach the kernel."""
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )

    def invent(situation: Any = None) -> str | None:
        stuck = [one for one in _KERNEL.residuals() if one.persistent]
        if not stuck:
            return None
        residual = stuck[0]
        probes = list(residual.probes) or [1, 2, 3, 5, 8]

        def it_solves(run: Callable[..., Any], family: str) -> bool:
            # Solving means answering every probe without raising, and the
            # kernel checks compression and adversarial behaviour after this.
            try:
                return all(run(one) is not None for one in probes)
            except Exception:  # noqa: BLE001 - a candidate that raises has not solved
                return False

        judge = solves or it_solves
        for candidate in _a_candidate_for(residual.family, probes):
            verdict = _KERNEL.consider(
                candidate,
                family=residual.family,
                probes=probes,
                solves=judge,
                compression=1,
            )
            if verdict.installed:
                logger.info(
                    "the kernel accepted %s for %s", candidate.name, residual.family
                )
                return f"invented {candidate.name}"
        return None

    if "invent an operator for what keeps failing" not in WHAT_SHE_COULD_DO:
        what_she_could_do(
            "invent an operator for what keeps failing",
            over="the words",
            kind="an operator",
            do_it=invent,
            needs_a_case=False,
        )
