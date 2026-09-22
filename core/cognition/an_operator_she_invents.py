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
from collections.abc import Callable, Iterator, Sequence
from typing import Any

from core.cognition.operator_invention import Candidate, OperatorKernel

__all__ = [
    "how_far_the_last_search_reached",
    "how_it_has_gone",
    "note_how_it_went",
    "offer_inventing_an_operator",
    "the_kernel",
]

logger = logging.getLogger("Aura.AnOperatorSheInvents")

_KERNEL = OperatorKernel()


def the_kernel() -> OperatorKernel:
    return _KERNEL


def note_how_it_went(
    family: str, *, solved: bool, probes: Sequence[Any] = (),
    cases: Sequence[tuple[Any, Any]] = (),
) -> Any:
    """Tell the kernel what happened. Called from the answering path.

    Without this every residual is empty, every family is transient, and the
    kernel refuses everything for the right reason and the wrong cause.
    """
    return _KERNEL.attempt(family, solved=solved, probes=probes, cases=cases)


#: What the last enumeration covered. Written by the proposer, read by the
#: report, so the reach is measured on the walk that happened rather than
#: recomputed from what the parameters say it should have been.
_NOTE_THE_REACH: dict[str, int] = {}


def how_far_the_last_search_reached() -> dict[str, Any]:
    """The space the proposer had, and how much of it the last walk covered."""
    from core.cognition.how_far_the_search_reaches import how_far_it_reaches

    if not _NOTE_THE_REACH:
        return {"searched": False, "why": "no operator search has run this process"}
    said = how_far_it_reaches(
        deepest=int(_NOTE_THE_REACH.get("deepest", 3)),
        leaves=int(_NOTE_THE_REACH.get("leaves", 5)),
        would_examine=int(_NOTE_THE_REACH.get("would_examine", 0)),
        from_her_library=int(_NOTE_THE_REACH.get("from_her_library", 0)),
        examined=int(_NOTE_THE_REACH.get("examined", 0)),
        computed=int(_NOTE_THE_REACH.get("offered", 0)),
        offered=int(_NOTE_THE_REACH.get("offered", 0)),
    )
    said["searched"] = True
    said["reach"]["offered"] = int(_NOTE_THE_REACH.get("offered", 0))
    said["reach"]["walked"] = int(_NOTE_THE_REACH.get("examined", 0))
    return said


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
        run(probes[0])
        return True
    # not a failure: an operator that refuses its first probe is not arithmetic.
    except Exception:  # noqa: BLE001 - foreign code: anything else is not arithmetic
        return False


def _a_candidate_for(
    family: str, probes: Sequence[Any], *, how_many: int = 4000,
    deepest: int = 3, max_offered: int | None = 64,
) -> Iterator[Candidate]:
    """Terms to offer, shortest first, over the floor.

    The proposer here is enumeration, and that is the honest description: the
    kernel's job is to refuse, and what it refuses has to come from somewhere
    before a better proposer exists. What enumeration offers is filtered by the
    cheap probe above, so the kernel spends its refusals on candidates that at
    least compute.
    """
    from core.cognition.the_floor_she_stands_on import every_code, how_long
    from core.cognition.what_she_already_knows_how_to_say import (
        what_she_already_knows_how_to_say,
    )

    # Her own terms as leaves. `every_code` names `also` as the only channel
    # by which a long term becomes reachable, and this call passed `()` — so
    # the operator search walked the same 380 terms at depth three forever,
    # over a language that is computationally universal. The budget was never
    # the bottleneck; the horizon was, and a library is what moves it.
    hers = what_she_already_knows_how_to_say()
    # What the space is, beside what this walk covers. A search that reports
    # only what it examined implies it looked at what mattered: over the bare
    # floor at depth three there are 380 terms and the cap is four thousand,
    # so the walk is exhaustive and the horizon is the constraint. Her own
    # terms as leaves are the only thing that moves it.
    _NOTE_THE_REACH.clear()
    _NOTE_THE_REACH.update(
        {"leaves": 5 + len(hers), "would_examine": how_many,
         "from_her_library": len(hers), "deepest": deepest}
    )
    offered = 0
    examined = 0
    for at, body in enumerate(
        itertools.islice(
            every_code(
                deepest=deepest,
                variables=1,
                constants=(0, 1, 2),
                also=hers,
            ),
            how_many,
        )
    ):
        examined = at + 1
        _NOTE_THE_REACH["examined"] = examined
        if how_long(body) < 2 or not _computes_a_number(body, probes):
            continue
        offered += 1
        _NOTE_THE_REACH["offered"] = offered
        _NOTE_THE_REACH["examined"] = examined
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
        if max_offered is not None and offered >= max_offered:
            _NOTE_THE_REACH["examined"] = examined
            return
    _NOTE_THE_REACH["examined"] = examined


def offer_inventing_an_operator(
    *, solves: Callable[[Callable[..., Any], str], bool] | None = None
) -> None:
    """Put it in the registry, so the ranking can reach the kernel."""
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )

    def invent(situation: Any = None) -> str | None:
        stuck = [one for one in _KERNEL.residuals() if one.cases]
        if not stuck:
            return None
        residual = stuck[0]
        cases = residual.cases
        probes = [before for before, _ in cases] or list(residual.probes)

        def comparable(value: Any) -> Any:
            if isinstance(value, (tuple, list)):
                return tuple(comparable(part) for part in value)
            return value

        def it_solves(run: Callable[..., Any], family: str) -> bool:
            try:
                return bool(cases) and all(
                    run(before) == comparable(expected) for before, expected in cases
                )
            # not a failure: a candidate that refuses has not solved it, which is the verdict
            # this search is here to reach.
            except Exception:  # noqa: BLE001 - foreign code: a candidate that raises has not solved
                return False

        def judge(run: Callable[..., Any], family: str) -> bool:
            return it_solves(run, family) and (solves is None or solves(run, family))
        from core.cognition.library_compression import REFERENCE_COST
        from core.cognition.the_floor_she_stands_on import how_long

        # Depth three cannot express the binder and expression of x + x.
        # The examined-term budget bounds this walk; returning constants must
        # not consume a second offer cap before a correct candidate is reached.
        for candidate in _a_candidate_for(
            residual.family, probes, deepest=4, max_offered=None
        ):
            size = how_long(candidate.term)
            uses = len({repr(comparable(before)) for before, _ in cases})
            compression = uses * (size - REFERENCE_COST) - (size + REFERENCE_COST)
            verdict = _KERNEL.consider(
                candidate,
                family=residual.family,
                probes=probes,
                solves=judge,
                compression=compression,
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
