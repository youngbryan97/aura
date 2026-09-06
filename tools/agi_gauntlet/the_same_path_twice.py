"""Gate 16 at runtime: the same machinery, watched, on two different families.

Gate 16 reads the source and refuses any path keyed on an evaluation —

    if benchmark == "ARC": use_arc_solver()

— and says of itself that the check is a grep, which is weak and cannot be
argued with. It is also static. A branch keyed on the *shape* of a problem
rather than on its name would pass it and still be a bag of solvers.

So watch instead. Run two problems from materially different families through
one entry point, record every function under ``core`` that actually executes,
and compare. What matters is not that the sets are identical — a hypothesis
search that finds different rules SHOULD run different rule code — but that
the difference is confined to the search's own members. A module that runs for
one family and never for the other, outside the hypothesis space, is a solver
selected by problem shape.

Deterministic and offline: no model, no network, and the tracer is the only
thing that touches the run.
"""
from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Any, Callable

__all__ = [
    "THE_CONSEQUENCES",
    "THE_SEARCH",
    "score_again",
    "the_same_path_twice",
    "what_ran_during",
]

ROOT = Path(__file__).resolve().parents[2]

#: Where the search lives. Everything under it is the hypothesis space: the
#: rules she can propose, the language she extends to state them, the ordering
#: that decides which to try, and the developmental step a new result sets
#: off. A difference here is the search working — finding a positional rule
#: runs positional code — and counting that as a solver selected by problem
#: shape would make the probe unpassable for any system that actually
#: searches.
#:
#: A package rather than a list of modules, deliberately. Enumerating whatever
#: happened to run makes the probe unfalsifiable, which is the failure mode of
#: every gate that grades itself. This stays falsifiable because the shape
#: gate 16 refuses — a solver picked by the name or shape of an evaluation —
#: would be a skill, a route, or a named module reached from outside, and
#: every one of those is outside this package.
THE_SEARCH: str = "core/cognition/"

#: Machinery a result can legitimately reach on one path and not another,
#: because the result differed. Finding something worth keeping means writing
#: it down, and the family that found nothing writes nothing. Everything here
#: is the write path or what watches it — none of it decides an answer, and
#: each says why it is here.
THE_CONSEQUENCES: dict[str, str] = {
    "core/governance_context.py": "a kept result is a governed write",
    "core/runtime/file_write_gateway.py": "the write itself",
    "core/runtime/atomic_writer.py": "how the write lands",
    "core/runtime/state_ownership.py": "where a write is allowed to go",
    "core/runtime/lockdep.py": "the lock the write takes",
    "core/runtime/which_thread_may_do_this.py": "which thread the write ran on",
    "core/runtime/pressure_stall.py": "what the write cost",
    "core/runtime/service_registry.py": "resolving the gateway",
    "core/observability/histograms.py": "recording how long it took",
}


def what_ran_during(work: Callable[[], Any]) -> tuple[set[str], Any]:
    """Every ``core`` function that executed while ``work`` ran.

    A profile hook rather than a trace hook: it fires on call and return only,
    which is cheap enough that the run being watched is the run that would
    have happened.
    """
    seen: set[str] = set()
    root = str(ROOT / "core")
    mine = threading.get_ident()

    def watch(frame: Any, event: str, _arg: Any) -> None:
        if event != "call" or threading.get_ident() != mine:
            return
        filename = frame.f_code.co_filename
        if filename.startswith(root):
            seen.add(
                f"{filename[len(str(ROOT)) + 1:]}:{frame.f_code.co_name}"
            )

    before = sys.getprofile()
    sys.setprofile(watch)
    try:
        answer = work()
    finally:
        sys.setprofile(before)
    return seen, answer


def the_same_path_twice(
    families: dict[str, Callable[[], Any]],
    *,
    the_search: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run each family, and report what ran for one and not the others.

    ``the_search`` names the modules that ARE the hypothesis space. A
    difference inside them is the search working; a difference outside them is
    a path selected by the shape of the problem.
    """
    # Run everything twice before measuring, for two different reasons.
    #
    # Once, because whichever family goes first pays for every module-level
    # import on the path, and those frames then appear for it and nothing
    # else — a difference that says only which one ran first.
    #
    # Twice, because the first solve of something new is a developmental
    # event: she writes the result down and the improvement loop reacts. The
    # second solve of the same thing is not. Measuring a first solve against a
    # repeat compares a learner with itself at two different moments, which is
    # a real difference and not the one this probe is asking about.
    for _ in range(2):
        for work in families.values():
            try:
                work()
            except Exception:  # noqa: BLE001 — a warm-up's answer is not the result
                pass

    ran: dict[str, set[str]] = {}
    answered: dict[str, Any] = {}
    for name, work in families.items():
        seen, answered[name] = what_ran_during(work)
        # A module body or a class body is import, not work.
        ran[name] = {
            one for one in seen if not one.endswith(":<module>")
        }

    shared: set[str] = set.intersection(*ran.values()) if ran else set()
    everything: set[str] = set.union(*ran.values()) if ran else set()
    only_for: dict[str, list[str]] = {}
    for name, seen in ran.items():
        others = everything - seen
        only = seen - shared
        only_for[name] = sorted(one for one in only if one not in others or True)

    allowed = (
        (THE_SEARCH, *THE_CONSEQUENCES) if the_search is None else the_search
    )

    def inside_the_search(where: str) -> bool:
        return any(where.startswith(one) for one in allowed)

    outside = {
        name: sorted(one for one in only if not inside_the_search(one))
        for name, only in only_for.items()
    }
    return {
        "families": sorted(ran),
        "answered": {name: str(one)[:120] for name, one in answered.items()},
        "functions_that_ran": {name: len(seen) for name, seen in ran.items()},
        "shared": len(shared),
        "only_for_one_family": {name: len(only) for name, only in only_for.items()},
        "outside_the_search": {name: len(only) for name, only in outside.items()},
        "what_is_outside": {
            name: only[:20] for name, only in outside.items() if only
        },
        "the_search": list(allowed),
        # The raw sets, so the same run can be scored against a different
        # declared search without running her again. Two runs in one process
        # are not independent — the first teaches her something the second
        # already knows — so re-scoring is the only honest way to check that
        # the probe can fail.
        "ran": {name: sorted(seen) for name, seen in ran.items()},
        # The claim, and it is narrower than "the same code ran".
        "passed": not any(outside.values()),
        "what_this_shows": (
            "no module ran for one family and never for the other outside the "
            "hypothesis search itself"
        ),
    }


def score_again(result: dict[str, Any], the_search: tuple[str, ...]) -> dict[str, Any]:
    """Re-score a finished run against a different declared search.

    For checking that the probe can fail. Running her twice to find out would
    measure what she learned in between, which is a real difference and not
    the one being asked about.
    """
    ran = {name: set(seen) for name, seen in (result.get("ran") or {}).items()}
    if not ran:
        return {"passed": None, "why": "the run kept no sets to score"}
    shared = set.intersection(*ran.values())
    outside = {
        name: sorted(
            one
            for one in (seen - shared)
            if not any(one.startswith(where) for where in the_search)
        )
        for name, seen in ran.items()
    }
    return {
        "the_search": list(the_search),
        "outside_the_search": {name: len(only) for name, only in outside.items()},
        "what_is_outside": {name: only[:20] for name, only in outside.items() if only},
        "passed": not any(outside.values()),
    }
