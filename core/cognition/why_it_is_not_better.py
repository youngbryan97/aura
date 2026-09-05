"""Why this is not better than it is, answered by intervening rather than by labelling.

The obvious way to ask which part of her is the bottleneck is to classify: is
this a representation problem, a search problem, a data problem? That question
has eight answers in the literature and none of them is decidable, and a
classifier over them is the hand-written taxonomy this work exists to remove
wearing a different hat.

So the question is asked the other way round. A cause is a PART plus a CHANGE
to it, and its strength is what the change is worth: how much the families she
has met improve if she makes it, over what making it costs. The part with the
largest number is the bottleneck. Whether a reader calls that a representation
problem or a search problem is decided afterwards by looking at which part won,
and `what_a_reader_would_call_it` does exactly that and nothing else.

Two things follow that a classifier cannot give.

The set of causes is open. `a_cause_she_wrote` takes a term and a part and
gives back a cause, so a kind of problem nobody anticipated is admitted rather
than added to an enumeration. That is the escape hatch, and without it the
first fourteen causes become the next ceiling.

The answer is causal. It is measured by doing the thing, on families held out
from the one that raised the question. Correlational readings — this part was
active when it went badly — are what make a system confident about the wrong
component, and they cost nothing to produce, which is why they are everywhere.

Two rivals, a little each
-------------------------
"Search longer" and "search differently" look identical from outside and the
difference is the whole question in a hard case. `spend_a_little_on_both` gives
each a small share of the budget, measures, and keeps the winner. That is what
tells them apart without anybody naming the difference, and it is cheap enough
to run before committing to either.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "ACause",
    "a_cause_she_wrote",
    "spend_a_little_on_both",
    "what_a_reader_would_call_it",
    "why_it_is_not_better",
]

logger = logging.getLogger("Aura.WhyItIsNotBetter")


@dataclass(frozen=True, slots=True)
class ACause:
    """A part, a change to it, and what the change turned out to be worth."""

    #: Which part. `kind/name`, the address `what_she_is_made_of` uses.
    at: str
    #: What the change is called, for the record.
    change: str
    #: Making it. Returns whether it was made.
    make_it: Callable[[], bool]
    #: Putting it back.
    put_it_back: Callable[[], None] = lambda: None
    #: What it cost to try, in candidates.
    costs: int = 1
    #: Whether she wrote this cause rather than being handed it.
    hers: bool = False
    #: The term, where she wrote one. This is the open constructor: a kind of
    #: problem nobody anticipated arrives here rather than in an enumeration.
    written: Any = None

    def describes(self) -> str:
        return f"{self.change} at {self.at}"


def what_a_reader_would_call_it(cause: ACause) -> str:
    """The old name for this kind of problem, for a reader and for nothing else.

    Nothing in the choosing consults this. It exists because a person reading
    a record wants the word, and because saying which word applies AFTER the
    measurement is a different act from using the word to decide.
    """
    kind = cause.at.split("/", 1)[0]
    return {
        "word": "a representation problem",
        "what is done": "a representation problem",
        "way of building": "a representation problem",
        "way of computing": "an algorithm problem",
        "rule": "an algorithm problem",
        "the search": "a search problem",
        "the deciding": "a problem with how she decides",
    }.get(kind, "something with no old name")


def _lesion_causes() -> list[ACause]:
    """One cause per part: what happens without it."""
    from core.cognition.sequence_induction import (  # noqa: PLC2701
        _everything_she_can_say,
        _put_back,
    )
    from core.cognition.what_she_is_made_of import (
        _take_it_out,  # noqa: PLC2701
        what_she_is_made_of,
    )

    found: list[ACause] = []
    for part in what_she_is_made_of():
        if part.kind in {"the search", "the deciding"}:
            continue
        held: list[Any] = []

        def make_it(one: Any = part, keep: list[Any] = held) -> bool:
            keep.append(_everything_she_can_say())
            return _take_it_out(one)

        def put_it_back(keep: list[Any] = held) -> None:
            if keep:
                _put_back(keep.pop())

        found.append(
            ACause(at=part.at, change="without it", make_it=make_it,
                   put_it_back=put_it_back)
        )
    return found


def _machinery_causes(*, within: float) -> list[ACause]:
    """Causes at the search and at the deciding, where the change is a search."""
    from core.cognition.she_improves_her_own_deciding import (
        a_worth_that_would_have_chosen_better,
        an_order_that_finds_them_sooner,
    )
    from core.cognition.the_order_she_tries_them_in import (
        forget_the_order,
        the_order_she_wrote,
    )
    from core.cognition.what_it_is_worth_doing import (
        forget_the_worth,
        the_worth_she_wrote,
    )

    def a_better_order() -> bool:
        found = an_order_that_finds_them_sooner(within=within)
        if found is None:
            return False
        the_order_she_wrote(found)
        return True

    def a_better_worth() -> bool:
        found = a_worth_that_would_have_chosen_better(within=within)
        if found is None:
            return False
        the_worth_she_wrote(found)
        return True

    return [
        ACause(
            at="the search/the order she tries them in",
            change="a different order",
            make_it=a_better_order,
            put_it_back=lambda: forget_the_order() and None,
            costs=max(1, int(within * 1000)),
        ),
        ACause(
            at="the deciding/what a change is worth",
            change="a different way of deciding",
            make_it=a_better_worth,
            put_it_back=lambda: forget_the_worth() and None,
            costs=max(1, int(within * 1000)),
        ),
    ]


def a_cause_she_wrote(
    at: str, change: str, term: Any, *, make_it: Callable[[], bool]
) -> ACause:
    """A cause she wrote. The open constructor.

    Without this the causes above are an enumeration and an enumeration is a
    ceiling. A kind of problem nobody anticipated is a term plus a part, and
    both are values.
    """
    return ACause(at=at, change=change, make_it=make_it, hers=True, written=term)


def why_it_is_not_better(
    probe: Sequence[tuple[str, Sequence[Any]]],
    *,
    costs: Callable[[Sequence[Any]], int],
    within: float = 4.0,
    among: Sequence[ACause] | None = None,
) -> list[dict[str, Any]]:
    """Every cause, ranked by what its change turns out to be worth.

    Worth being the improvement over the cost of making it, so a change that
    helps a little and costs nothing can beat one that helps more and costs a
    great deal. Measured on the probe, which is held out.
    """
    if not probe:
        return []
    causes = list(
        among
        if among is not None
        else [*_lesion_causes(), *_machinery_causes(within=within)]
    )
    before = sum(costs(cases) for _name, cases in probe)
    found: list[dict[str, Any]] = []
    for cause in causes:
        try:
            made = cause.make_it()
        except Exception:  # noqa: BLE001 - a change that raises was not made
            logger.info("could not make %s", cause.describes(), exc_info=True)
            made = False
        if not made:
            continue
        try:
            after = sum(costs(cases) for _name, cases in probe)
        finally:
            try:
                cause.put_it_back()
            except Exception:  # noqa: BLE001 - putting back is best effort
                logger.info("could not undo %s", cause.describes(), exc_info=True)
        # Positive means the change made things worse, which for a lesion means
        # the part was earning its place. For a proposed replacement it means
        # the replacement is bad. Both readings are the same arithmetic.
        moved = before - after
        found.append(
            {
                "at": cause.at,
                "change": cause.change,
                "worth": moved / max(1, cause.costs),
                "moved": moved,
                "cost": cause.costs,
                "a reader would call it": what_a_reader_would_call_it(cause),
            }
        )
    found.sort(key=lambda row: -row["worth"])
    return found


def spend_a_little_on_both(
    first: ACause,
    second: ACause,
    probe: Sequence[tuple[str, Sequence[Any]]],
    *,
    costs: Callable[[Sequence[Any]], int],
) -> dict[str, Any]:
    """Try both cheaply and keep whichever won.

    The way to tell "search longer" from "search differently" without anybody
    naming the difference. Both get the same small share, both are measured on
    the same held-out families, and the answer is the one that moved the
    number. Nothing here knows what either of them is.
    """
    ranked = why_it_is_not_better(
        probe, costs=costs, among=[first, second]
    )
    if not ranked:
        return {"won": None, "why": "neither change could be made"}
    won = ranked[0]
    return {
        "won": won["at"],
        "change": won["change"],
        "worth": won["worth"],
        "over": [row["at"] for row in ranked[1:]],
        "why": f"{won['moved']:,} candidates over {len(probe)} families",
    }
