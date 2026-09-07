"""Thirteen readings of her own record, and a queue of what they turned up.

An opportunity is not a kind of thing that can go wrong. It is a number that
can be read off what she has already written down, and the reason that matters
is that a list of kinds is the hand-written taxonomy this work exists to
remove. Adding a fourteenth reading is admitting a term, not editing a list.

What each one reads

    recurrence          how often this shape comes back
    expense             what it costs against what comparable families cost
    redundancy          what the library would save by naming a shared part
    disuse              how long since something was last any use
    silence             how often nothing could be said at all
    unevenness          how much the cost of one family varies
    slack               how much of the budget went unspent
    a hint of transfer  a stored term shaped like the case in hand
    the slow route      which route spends most per answer it gives
    a flat improver     admissions that have stopped paying
    waste               how much of the search was spent on nothing
    yield               answers per candidate, now against before
    progress            whether the corpus is getting shorter

Each gives a score between nothing and one, and evidence. Nothing is not
"no opportunity"; it is "this reading has nothing to say", and the difference
matters when the answer to a question is that she should do nothing.

The queue
---------
Opportunities persist. A reading that fires once and is not acted on is still
true next time, and rediscovering it every tick would make the loop a loop.
What ages an entry out is its evidence going stale, not a clock.
"""

from __future__ import annotations

import logging
import statistics
from dataclasses import dataclass, field
from typing import Any, Callable

__all__ = [
    "AnOpportunity",
    "HOW_MANY_ARE_QUEUED",
    "THE_READINGS",
    "forget_the_agenda",
    "the_agenda",
    "what_she_notices",
    "a_reading_she_wrote",
]

logger = logging.getLogger("Aura.WhatSheNoticesAboutHerself")

#: How many opportunities are held. Small, because an agenda longer than the
#: number of things she could do about them is a list of regrets.
HOW_MANY_ARE_QUEUED = 32


@dataclass(frozen=True, slots=True)
class AnOpportunity:
    """Something the record says might be worth doing something about."""

    #: Which reading turned it up.
    noticed_by: str
    #: What it is about: a family, a part, or her as a whole.
    about: str
    #: Between nothing and one. Not a probability; a strength of evidence.
    strength: float
    #: The numbers behind it, so the reading can be checked rather than trusted.
    evidence: dict[str, Any] = field(default_factory=dict)
    #: What it would cost to act, where the reading can say.
    costs: int = 0

    def describes(self) -> str:
        return f"{self.noticed_by} on {self.about}: {self.strength:.2f}"


# ── the readings ─────────────────────────────────────────────────────────


def _episodes() -> list[Any]:
    from core.cognition.the_record_of_her_own_work import the_record

    return list(the_record().kept)


def _by_family() -> dict[str, list[Any]]:
    found: dict[str, list[Any]] = {}
    for one in _episodes():
        found.setdefault(one.family, []).append(one)
    return found


def _share(part: float, whole: float) -> float:
    return 0.0 if whole <= 0 else max(0.0, min(1.0, part / whole))


def recurrence() -> list[AnOpportunity]:
    """A shape that keeps coming back is a shape worth spending on."""
    from core.cognition.the_record_of_her_own_work import the_record

    seen = max(1, the_record().seen)
    return [
        AnOpportunity(
            "recurrence",
            family,
            _share(len(rows), seen),
            {"times": len(rows), "of": seen},
            costs=int(statistics.mean(one.walked for one in rows)),
        )
        for family, rows in _by_family().items()
        if len(rows) > 1
    ]


def expense() -> list[AnOpportunity]:
    """A family that costs far more than the ones like it.

    Against the median rather than the mean, because one dear family would
    otherwise raise the bar it is being judged against.
    """
    rows = _by_family()
    costs = {
        family: statistics.mean(one.walked for one in each)
        for family, each in rows.items()
    }
    if len(costs) < 2:
        return []
    usual = statistics.median(costs.values())
    if usual <= 0:
        return []
    return [
        AnOpportunity(
            "expense",
            family,
            _share(cost - usual, cost),
            {"costs": round(cost, 1), "usual": round(usual, 1)},
            costs=int(cost),
        )
        for family, cost in costs.items()
        if cost > usual
    ]


def redundancy() -> list[AnOpportunity]:
    """Two parts with something in common, and what naming it would save."""
    from core.cognition.the_floor_she_stands_on import how_long
    from core.cognition.what_she_is_made_of import (
        the_most_they_have_in_common,
        what_she_is_made_of,
    )

    terms = [one for one in what_she_is_made_of() if one.term is not None]
    found: list[AnOpportunity] = []
    for at, first in enumerate(terms):
        for second in terms[at + 1 :]:
            shared = the_most_they_have_in_common(first.term, second.term)
            if shared is None:
                continue
            saved = how_long(shared) - 1
            whole = how_long(first.term) + how_long(second.term)
            found.append(
                AnOpportunity(
                    "redundancy",
                    f"{first.at} and {second.at}",
                    _share(saved, whole),
                    {"shared": saved, "of": whole},
                    costs=saved,
                )
            )
    return found


def disuse() -> list[AnOpportunity]:
    """Something nothing has needed for a long time."""
    from core.cognition.the_record_of_her_own_work import the_record
    from core.cognition.what_she_is_made_of import what_she_is_made_of

    seen = max(1, the_record().seen)
    return [
        AnOpportunity(
            "disuse",
            one.at,
            _share(one.idle, seen),
            {"idle": one.idle, "used": one.used},
        )
        for one in what_she_is_made_of()
        if one.idle is not None and one.idle > 0 and not one.holds_up
    ]


def silence() -> list[AnOpportunity]:
    """How often nothing could be said at all."""
    rows = _by_family()
    found = []
    for family, each in rows.items():
        quiet = sum(1 for one in each if one.route is None)
        if quiet:
            found.append(
                AnOpportunity(
                    "silence",
                    family,
                    _share(quiet, len(each)),
                    {"silent": quiet, "of": len(each)},
                    costs=int(statistics.mean(one.walked for one in each)),
                )
            )
    return found


def nothing_she_has() -> list[AnOpportunity]:
    """A family where everything she has was tried and none of it held.

    The other twelve readings are about what happened: what recurs, what
    costs, what sits idle. This one is about what is MISSING, and it is the
    only reading whose answer is "write a new kind of action" rather than
    "take one of the actions you have".

    The distinction it turns on is the one `silence` cannot make. Silence
    counts occasions where nothing could be said, which happens both when she
    never tried and when she tried everything. Only the second says the set of
    operators has a gap in it, and telling them apart needs the record to say
    what was tried rather than only what worked.

    Strength is the share of what she has that this family has already
    defeated, read at its lower bound over the episodes. A family met twice
    that beat both actions she owns is weaker evidence than one met forty
    times that beat all nine, and the lower bound is what makes that true of
    the number rather than only of the sentence.

    The first version of this multiplied the share by the episode count and
    divided by it again inside ``_share``, so the count cancelled and both
    families above scored 1.000. An authored metric that does not measure
    what its docstring says is worse than no metric, because the docstring is
    what anyone reads.
    """
    from core.cognition.how_sure_she_is import how_sure
    from core.cognition.what_she_could_do_next import the_actions_she_has

    has = {one.name for one in the_actions_she_has()}
    if not has:
        return []
    found: list[AnOpportunity] = []
    for family, each in _by_family().items():
        tried = {one.tried for one in each if one.tried} & has
        held = {one.route for one in each if one.route}
        if not tried or held:
            # Something worked here, or nothing has been tried. Neither is a
            # gap in what she can do.
            continue
        beaten = len(tried) / len(has)
        # One observation per episode: on this occasion, did what she has
        # answer it. The mean is `beaten`; the half-width is what the episode
        # count buys. Reading the LOWER bound is what makes forty episodes
        # stronger evidence than two, rather than the same number.
        middle, width = how_sure([beaten] * len(each))
        found.append(
            AnOpportunity(
                "nothing she has",
                family,
                _share(middle - width, 1.0),
                {
                    "tried": sorted(tried),
                    "of": len(has),
                    "over": len(each),
                    "share": round(beaten, 3),
                    "how_far_out": round(width, 3),
                    "what it wants": "an action of a kind she does not have",
                },
                costs=int(statistics.mean(one.walked for one in each)),
            )
        )
    return found


def unevenness() -> list[AnOpportunity]:
    """A family whose cost is unpredictable, which is a thing to be curious about.

    Uncertainty that an experiment could reduce, and it is measured rather than
    asserted: a family answered in four candidates once and four thousand the
    next time is not understood.
    """
    found = []
    for family, each in _by_family().items():
        costs = [one.walked for one in each]
        if len(costs) < 3:
            continue
        middle = statistics.mean(costs)
        if middle <= 0:
            continue
        spread = statistics.pstdev(costs)
        found.append(
            AnOpportunity(
                "unevenness",
                family,
                _share(spread, middle),
                {"spread": round(spread, 1), "middle": round(middle, 1)},
                costs=int(middle),
            )
        )
    return found


def slack() -> list[AnOpportunity]:
    """Budget that went unspent, which is when development is cheapest.

    What she may spend on a family against what answering it actually took. A
    wide gap means development costs almost nothing in foregone work, and that
    is the whole of what idle time means here.
    """
    from core.cognition.she_decides_to_develop import what_it_may_spend

    found = []
    for family, each in _by_family().items():
        spent = statistics.mean(one.walked for one in each)
        may = what_it_may_spend(family, costs_now=int(spent))
        if may > spent > 0:
            found.append(
                AnOpportunity(
                    "slack",
                    family,
                    _share(may - spent, may),
                    {"may spend": may, "spent": round(spent, 1)},
                    costs=int(may - spent),
                )
            )
    return found


def a_hint_of_transfer() -> list[AnOpportunity]:
    """A stored term shaped like a family she has met, with no label saying so."""
    from core.cognition.what_she_is_made_of import (
        the_most_they_have_in_common,
        what_she_is_made_of,
    )
    from core.cognition.the_floor_she_stands_on import how_long

    parts = [one for one in what_she_is_made_of() if one.term is not None]
    found = []
    for at, first in enumerate(parts):
        for second in parts[at + 1 :]:
            if first.kind == second.kind:
                continue  # a hint across kinds, not within one
            shared = the_most_they_have_in_common(first.term, second.term)
            if shared is None:
                continue
            found.append(
                AnOpportunity(
                    "a hint of transfer",
                    f"{first.at} into {second.at}",
                    _share(how_long(shared), how_long(first.term)),
                    {"shared": how_long(shared)},
                )
            )
    return found


def the_slow_route() -> list[AnOpportunity]:
    """Which route spends most per answer it actually gives."""
    from core.cognition.what_it_is_worth_doing import what_the_record_says_is_slow

    rows = what_the_record_says_is_slow()
    if len(rows) < 2:
        return []
    worst = rows[0]["per answer"]
    best = rows[-1]["per answer"]
    if worst <= 0:
        return []
    return [
        AnOpportunity(
            "the slow route",
            rows[0]["route"],
            _share(worst - best, worst),
            {"per answer": round(worst, 1), "best": round(best, 1)},
            costs=int(worst),
        )
    ]


def a_flat_improver() -> list[AnOpportunity]:
    """Changes that have stopped paying — the one reading that is about her deciding."""
    from core.cognition.what_it_is_worth_doing import how_often_a_change_has_paid

    paid = how_often_a_change_has_paid()
    admissions = sum(1 for one in _episodes() if one.admitted)
    if admissions < 2:
        return []
    return [
        AnOpportunity(
            "a flat improver",
            "what a change is worth",
            _share(1.0 - paid, 1.0),
            {"paid": round(paid, 2), "changes": admissions},
        )
    ]


def waste() -> list[AnOpportunity]:
    """How much of a search went on candidates that were not the answer.

    One minus the answer's share of the walk. Near one everywhere is what a
    search with no idea where to look produces, and it is the signal that says
    the ORDER is what to change rather than the language.
    """
    from core.cognition.how_she_learns_to_look import how_the_last_ones_looked

    lived = how_the_last_ones_looked()
    if not lived:
        return []
    ranks = []
    for one in lived:
        how_many = max(1, len(one.get("features") or ()))
        ranks.append(1.0 - (1.0 / how_many))
    return [
        AnOpportunity(
            "waste",
            "the order she tries them in",
            _share(statistics.mean(ranks), 1.0),
            {"rankings": len(lived)},
        )
    ]


def yield_now_against_before() -> list[AnOpportunity]:
    """Answers per candidate, lately against earlier.

    Falling yield is the plateau signal: she is spending more and getting less,
    and that is a fact about her rather than about any one family.
    """
    rows = _episodes()
    if len(rows) < 8:
        return []
    half = len(rows) // 2
    def per(each: list[Any]) -> float:
        spent = sum(one.walked for one in each) or 1
        got = sum(1 for one in each if one.route is not None)
        return got / spent

    before, now = per(rows[:half]), per(rows[half:])
    if before <= 0 or now >= before:
        return []
    return [
        AnOpportunity(
            "yield",
            "herself",
            _share(before - now, before),
            {"was": round(before, 6), "now": round(now, 6)},
        )
    ]


def progress() -> list[AnOpportunity]:
    """Whether the corpus is getting shorter — compression progress rather than surprise.

    Raw surprise is what gets a system stuck in front of noise. What matters is
    whether the description is shrinking, and a corpus that has stopped
    shrinking is one where a new abstraction would be worth looking for.
    """
    from core.cognition.the_floor_she_stands_on import how_long
    from core.cognition.what_she_is_made_of import what_she_is_made_of

    terms = [one for one in what_she_is_made_of() if one.term is not None]
    if len(terms) < 2:
        return []
    whole = sum(how_long(one.term) for one in terms)
    return [
        AnOpportunity(
            "progress",
            "the library",
            _share(whole, whole + len(terms) * 8),
            {"symbols": whole, "parts": len(terms)},
            costs=whole,
        )
    ]


#: The readings in force. Not a taxonomy of what can go wrong — a list of the
#: numbers her record can currently produce, and `a_reading_she_wrote` adds one
#: without an edit here.
THE_READINGS: dict[str, Callable[[], list[AnOpportunity]]] = {
    "recurrence": recurrence,
    "expense": expense,
    "redundancy": redundancy,
    "disuse": disuse,
    "silence": silence,
    "nothing she has": nothing_she_has,
    "unevenness": unevenness,
    "slack": slack,
    "a hint of transfer": a_hint_of_transfer,
    "the slow route": the_slow_route,
    "a flat improver": a_flat_improver,
    "waste": waste,
    "yield": yield_now_against_before,
    "progress": progress,
}


_AGENDA: list[AnOpportunity] = []


def a_reading_she_wrote(name: str, reads: Callable[[], list[AnOpportunity]]) -> None:
    """Add a reading. The call that keeps the thirteen from being a ceiling."""
    THE_READINGS[str(name)] = reads


def what_she_notices() -> list[AnOpportunity]:
    """Run every reading and keep what they turned up, strongest first.

    A reading that raises is a reading that had nothing to say. Nothing here
    stops because one of them is broken, because a detector is a guess and a
    broken guess should cost a reading rather than a tick.
    """
    found: list[AnOpportunity] = []
    for name, reads in THE_READINGS.items():
        try:
            found.extend(one for one in reads() if one.strength > 0)
        except Exception:  # noqa: BLE001 - a reading that raises read nothing
            logger.info("the %s reading gave nothing", name, exc_info=True)
    found.sort(key=lambda one: -one.strength)
    _AGENDA[:] = found[:HOW_MANY_ARE_QUEUED]
    return list(_AGENDA)


def the_agenda() -> list[AnOpportunity]:
    """What the last reading turned up, without reading again."""
    return list(_AGENDA)


def forget_the_agenda() -> None:
    _AGENDA.clear()
