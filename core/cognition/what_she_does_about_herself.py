"""Three more things she can do about herself, and a reason to keep them.

The eight rungs widen a language. The two in `she_improves_her_own_deciding`
change how she searches and how she decides. These three change what she is
made of, and each is here because the record can say when it is worth doing.

**Let go of a part that pays nothing.** Every entry taxes every later search, so
a language that only ever grows gets slower at everything. Disuse alone is not
grounds — a thing can be unused and still the only route to something. The
grounds are a lesion: take it out, measure the families she has met, and let go
only when the number does not move.

**One name for what two parts share.** Anti-unification over two terms gives the
least general term both are instances of. Where that is more than a hole, the
shared part is a thing, and naming it shortens every later term containing it.
This is the action a system that only appends can never take.

**Ask for the one example that would settle it.** Sometimes the answer is not in
any amount of searching, because several readings fit everything she has been
shown and they disagree. Then what is wanted is information rather than
computation, and the two are different decisions. She cannot answer her own
question, so what this does is decide to ask and record that she did — which is
the part that was missing, since deciding to ask was never in the choice set at
all.

The stepping stone
------------------
A change is kept when the held-out families get cheaper, and also when one of
them that could not be said before can be said now. The second is the more
interesting case and it is the whole of what a stepping stone is: the change
paid nothing today and moved something from out of reach to in reach. Without
that clause the value is one step deep and a stone that only enables a later
step scores negative, which is the honest limit this removes.
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

__all__ = [
    "offer_what_she_can_do_about_what_she_is_made_of",
    "the_one_she_should_let_go",
    "what_two_parts_share",
    "worth_keeping",
]

logger = logging.getLogger("Aura.WhatSheDoesAboutHerself")


def _probe(than: str = "") -> list[tuple[str, tuple]]:
    from core.cognition.the_record_of_her_own_work import other_families

    return other_families(than=than)


def _costs(cases: Sequence[Any]) -> int:
    from core.cognition.an_invented_kind import (
        how_many_were_walked,
        induce_from,
        start_counting_again,
    )

    start_counting_again()
    induce_from(cases)
    return max(1, how_many_were_walked())


def _sayable(cases: Sequence[Any]) -> bool:
    from core.cognition.an_invented_kind import induce_from

    return induce_from(cases) is not None


def worth_keeping(
    before: dict[str, tuple[int, bool]], probe: Sequence[tuple[str, tuple]]
) -> tuple[bool, str]:
    """Did this change pay, on families it was not chosen for?

    Two ways to pay, and the second is the one a one-step value misses. Either
    the held-out families cost less, or one of them that could not be said at
    all can be said now. A stone that enables a later step scores nothing on
    cost and everything on reach.
    """
    from core.cognition.how_sure_she_is import more_likely_than_not_better
    from core.cognition.what_it_is_worth_doing import how_often_a_change_has_paid

    cheaper = 0
    opened: list[str] = []
    each: list[float] = []
    for name, cases in probe:
        was, could = before.get(name, (0, False))
        now = _costs(cases)
        can = _sayable(cases)
        cheaper += was - now
        # One bounded observation per family: did this one get cheaper. Bounded
        # because the interval below needs it to be, and a share is the honest
        # bounded form of a saving.
        each.append(1 if now < was else 0)
        if can and not could:
            opened.append(name)
    if opened:
        return True, f"it says {', '.join(opened)}, which it could not before"
    # Not "the total went down". One family falling by a lot while three rise
    # is a total that fell and a change that did not work, and a rule promoting
    # on the total promotes noise.
    #
    # The bar is her own history: how often a change has paid before. Beating a
    # coin would be the bar if she had never changed anything, and after a few
    # changes it is whatever the record says, which is the right comparison and
    # not a number anybody set.
    usual = how_often_a_change_has_paid()
    better, why = more_likely_than_not_better(
        sum(each), len(each), than=usual
    )
    if better:
        return True, f"{cheaper:,} fewer candidates over {len(probe)} families ({why})"
    return False, f"no better than her usual over {len(probe)} families ({why})"


def _how_it_stands(probe: Sequence[tuple[str, tuple]]) -> dict[str, tuple[int, bool]]:
    return {name: (_costs(cases), _sayable(cases)) for name, cases in probe}


def _the_library_is_over_budget(parts: Sequence[Any]) -> bool:
    """Whether she is carrying more entries than they are worth.

    Every entry is a branch in every later search, so the tax of the nth entry
    is n and the budget is where that exceeds what an entry saves. Both sides
    come off the record; where the record cannot say, the budget is the number
    of entries there are, which refuses to retire anything rather than guessing
    a ceiling.
    """

    try:
        from core.cognition.the_shape_of_her_library import where_the_budget_is

        carried = sum(1 for one in parts if getattr(one, "term", None) is not None)
        return carried > int(where_the_budget_is())
    except (AttributeError, ImportError, RuntimeError, TypeError, ValueError):
        return False


def the_one_she_should_let_go(probe: Sequence[tuple[str, tuple]]) -> Any | None:
    """The part whose absence costs least, where that is nothing at all.

    Disuse is the hint and the lesion is the evidence. A part that has not been
    used for a long time may still be the only route to something, and the only
    way to know is to take it out and look.
    """
    from core.cognition.what_she_is_made_of import (
        what_a_part_is_worth,
        what_she_is_made_of,
    )

    if not probe:
        return None
    parts = what_she_is_made_of()
    idle = [
        one
        for one in parts
        if one.kind in {"word", "what is done", "way of building", "way of computing", "rule"}
        and not one.holds_up
        and (one.idle is None or one.idle > 0)
    ]
    # Disuse is one hint. Being over the library's budget is the other, and it
    # is the question no per-entry gate can ask: two entries can each pay and
    # still both be worse than one entry that generalises them.
    #
    # `core/cognition/the_shape_of_her_library.py` was written to ask it and
    # nothing outside its own test ever called it, so every retirement decision
    # here has been made one entry at a time since the argument for asking
    # otherwise was written down. Over budget, the candidates widen to include
    # parts that are used: a part being useful is not the same as the library
    # being the right size.
    if _the_library_is_over_budget(parts):
        idle = [
            one
            for one in parts
            if one.kind
            in {"word", "what is done", "way of building", "way of computing", "rule"}
            and not one.holds_up
        ] or idle
    for part in sorted(idle, key=lambda one: -(one.idle or 0)):
        pays = what_a_part_is_worth(part, probe, costs=_costs)
        if pays is not None and pays <= 0:
            return part
    return None


def what_two_parts_share() -> tuple[Any, Any, Any] | None:
    """Two parts and the term they are both instances of, where one exists."""
    from core.cognition.what_she_is_made_of import (
        the_most_they_have_in_common,
        what_she_is_made_of,
    )

    terms = [one for one in what_she_is_made_of() if one.term is not None]
    for at, first in enumerate(terms):
        for second in terms[at + 1 :]:
            shared = the_most_they_have_in_common(first.term, second.term)
            if shared is not None:
                return first, second, shared
    return None


def offer_what_she_can_do_about_what_she_is_made_of() -> None:
    """Put the three in the registry, priced like everything else."""
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )

    def let_go(situation: Any = None) -> str | None:
        from core.cognition.what_rests_on_what import retract
        from core.cognition.what_she_can_take_back import only_if_it_pays
        from core.cognition.what_she_is_made_of import _take_it_out  # noqa: PLC2701

        probe = _probe()
        part = the_one_she_should_let_go(probe)
        if part is None:
            return None
        before = _how_it_stands(probe)
        # A removal that does not pay used to log "she kept it after all" over
        # a part that was already gone — the only way to know was to look at
        # the registry. The trial puts it back on every way out of this block.
        with only_if_it_pays(f"letting go of {part.at}") as trial:
            if part.kind == "way of computing":
                retract(part.name)
            elif not _take_it_out(part):
                return None
            kept, why = worth_keeping(before, probe)
            if not kept:
                logger.info("she kept %s after all: %s", part.at, why)
                return None
            trial.keep(why)
        logger.info("she let go of %s: %s", part.at, why)
        return f"let go of {part.at}"

    def one_name_for_both(situation: Any = None) -> str | None:
        from core.cognition.a_way_of_computing_she_wrote import as_a_head
        from core.cognition.one_algebra import DERIVED_HEADS, the_head_she_wrote

        from core.cognition.what_she_can_take_back import only_if_it_pays

        found = what_two_parts_share()
        if found is None:
            return None
        first, second, shared = found
        probe = _probe()
        if not probe:
            return None
        before = _how_it_stands(probe)
        name = f"what {first.name} and {second.name} share ({len(DERIVED_HEADS)})"
        # This one remembered to pop the head it added. The trial takes over
        # so that being right here stops depending on remembering, and so
        # that anything the head's installation touched comes back too.
        with only_if_it_pays(f"naming {name}") as trial:
            the_head_she_wrote(name, 3, as_a_head(shared))
            kept, why = worth_keeping(before, probe)
            if not kept:
                logger.info("the shared part bought nothing: %s", why)
                return None
            trial.keep(why)
        logger.info("she named what two parts share: %s — %s", name, why)
        return f"one name for what {first.name} and {second.name} share"

    def ask_for_an_example(situation: Any = None) -> str | None:
        from core.cognition.an_invented_kind import (
            UNSETTLED,
            what_would_tell_them_apart,
        )
        from core.cognition.sequence_induction import WHAT_WOULD_SETTLE_IT

        # The positional path first, because that is where most questions are
        # answered and so where most of what stays open lives. Reading only the
        # induced-meaning registry meant the action could see nothing on the
        # very cases it exists for.
        for family, asked in list(WHAT_WOULD_SETTLE_IT.items()):
            logger.info(
                "she decided to ask rather than to search about %s", family
            )
            return (
                f"asked what {list(asked)} gives, which settles {family}"
            )
        for unsure, meanings in list(UNSETTLED.items()):
            if len(meanings) < 2:
                continue
            for at, one in enumerate(meanings):
                for other in meanings[at + 1 :]:
                    asked = what_would_tell_them_apart(one, other)
                    if asked is None:
                        continue
                    logger.info(
                        "she decided to ask rather than to search: %s", list(asked)
                    )
                    return (
                        f"asked what {list(asked)} gives, which settles {unsure}"
                    )
        return None


    def take_the_cause_that_pays(situation: Any = None) -> str | None:
        """Rank the causes of it not being better, and make the top one.

        The obvious way to ask which part of her is the bottleneck is to
        classify — representation, search, data — and that question has eight
        answers in the literature, none decidable, and a classifier over them
        is the hand-written taxonomy this work exists to remove wearing a
        different hat. `core/cognition/why_it_is_not_better.py` asks it the
        other way: every cause is a change that can be MADE, ranked by what
        making it turns out to be worth on a held-out probe.

        It had no caller outside its own test, so the causes were ranked
        nowhere and nothing was ever made from the ranking. It is one of the
        actions she chooses among now, priced like the rest.
        """

        from core.cognition.why_it_is_not_better import (
            _lesion_causes,  # noqa: PLC2701
            _machinery_causes,  # noqa: PLC2701
            why_it_is_not_better,
        )

        probe = _probe()
        if not probe:
            return None
        # The same causes are ranked and then chosen from, so the winner is a
        # cause she can make rather than a row she can read. Ranking undoes
        # every change it tries — that is what makes the number a measurement
        # — so making the winner is a separate act.
        causes = [*_lesion_causes(), *_machinery_causes(within=4.0)]
        if not causes:
            return None
        ranked = why_it_is_not_better(probe, costs=_costs, among=causes)
        best = next(
            (row for row in ranked if float(row.get("worth") or 0.0) > 0.0), None
        )
        if best is None:
            return None
        for cause in causes:
            if cause.change != best["change"] or cause.at != best["at"]:
                continue
            try:
                if not cause.make_it():
                    return None
            except Exception:  # noqa: BLE001 - a change that raises was not made
                logger.info("could not make %s", cause.describes(), exc_info=True)
                return None
            logger.info(
                "she made the change that pays most: %s at %s (worth %.3f)",
                best["change"], best["at"], float(best["worth"]),
            )
            return str(best["change"])
        return None

    for name, over, kind, do_it in (
        ("let go of a part that pays nothing", "the words", "letting go", let_go),
        (
            "make the change that pays most",
            "the ways of computing",
            "a cause",
            take_the_cause_that_pays,
        ),
        (
            "one name for what two parts share",
            "the ways of computing",
            "a shared name",
            one_name_for_both,
        ),
        (
            "ask for the example that settles it",
            "the words",
            "asking",
            ask_for_an_example,
        ),
    ):
        if name not in WHAT_SHE_COULD_DO:
            what_she_could_do(
                name, over=over, kind=kind, do_it=do_it, needs_a_case=False
            )
