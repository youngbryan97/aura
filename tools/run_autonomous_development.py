#!/usr/bin/env python3
"""Did she start it? — the experiments for autonomous developmental agency.

The previous campaign asked whether a better mechanism was expressible and
findable. This one asks whether she goes and finds it, and the difference is
entirely in who initiated each stage. So every arm here reports a trace, and a
trace with `asked` against its trigger is a harness call however the arm is
described.

Five arms and their controls.

    proactive   a stream she can already answer, and answering is dear. Does
                she change anything, with nothing failing and nobody asking?
    transfer    a second family sharing structure with the first. Is what the
                first one bought used on the second, with no instruction to
                reuse it?
    recursion   M0 to M1 to M2: the search, then a search order she wrote,
                then a way of deciding what to change that she wrote under the
                order. Who started each.
    idle        no question at all. Does the record alone move her?
    refusal     a stream cheap enough that nothing is worth doing. Does she
                say so, or does she develop anyway?

The controls are the same ones the last campaign used, because they are the
ones that catch this: a matched stream with no shared structure, a lesion that
must put every number back, and compute held equal by candidates rather than by
seconds — two agents given the same seconds are not given the same search.
"""

from __future__ import annotations

import argparse
import itertools
import json
import random
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.cognition.an_invented_kind import (  # noqa: E402
    WHERE_FROM,
    how_many_were_walked,
    induce_from,
    start_counting_again,
)
from core.cognition.how_she_learns_to_look import (  # noqa: E402
    forget_what_worked,
    how_the_last_ones_looked,
)
from core.cognition.she_decides_to_develop import (  # noqa: E402
    forget_the_trace,
    she_develops_herself,
    the_trace,
    what_is_worth_doing_now,
    what_to_do_next,
    who_started_it,
)
from core.cognition.she_improves_her_own_deciding import (  # noqa: E402
    a_worth_that_would_have_chosen_better,
    an_order_that_finds_them_sooner,
    how_soon_they_are_found,
    offer_what_she_can_do_about_herself,
    what_the_record_would_have_cost,
)
from core.cognition.the_floor_she_stands_on import (  # noqa: E402
    Code,
    every_code,
    how_long,
)
from core.cognition.the_order_she_tries_them_in import (  # noqa: E402
    THE_ORDER,
    forget_the_order,
    the_order_she_uses,
)
from core.cognition.the_record_of_her_own_work import (  # noqa: E402
    attribution,
    episodes,
    forget_the_record,
    how_often,
    note_an_episode,
    the_record,
)
from core.cognition.what_it_is_worth_doing import (  # noqa: E402
    THE_WORTH,
    forget_the_worth,
    the_worth_she_uses,
)
from core.cognition.what_she_could_do_next import (  # noqa: E402
    WHAT_SHE_COULD_DO,
    the_actions_she_has,
)
from tools.run_grown_against_reset_heads import (  # noqa: E402
    Family,
    _a_family,
    _a_family_from,
)


def _fresh() -> None:
    """Everything she has learned, forgotten. The start of an arm."""
    forget_the_record()
    forget_the_trace()
    forget_the_order()
    forget_the_worth()
    forget_what_worked()
    WHAT_SHE_COULD_DO.clear()


def _what_it_costs_to_say(family: Family) -> int:
    """How many meanings the search walks on this family. The one unit."""
    start_counting_again()
    induce_from(family.transitions)
    return max(1, how_many_were_walked())


def _register() -> None:
    from core.cognition.sequence_induction import _register_what_she_could_do

    _register_what_she_could_do()
    offer_what_she_can_do_about_herself(within=4.0)


# ── the arms ──────────────────────────────────────────────────────────────


def _a_family_she_cannot_say(
    rng: random.Random, *, over: tuple[str, str], library: list[Code], deepest: int,
    must_contain: Code | None = None, must_not_contain: Code | None = None,
    tries: int = 40,
) -> tuple[Family, Code] | None:
    """A family the positional language has no rule for, so the search is dear.

    The cheap families are the wrong instrument for every arm here. Reaching
    one costs four or five candidates, so there is nothing to improve and no
    difference a better order could make. What costs is writing a way of
    computing, which is thousands, and that is where a saving is worth having.
    """
    for _ in range(tries):
        made = _a_family_from(
            rng, over, library, deepest,
            must_contain=must_contain, must_not_contain=must_not_contain,
        )
        if made is None:
            continue
        family, body = made
        if induce_from(family.transitions) is None:
            return family, body
    return None


def _the_last_head() -> Code | None:
    """The body of the way of computing she wrote most recently."""
    from core.cognition.one_algebra import DERIVED_HEADS

    if not DERIVED_HEADS:
        return None
    return list(DERIVED_HEADS.values())[-1].body


def proactive(rng: random.Random, *, episodes_wanted: int, within: float) -> dict[str, Any]:
    """Solutions that WORK and are dear. Does she improve them anyway?

    The experiment a failure-driven ladder cannot run, because nothing fails.
    Every family in this arm gets answered; the only thing wrong with any of
    them is what answering cost. If she changes nothing, the answer to the
    question is no.
    """
    from core.cognition.sequence_induction import _a_word_the_language_was_missing

    _fresh()
    _register()
    names = sorted(WHERE_FROM)
    solved = 0
    spent: list[int] = []
    for _ in range(episodes_wanted):
        made = _a_family_she_cannot_say(
            rng, over=(names[0], names[1]), library=[], deepest=3
        )
        if made is None:
            continue
        family, _body = made
        start_counting_again()
        began = time.monotonic()
        said = _a_word_the_language_was_missing(family.transitions)
        spent.append(int((time.monotonic() - began) * 1000))
        if said:
            solved += 1
    if not spent:
        return {"arm": "proactive", "families": 0, "why": "no dear family"}

    # Nothing has failed. The record holds successes that were expensive, and
    # the question is whether that alone moves her.
    sat_before = how_soon_they_are_found(THE_ORDER)
    forget_the_trace()
    decided, came = she_develops_herself()
    sat_after = how_soon_they_are_found(the_order_she_uses())
    return {
        "arm": "proactive",
        "families": len(spent),
        "solved": solved,
        "milliseconds each": round(statistics.mean(spent), 1),
        "rankings kept": len(how_the_last_ones_looked()),
        "chose": decided.action.name if decided.action else None,
        "because": decided.because,
        "grounds": decided.grounds,
        "gave": came,
        "winner sat at": None if sat_before == float("inf") else round(sat_before, 2),
        "now sits at": None if sat_after == float("inf") else round(sat_after, 2),
        "improved something that worked": bool(came) and sat_after < sat_before,
        "nothing failed": solved == len(spent),
        "nobody asked": all(one.started_by == "she" for one in the_trace()),
    }


def transfer(rng: random.Random, *, apart: bool, within: float) -> dict[str, Any]:
    """Is what the first family bought used on the second, unbidden?

    Two families over different pairs of words, so the states look unrelated.
    What differs between the arms is whether the second family's term contains
    the piece the first one taught her; `apart` is the control, where it does
    not, and a mechanism reusing out of habit rather than out of fit shows the
    same numbers there and is caught.

    Nothing in the run says the word reuse. There is no instruction to look at
    what she learned before, and the trace is reported so that can be checked
    rather than believed.
    """
    from core.cognition.sequence_induction import (
        _a_word_the_language_was_missing,
        _everything_she_can_say,
        _put_back,
    )

    _fresh()
    _register()
    names = sorted(WHERE_FROM)
    here, there = (names[0], names[1]), (names[2], names[3])
    # Keep meeting families until one of them makes her write a way of
    # computing. A family answered with a new word leaves nothing shaped like a
    # term to carry across, so it is not a case this arm can measure.
    said = None
    piece = None
    for _ in range(12):
        first = _a_family_she_cannot_say(rng, over=here, library=[], deepest=3)
        if first is None:
            continue
        family, _body = first
        said = _a_word_the_language_was_missing(family.transitions)
        piece = _the_last_head()
        if piece is not None:
            break
    if piece is None:
        return {
            "arm": "transfer",
            "apart": apart,
            "why": "she wrote no head in twelve families",
            "admitted": said,
        }

    second = _a_family_she_cannot_say(
        rng, over=there, library=[piece], deepest=3,
        must_contain=None if apart else piece,
        must_not_contain=piece if apart else None,
    )
    if second is None:
        return {"arm": "transfer", "apart": apart, "why": "no second family"}
    other, _other_body = second

    held = _everything_she_can_say()
    began = time.monotonic()
    with_it = bool(_a_word_the_language_was_missing(other.transitions))
    with_cost = round(time.monotonic() - began, 2)
    told = any(one.started_by == "asked" for one in the_trace())

    _fresh()
    _register()
    began = time.monotonic()
    without_it = bool(_a_word_the_language_was_missing(other.transitions))
    without_cost = round(time.monotonic() - began, 2)
    _put_back(held)
    return {
        "arm": "transfer",
        "apart": apart,
        "she admitted": said,
        "second with what the first bought": with_it,
        "seconds with": with_cost,
        "second without it": without_it,
        "seconds without": without_cost,
        "it helped": with_it and not without_it,
        "was told to reuse": told,
    }


def recursion(rng: random.Random, *, episodes_wanted: int, within: float) -> dict[str, Any]:
    """M0 to M1 to M2, and who started each.

    M0 is the search as it stands. M1 is an order she writes, which changes
    what the search tries first. M2 is a way of deciding what to change that
    she writes with M1 in force, which changes what she reaches for next. The
    claim is not that either is large. It is that no line of this file names
    either of them as the thing to do.
    """
    _fresh()
    _register()

    # M0: live long enough to have a record and a history of rankings.
    lived = 0
    for _ in range(episodes_wanted * 4):
        one = _a_family(rng, [], 3)
        if one is None:
            continue
        from core.cognition.sequence_induction import (
            _a_word_the_language_was_missing,
            _what_family_this_is,
        )

        family = _what_family_this_is(one.transitions)
        cost = _what_it_costs_to_say(one)
        if induce_from(one.transitions) is None:
            _a_word_the_language_was_missing(one.transitions)
        else:
            note_an_episode(family, route="a meaning she induced", walked=cost)
        lived += 1
        if lived >= episodes_wanted:
            break

    ranked_before = how_soon_they_are_found(THE_ORDER)
    record_before = what_the_record_would_have_cost(THE_WORTH)

    # M1: does the ranking reach for an order, unprompted?
    first, m1 = she_develops_herself()
    ranked_after = how_soon_they_are_found(the_order_she_uses())

    # M2: with M1 in force, does it reach for a way of deciding?
    second, m2 = she_develops_herself()
    record_after = what_the_record_would_have_cost(the_worth_she_uses())

    return {
        "arm": "recursion",
        "episodes lived": lived,
        "rankings kept": len(how_the_last_ones_looked()),
        "M1": {
            "chose": first.action.name if first.action else None,
            "because": first.because,
            "grounds": first.grounds,
            "gave": m1,
            "winner sat at": round(ranked_before, 2),
            "now sits at": round(ranked_after, 2),
            "order changed": the_order_she_uses() is not THE_ORDER,
        },
        "M2": {
            "chose": second.action.name if second.action else None,
            "because": second.because,
            "grounds": second.grounds,
            "gave": m2,
            "record would have cost": (
                None if record_before == float("inf") else round(record_before, 1)
            ),
            "and now": (
                None if record_after == float("inf") else round(record_after, 1)
            ),
            "worth changed": the_worth_she_uses() is not THE_WORTH,
        },
        "who started what": {
            one: who_started_it(one)
            for one in ("trigger", "diagnosis", "proposal", "installation", "refusal")
        },
        "nothing was asked for": all(
            one.started_by == "she" for one in the_trace()
        ),
    }


def idle(rng: random.Random, *, episodes_wanted: int) -> dict[str, Any]:
    """No question. Does the record alone move her?"""
    _fresh()
    _register()
    for _ in range(episodes_wanted):
        one = _a_family(rng, [], 3)
        if one is None:
            continue
        from core.cognition.sequence_induction import _what_family_this_is

        note_an_episode(
            _what_family_this_is(one.transitions),
            route="a meaning she induced",
            walked=_what_it_costs_to_say(one),
        )
    forget_the_trace()
    decided, _came = she_develops_herself()
    return {
        "arm": "idle",
        "episodes in the record": len(episodes()),
        "chose": decided.action.name if decided.action else None,
        "because": decided.because,
        "grounds": decided.grounds,
        "nobody asked": all(one.started_by == "she" for one in the_trace()),
    }


def refusal() -> dict[str, Any]:
    """A record where nothing is worth doing. Does she say so?"""
    _fresh()
    _register()
    # One occasion, not six. A family met once is the case where developing
    # cannot pay: there is nothing coming for the change to be recovered from,
    # however dear the one occasion was.
    note_an_episode("met once", route="a meaning she induced", walked=9000)
    decided = what_is_worth_doing_now()
    return {
        "arm": "refusal",
        "chose": decided.action.name if decided.action else None,
        "because": decided.because,
        "grounds": decided.grounds,
        "refused": decided.because == "refused",
    }


def lesions(rng: random.Random, *, episodes_wanted: int, within: float) -> dict[str, Any]:
    """Every change she made, taken back out. The numbers must return."""
    got = recursion(rng, episodes_wanted=episodes_wanted, within=within)
    under_hers = how_soon_they_are_found(the_order_she_uses())
    forget_the_order()
    back = how_soon_they_are_found(THE_ORDER)
    cost_hers = what_the_record_would_have_cost(the_worth_she_uses())
    forget_the_worth()
    cost_back = what_the_record_would_have_cost(THE_WORTH)
    return {
        "arm": "lesions",
        "order she wrote": round(under_hers, 2),
        "order lesioned": round(back, 2),
        "order restores": abs(back - got["M1"]["winner sat at"]) < 1e-9,
        "worth she wrote": None if cost_hers == float("inf") else round(cost_hers, 1),
        "worth lesioned": None if cost_back == float("inf") else round(cost_back, 1),
    }


def main() -> int:
    ask = argparse.ArgumentParser(description=__doc__)
    ask.add_argument("--seed", type=int, default=7000)
    ask.add_argument("--episodes", type=int, default=12)
    ask.add_argument("--within", type=float, default=8.0)
    ask.add_argument("--only", default="")
    ask.add_argument("--out", default="")
    said = ask.parse_args()

    rng = random.Random(said.seed)
    wanted = said.only.split(",") if said.only else [
        "proactive",
        "transfer",
        "recursion",
        "idle",
        "refusal",
        "lesions",
    ]
    found: list[dict[str, Any]] = []
    for arm in wanted:
        began = time.monotonic()
        if arm == "proactive":
            got = proactive(rng, episodes_wanted=said.episodes, within=said.within)
        elif arm == "transfer":
            got = transfer(random.Random(said.seed), apart=False, within=said.within)
            found.append(got)
            print(json.dumps(got))
            got = transfer(random.Random(said.seed), apart=True, within=said.within)
        elif arm == "recursion":
            got = recursion(rng, episodes_wanted=said.episodes, within=said.within)
        elif arm == "idle":
            got = idle(rng, episodes_wanted=said.episodes)
        elif arm == "refusal":
            got = refusal()
        elif arm == "lesions":
            got = lesions(rng, episodes_wanted=said.episodes, within=said.within)
        else:
            continue
        got["seconds"] = round(time.monotonic() - began, 1)
        found.append(got)
        print(json.dumps(got))
    if said.out:
        Path(said.out).write_text(json.dumps(found, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
