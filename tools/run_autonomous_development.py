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
from tools.run_grown_against_reset_heads import Family, _a_family  # noqa: E402


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


def proactive(rng: random.Random, *, episodes_wanted: int) -> dict[str, Any]:
    """A stream she can already answer. Does she improve it anyway?

    The experiment a failure-driven ladder cannot run, because nothing fails.
    """
    from core.cognition.sequence_induction import (
        _Situation,
        _she_may_improve_a_working_answer,
        _what_family_this_is,
    )

    _fresh()
    _register()
    made: list[Family] = []
    while len(made) < episodes_wanted:
        one = _a_family(rng, [], 3)
        if one is None:
            continue
        if induce_from(one.transitions) is None:
            continue  # she can already say it, or this arm is not about it
        made.append(one)
    if not made:
        return {"arm": "proactive", "families": 0, "why": "no sayable family"}

    was = [_what_it_costs_to_say(one) for one in made]
    changed: list[str] = []
    for one in made:
        family = _what_family_this_is(one.transitions)
        cost = _what_it_costs_to_say(one)
        note_an_episode(family, route="a meaning she induced", walked=cost)
        said = _she_may_improve_a_working_answer(one.transitions, family, cost)
        if said:
            changed.append(said)
    now = [_what_it_costs_to_say(one) for one in made]
    return {
        "arm": "proactive",
        "families": len(made),
        "walked before": round(statistics.mean(was), 1),
        "walked after": round(statistics.mean(now), 1),
        "changed": changed,
        "she started": who_started_it("trigger"),
        "nothing failed": True,
    }


def transfer(rng: random.Random, *, apart: bool) -> dict[str, Any]:
    """Is what the first family bought used on the second, unbidden?

    `apart` is the control: a second family with no structure in common, where
    a mechanism that reuses out of habit rather than out of fit would show the
    same numbers and be caught.
    """
    _fresh()
    _register()
    first = None
    while first is None:
        first = _a_family(rng, [], 3)
    library = [one for one in WHERE_FROM.values()]
    second = None
    tries = 0
    while second is None and tries < 60:
        tries += 1
        second = _a_family(rng, [] if apart else list(library)[:3], 3)
    if second is None:
        return {"arm": "transfer", "apart": apart, "why": "no second family"}

    before = _what_it_costs_to_say(second)
    note_an_episode("first", route="a meaning she induced", walked=_what_it_costs_to_say(first))
    after = _what_it_costs_to_say(second)
    told_to_reuse = any(one.started_by == "asked" for one in the_trace())
    return {
        "arm": "transfer",
        "apart": apart,
        "second before": before,
        "second after": after,
        "used what the first bought": after < before,
        "was told to reuse": told_to_reuse,
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
    first = what_is_worth_doing_now()
    m1 = None
    if first.action is not None:
        m1 = first.action.do_it(None)
    ranked_after = how_soon_they_are_found(the_order_she_uses())

    # M2: with M1 in force, does it reach for a way of deciding?
    second = what_is_worth_doing_now()
    m2 = None
    if second.action is not None:
        m2 = second.action.do_it(None)
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
    decided = what_is_worth_doing_now()
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
            got = proactive(rng, episodes_wanted=said.episodes)
        elif arm == "transfer":
            got = transfer(rng, apart=False)
            found.append(got)
            print(json.dumps(got))
            got = transfer(rng, apart=True)
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
