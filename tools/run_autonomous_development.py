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
import ast
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
    forget_the_record,
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
    _contains,
    _correspondence,
)


def _fresh() -> None:
    """Everything she has learned, forgotten. The start of an arm.

    Everything means everything. Leaving the proposer installed between arms
    meant the first family in a run installed the deeper one and every family
    after reported that action as giving nothing — which reads as a failure
    and was a success that had already happened. A reset that misses one thing
    is a control that quietly stops being one.
    """
    from core.cognition.the_proposer_she_can_replace import forget_the_proposer
    from core.cognition.what_counts_as_better import forget_the_objective
    from core.cognition.what_she_could_do_next import WHAT_THEY_HAVE_DONE
    from core.cognition.what_she_expects_of_herself import forget_what_she_expected
    from core.cognition.what_she_notices_about_herself import forget_the_agenda

    forget_the_record()
    forget_the_trace()
    forget_the_order()
    forget_the_worth()
    forget_what_worked()
    forget_the_proposer()
    forget_the_objective()
    forget_the_agenda()
    forget_what_she_expected()
    WHAT_THEY_HAVE_DONE.clear()
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


#: Bodies drawn once and kept. Drawing them per family meant taking four
#: thousand terms out of the enumerator on every attempt, forty attempts to a
#: family, and the campaign spent ten minutes without printing a line. The pool
#: is a property of the depth and the library, not of the family, so it is
#: built once per pair of those.
_POOL: dict[tuple[int, int], list[Code]] = {}


def _bodies(library: list[Code], deepest: int) -> list[Code]:
    key = (deepest, len(library))
    if key not in _POOL:
        _POOL[key] = list(
            itertools.islice(
                every_code(
                    deepest=deepest,
                    variables=6,
                    constants=(0, 1, 2),
                    also=tuple(library),
                ),
                4000,
            )
        )
    return _POOL[key]


def _a_family_she_cannot_say(
    rng: random.Random, *, over: tuple[str, str], library: list[Code], deepest: int,
    must_contain: Code | None = None, must_not_contain: Code | None = None,
    tries: int = 400,
) -> tuple[Family, Code] | None:
    """A family the positional language has no rule for, so the search is dear.

    The cheap families are the wrong instrument for every arm here. Reaching
    one costs four or five candidates, so there is nothing to improve and no
    difference a better order could make. What costs is writing a way of
    computing, which is thousands, and that is where a saving is worth having.
    """
    first, second = WHERE_FROM[over[0]], WHERE_FROM[over[1]]
    bodies = list(_bodies(library, deepest))
    rng.shuffle(bodies)
    for body in bodies[:tries]:
        if how_long(body) < 2:
            continue
        if must_contain is not None and not _contains(body, must_contain):
            continue
        if must_not_contain is not None and _contains(body, must_not_contain):
            continue
        places = _correspondence(body, first, second)
        if places is None:
            continue
        transitions = []
        for size, wanted in places.items():
            before = tuple(range(100, 100 + size))
            transitions.append((before, tuple(before[one] for one in wanted)))
        family = Family(
            transitions=transitions, over=over, from_a_term_of=how_long(body)
        )
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
    # Four, not twelve. On a cold record every action is unpriced and each
    # family costs about a minute of exploration, so a loop that keeps trying
    # until it gets what it wants spends longer than the whole campaign. Four
    # families and an honest "she wrote no head" is a result; twelve and a
    # timeout is not.
    said = None
    piece = None
    for _ in range(4):
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
            "why": "she wrote no head in four families",
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


# ── the lesions that decide whether any of this is agency ────────────────


def no_developmental_call_in_the_harness() -> dict[str, Any]:
    """Read this file and check it never calls a routine that develops.

    Asking whether development is self-directed while the harness installs
    things is asking nothing. The forbidden names are the ones that search for
    or install a change; asking whether anything is worth doing is allowed and
    is the point — a fixed decision opportunity is not a fixed decision.

    Lesions are allowed and named, because putting something back is how a
    claim gets tested rather than made.
    """
    forbidden = {
        "the_order_she_wrote",
        "the_proposer_she_wrote",
        "the_worth_she_wrote",
        "the_head_she_wrote",
        "a_way_of_computing_she_wrote",
        "a_rule_she_wrote",
        "a_maker_she_wrote",
        "grow_at",
        "an_order_that_finds_them_sooner",
        "a_worth_that_would_have_chosen_better",
        "the_action_she_wrote",
        "promote",
    }
    here = ast.parse(Path(__file__).read_text(encoding="utf-8"))
    called: list[str] = []
    for node in ast.walk(here):
        if isinstance(node, ast.Call):
            named = getattr(node.func, "id", None) or getattr(node.func, "attr", None)
            if named in forbidden:
                called.append(f"{named} at line {node.lineno}")
    return {
        "arm": "the harness",
        "forbidden calls": called,
        "clean": not called,
        "lesions it does call": sorted(
            {
                named
                for node in ast.walk(here)
                if isinstance(node, ast.Call)
                and (named := getattr(node.func, "id", ""))
                and named.startswith("forget_")
            }
        ),
    }


def initiation_follows_the_value(rng: random.Random) -> dict[str, Any]:
    """Does what she decides move when what it is worth moves?

    The scheduler is held fixed: the same call, the same number of times. Only
    the record differs, and with it the value. A chooser that answers the same
    thing either way is on a timer whatever it is called.
    """
    _fresh()
    _register()

    def a_record_worth_developing_for() -> None:
        forget_the_record()
        for _ in range(8):
            note_an_episode("dear", route=None, walked=40_000)
        note_an_episode("dear", route="a way of computing", walked=900,
                        admitted="a way of computing")
        for _ in range(8):
            note_an_episode("dear", route="a way of computing", walked=30)

    def a_record_not_worth_it() -> None:
        forget_the_record()
        note_an_episode("met once", route="an answer", walked=3)

    said: list[str] = []
    for put_it_in_place in (a_record_worth_developing_for, a_record_not_worth_it):
        put_it_in_place()
        decided = what_is_worth_doing_now()
        said.append(decided.because)
    return {
        "arm": "initiation",
        "with a reason": said[0],
        "without one": said[1],
        "the choice follows the value": said[0] != said[1],
    }


def the_opportunity_lesion(rng: random.Random, *, episodes_wanted: int) -> dict[str, Any]:
    """Destroy the evidence, keep the work. Development must stop.

    The record's episodes stay and their families are shuffled, so the same
    total cost is spent and no shape recurs. If she still develops she is
    firing on a clock rather than on evidence, and this is the arm that catches
    it.
    """
    _fresh()
    _register()
    for at in range(episodes_wanted):
        note_an_episode("a shape that recurs", route=None, walked=9_000,
                        about=[((1, 2, 3), (3, 2, 1))])
    with_evidence = what_is_worth_doing_now()

    forget_the_record()
    for at in range(episodes_wanted):
        # Same episodes, same cost, no shape. Recurrence is what is destroyed
        # and nothing else.
        note_an_episode(f"a shape seen once ({at})", route=None, walked=9_000,
                        about=[((1, 2, 3), (3, 2, 1))])
    without_evidence = what_is_worth_doing_now()
    return {
        "arm": "the opportunity lesion",
        "with the evidence": with_evidence.because,
        "with it shuffled": without_evidence.because,
        "stops without evidence": (
            with_evidence.action is not None and without_evidence.action is None
        ),
    }


def the_meta_causality_lesion(
    rng: random.Random, *, episodes_wanted: int, within: float
) -> dict[str, Any]:
    """Put the old machinery back before the second change. Does the second get harder?

    Two changes happening is not recursion. Recursion is the first change
    participating in producing the second, and the only way to show that is to
    take the first one away and watch the second become less likely.
    """
    got = recursion(rng, episodes_wanted=episodes_wanted, within=within)
    reached_with = got["M2"]["chose"] is not None and got["M2"]["gave"] is not None

    # Again, and this time the first change is undone before the second is
    # attempted, with everything else held.
    forget_the_order()
    forget_the_worth()
    second = what_is_worth_doing_now()
    reached_without = second.action is not None and second.because == "chosen"
    return {
        "arm": "the meta-causality lesion",
        "reached the second change with the first in place": reached_with,
        "reached it with the first undone": reached_without,
        "the first helped": reached_with and not reached_without,
        "M1 was": got["M1"]["chose"],
        "M2 was": got["M2"]["chose"],
    }


def information_against_computation() -> dict[str, Any]:
    """A question no amount of searching can answer. Does she ask instead?

    Two readings fit everything shown and disagree about the case in hand. No
    search resolves that, because the evidence does not contain the answer;
    what resolves it is one more example. The decision to ask rather than to
    search is a different decision from a longer search, and this is whether
    she can make it.
    """
    _fresh()
    _register()
    from core.cognition.an_invented_kind import UNSETTLED
    from core.cognition.what_she_could_do_next import WHAT_SHE_COULD_DO

    # One example is what leaves several readings standing. A reversal and a
    # rotation say the same thing about three cells shown once and different
    # things about four, so no amount of searching settles it and one more
    # example does.
    said = _the_answer(
        "Given 1 2 3 -> 3 2 1, what does 1 2 3 4 give?"
    )
    asked = WHAT_SHE_COULD_DO.get("ask for the example that settles it")
    came = asked.do_it(None) if asked is not None else None
    return {
        "arm": "information against computation",
        "readings she has not settled": len(UNSETTLED),
        "she asked": came,
        "chose to ask rather than search": bool(came),
        "the answer she gave": (said or "")[:60],
    }


def _the_answer(text: str) -> str:
    from core.cognition.sequence_induction import answer_sequence_question

    return answer_sequence_question(text)


def transfer_against_three_controls(
    rng: random.Random, *, within: float
) -> list[dict[str, Any]]:
    """The same first family, and three seconds: same shape, similar surface, neither.

    One arm is not a result. A part that helps everything helps nothing, and a
    part that helps the isomorphic domain and also the unrelated one is being
    reused out of habit. What has to hold is the pattern across the three.
    """
    found = []
    for which, apart in (("isomorphic", False), ("unrelated", True)):
        got = transfer(random.Random(rng.randrange(1 << 30)), apart=apart, within=within)
        got["second domain"] = which
        found.append(got)
    return found


def a_search_she_wrote_is_new(rng: random.Random, *, within: float) -> dict[str, Any]:
    """Is the order she wrote a different program, or the same one tuned?

    Judged behaviourally and causally, never by name. Behaviourally: is it a
    different term from the one it replaced. Causally: is the winner found
    sooner on occasions it was not fitted to. A rule that only changes a
    constant fails the first and a rule that changes nothing useful fails the
    second.

    The order is not produced here. An earlier version of this arm called the
    search directly, and the harness check caught it: producing the thing you
    are asking whether she produces is not evidence of anything. She is asked
    what is worth doing, and what is in force afterwards is inspected.
    """
    _fresh()
    _register()
    for _ in range(6):
        one = _a_family_she_cannot_say(
            rng, over=tuple(sorted(WHERE_FROM))[:2], library=[], deepest=3
        )
        if one is None:
            continue
        from core.cognition.sequence_induction import (  # noqa: PLC2701
            _a_word_the_language_was_missing,
        )

        _a_word_the_language_was_missing(one[0].transitions)
    was = how_soon_they_are_found(THE_ORDER)
    _decided, came = she_develops_herself()
    found = the_order_she_uses()
    if found is THE_ORDER:
        return {
            "arm": "a search she wrote",
            "wrote one": False,
            "she chose": _decided.action.name if _decided.action else None,
            "because": _decided.because,
            "rankings": len(how_the_last_ones_looked()),
        }
    now = how_soon_they_are_found(found)
    from core.cognition.the_floor_she_stands_on import how_long, written_down

    return {
        "arm": "a search she wrote",
        "wrote one": True,
        "she chose": _decided.action.name if _decided.action else None,
        "gave": came,
        "symbols": how_long(found),
        "the one it replaced": how_long(THE_ORDER),
        "a different program": written_down(found) != written_down(THE_ORDER),
        "winner sat at": round(was, 2),
        "now sits at": round(now, 2),
        "found sooner": now < was,
        "rankings it was judged on": len(how_the_last_ones_looked()),
    }


def the_metrics() -> dict[str, Any]:
    """What the whole run comes to, in the numbers the claim needs."""
    from core.cognition.how_a_change_is_promoted import the_chain_holds, the_receipts
    from core.cognition.what_she_could_do_next import how_wrong_she_was

    receipts = the_receipts()
    started = who_started_it()
    hers = started.get("she", 0)
    asked_for = started.get("asked", 0)
    return {
        "arm": "the metrics",
        "developmental stages": hers + asked_for,
        "how many she started": hers,
        "how many were asked for": asked_for,
        "initiation rate": round(hers / max(1, hers + asked_for), 3),
        "changes promoted": len(receipts),
        "promoted with no external command": sum(
            1 for one in receipts if one.asked_from_outside is None
        ),
        "the receipt chain holds": the_chain_holds(),
        "what each action has done": how_wrong_she_was(),
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
        "harness",
        "initiation",
        "proactive",
        "transfer",
        "recursion",
        "idle",
        "refusal",
        "information",
        "search",
        "opportunity-lesion",
        "meta-lesion",
        "lesions",
        "metrics",
    ]
    found: list[dict[str, Any]] = []
    for arm in wanted:
        began = time.monotonic()
        if arm == "transfer3":
            for one in transfer_against_three_controls(rng, within=said.within):
                found.append(one)
                print(json.dumps(one), flush=True)
            continue
        if arm == "proactive":
            got = proactive(rng, episodes_wanted=said.episodes, within=said.within)
        elif arm == "transfer":
            got = transfer(random.Random(said.seed), apart=False, within=said.within)
            found.append(got)
            print(json.dumps(got), flush=True)
            got = transfer(random.Random(said.seed), apart=True, within=said.within)
        elif arm == "recursion":
            got = recursion(rng, episodes_wanted=said.episodes, within=said.within)
        elif arm == "idle":
            got = idle(rng, episodes_wanted=said.episodes)
        elif arm == "refusal":
            got = refusal()
        elif arm == "lesions":
            got = lesions(rng, episodes_wanted=said.episodes, within=said.within)
        elif arm == "harness":
            got = no_developmental_call_in_the_harness()
        elif arm == "initiation":
            got = initiation_follows_the_value(rng)
        elif arm == "information":
            got = information_against_computation()
        elif arm == "search":
            got = a_search_she_wrote_is_new(rng, within=said.within)
        elif arm == "opportunity-lesion":
            got = the_opportunity_lesion(rng, episodes_wanted=said.episodes)
        elif arm == "meta-lesion":
            got = the_meta_causality_lesion(
                rng, episodes_wanted=said.episodes, within=said.within
            )
        elif arm == "metrics":
            got = the_metrics()
        else:
            continue
        got["seconds"] = round(time.monotonic() - began, 1)
        found.append(got)
        print(json.dumps(got), flush=True)
    if said.out:
        Path(said.out).write_text(json.dumps(found, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
