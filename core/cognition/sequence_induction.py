"""Answering a "what comes next" question by working the rule out.

The induction machinery has had no consumer outside its own battery. It could
learn a transformation from observations, keep it, compose with it and carry it
to the next world, and none of that ever met a person: the research
architecture had the mechanism and the live agent did not use it.

This is the seam. When somebody shows a few before-and-after examples and asks
what a new case becomes, the runtime works the rule out, applies it, and says
what the rule was — and keeps the shape, so the next question of the kind is
cheaper. No model is consulted to do it.

Where it stays quiet
--------------------
Single values. "45 becomes 15, 28 becomes 14" is a relation between numbers,
not a rearrangement of positions, and the language here is about position and
value substitution. Answering it would mean guessing. Sequences of two or more
are where the mechanism actually works, and anywhere else this returns nothing
and the ordinary path runs.

It also stays quiet when the rule it finds does not account for every example
it was shown, which is the same discipline the invention itself applies.
"""

from __future__ import annotations

import ast
import logging
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from core.cognition.language_limits import certify
from core.cognition.primitive_invention import (
    Transition,
    _index_forms,
    discriminating_probe,
)
from core.cognition.relation_language import RelationLanguage
from core.cognition.value_order import solve_ordering, solve_ordering_then_move
from core.runtime.errors import record_degradation

__all__ = ["SequenceQuestion", "answer_sequence_question", "read_sequence_question"]

logger = logging.getLogger("Aura.SequenceInduction")

#: A bracketed run of comma-separated cells. Deliberately not a bare comma
#: sequence: prose is full of commas, and a wrong reading here would answer a
#: question nobody asked.
_SEQUENCE = re.compile(r"[\[(]\s*([^\[\]()]{1,300}?)\s*[\])]")

#: A bare run of numbers, which is how a person writes one when they are not
#: writing code. "1 2 3 4 5 becomes 3 2 1 5 4" carries exactly what the
#: bracketed form carries, and refusing it sends a perfectly clear question to
#: be guessed at instead of worked out.
#:
#: What keeps prose out is not the brackets, it is everything after this: three
#: runs at least, an odd number of them, and something meaning "becomes"
#: between each pair. A sentence with numbers loose in it fails all three.
_BARE_RUN = re.compile(r"(?<![\w.,])(-?\d+(?:[ \t]*,?[ \t]+-?\d+)+)(?!\w)(?!\.\d)")

#: What sits between an example and its result. Any of them, or nothing at all
#: when the examples are simply listed in order.
_BECOMES = re.compile(
    r"\b(?:becomes?|gives?|turns?\s+into|maps?\s+to|yields?|->|=>)\b|→|->",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class SequenceQuestion:
    """Examples somebody showed, and the case they asked about."""

    shown: tuple[Transition, ...]
    asked: tuple[Any, ...]


def _cells(inside: str) -> tuple[Any, ...] | None:
    """The cells of one bracketed run, read as values rather than as text."""

    try:
        parsed = ast.literal_eval(f"[{inside}]")
    except (SyntaxError, ValueError):
        parsed = [
            piece.strip().strip("'\"")
            for piece in inside.split(",")
            if piece.strip()
        ]
    if not isinstance(parsed, list) or len(parsed) < 2:
        return None
    try:
        return tuple(parsed)
    except TypeError:
        return None


def _a_meaning_worked_out(question: SequenceQuestion) -> str | None:
    """An answer from a meaning she induced. Kept as its own step for the trace."""
    return _work_the_meaning_out(question)


def _everything_she_can_say() -> dict[str, dict]:
    """A copy of every registry a developmental action can add to.

    Not a list of what development is — a list of the places terms are kept, so
    a change that turned out not to pay can be taken back out whole.
    """
    from core.cognition.a_kind_of_thing_she_named import KINDS_OF_THING
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.an_invented_kind import (
        WAYS_TO_BUILD,
        WHAT_OF_IT,
        WHERE_FROM,
    )
    from core.cognition.one_algebra import DERIVED_HEADS

    return {
        "ways to build": dict(WAYS_TO_BUILD),
        "where from": dict(WHERE_FROM),
        "what of it": dict(WHAT_OF_IT),
        "kinds of thing": dict(KINDS_OF_THING),
        "derived heads": dict(DERIVED_HEADS),
        "rules with no shape": dict(RULES_WITH_NO_SHAPE),
    }


def _put_back(held: dict[str, dict]) -> None:
    """Undo a change that did not pay."""
    from core.cognition.a_kind_of_thing_she_named import KINDS_OF_THING
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.an_invented_kind import (
        WAYS_TO_BUILD,
        WHAT_OF_IT,
        WHERE_FROM,
    )
    from core.cognition.one_algebra import DERIVED_HEADS

    for registry, was in (
        (WAYS_TO_BUILD, held["ways to build"]),
        (WHERE_FROM, held["where from"]),
        (WHAT_OF_IT, held["what of it"]),
        (KINDS_OF_THING, held["kinds of thing"]),
        (DERIVED_HEADS, held["derived heads"]),
        (RULES_WITH_NO_SHAPE, held["rules with no shape"]),
    ):
        registry.clear()
        registry.update(was)


def _what_it_costs_to_say(pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]]) -> int:
    """How many meanings the search walks to reach this family. The measurement."""
    from core.cognition.an_invented_kind import (
        how_many_were_walked,
        induce_from,
        start_counting_again,
    )

    start_counting_again()
    induce_from(pairs)
    return max(1, how_many_were_walked())


def _she_may_improve_a_working_answer(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    family: str,
    walked: int,
) -> str | None:
    """Develop something that already works, because it costs too much.

    The whole point of the experiment this serves. A ladder driven by failure
    can never do it: there was no failure. The ranking can, because what it
    reads is the cost, and a cost is there whether or not the answer arrived.

    A candidate is kept only when OTHER families get cheaper to say, and taken
    back out when they do not. Judging it on the family that provoked it is how
    every compressor comes to overfit the incident that woke it: the change was
    chosen because it helped there, so of course it helps there. The record
    keeps the cases of a few families it has met, and those are the probe.

    With nothing held out there is nothing to judge on, and the honest answer
    is to make no change rather than to fall back on the trigger.
    """
    from core.cognition.she_decides_to_develop import what_to_do_next
    from core.cognition.the_record_of_her_own_work import (
        note_an_episode,
        other_families,
    )
    from core.cognition.what_she_could_do_next import the_actions_she_has

    _register_what_she_could_do()
    held_out = other_families(than=family)
    if not held_out:
        logger.info("nothing held out to judge a change on, so making none")
        return None
    decided = what_to_do_next(family, costs_now=walked, among=the_actions_she_has())
    if decided.action is None:
        return None
    held = _everything_she_can_say()
    was = sum(_what_it_costs_to_say(cases) for _name, cases in held_out)
    situation = _Situation(
        pairs=tuple((tuple(before), tuple(after)) for before, after in pairs),
        sayable=lambda: True,
    )
    try:
        said = decided.action.do_it(situation)
    except Exception:  # noqa: BLE001 - a failed action is a result
        logger.info("%s raised improving a working answer", decided.action.name,
                    exc_info=True)
        said = None
    now = (
        sum(_what_it_costs_to_say(cases) for _name, cases in held_out)
        if said
        else was
    )
    if not said or now >= was:
        _put_back(held)
        note_an_episode(
            family,
            route=None,
            walked=decided.worth.cost if decided.worth else walked,
        )
        logger.info(
            "she tried %s on a working answer and it did not pay on %d held-out "
            "families: %d then %d",
            decided.action.name,
            len(held_out),
            was,
            now,
        )
        return None
    note_an_episode(
        family,
        route=decided.action.name,
        walked=decided.worth.cost if decided.worth else walked,
        admitted=decided.action.kind,
    )
    logger.info(
        "she improved a working answer with %s: %d then %d over %d held-out "
        "families",
        decided.action.name,
        was,
        now,
        len(held_out),
    )
    return said


def _work_the_meaning_out(question: SequenceQuestion) -> str | None:
    """An answer from a meaning she induced, when the language has none.

    The examples are before-and-after pairs, which is exactly what a meaning is
    induced from: which two places each value is read from, and what is done
    with the pair. Solved on half of them and held to the half it never saw.

    Where SEVERAL readings account for everything shown, none of them is
    learned. Saying "your examples do not settle it" and then keeping one of
    the candidates says two different things — and the kept one is what steers
    the next answer, so the saying was decoration. She keeps the set, answers
    the cases every reading agrees about, and refuses the ones they do not.
    """
    from core.cognition.an_invented_kind import (
        KINDS,
        UNSETTLED,
        admit,
        everything_that_fits,
        hold_unsettled,
        induce_from,
        settle_with,
        what_they_agree_on,
        what_would_tell_them_apart,
    )

    pairs = [(one.before, one.after) for one in question.shown]

    # Anything she is unsure about, first: these examples may be what settles
    # it, and that is worth more than another answer.
    for unsure in list(UNSETTLED):
        if settle_with(unsure, pairs) == "settled":
            known = KINDS[unsure]
            answer = known.read(tuple(question.asked))
            if answer is not None:
                return (
                    f"{list(answer)}\n\n"
                    f"That settles something I was unsure about. Several readings "
                    f"fitted what you had shown me before; these rule out all but "
                    f"one, and it is {known.name}."
                )

    # Then something already settled, which is what makes the second question
    # of a kind cheaper than the first.
    for _kind, known in list(KINDS.items()):
        if not all(known.read(before) == after for before, after in pairs):
            continue
        answer = known.read(tuple(question.asked))
        if answer is None:
            continue
        return (
            f"{list(answer)}\n\n"
            f"I have met this before: {known.name}. I worked that meaning out "
            "from examples of it earlier and kept it, so this one needed no "
            "working out at all."
        )

    # Then a case she has an unsettled reading of already, where the readings
    # happen to agree.
    for unsure, meanings in list(UNSETTLED.items()):
        if not all(
            any(one.read(before) == after for one in meanings) for before, after in pairs
        ):
            continue
        agreed = what_they_agree_on(unsure, tuple(question.asked))
        if agreed is not None:
            return (
                f"{list(agreed)}\n\n"
                f"I am still unsure which of {len(meanings)} readings this is, "
                "but they all say the same thing about this one."
            )

    meaning = induce_from(pairs)
    if meaning is None:
        # Nothing in the language can say it — so the LANGUAGE is what is
        # missing, not the hypothesis.
        #
        # Every meaning she can form is a point in the closure of a handful of
        # ways to say where a value comes from and what is done with a pair,
        # and those were written down by a person. A family outside that
        # closure is not merely unsolved, it is unsayable, and no amount of
        # searching finds it because the search is over the wrong set.
        #
        # Where the values are distinct, where each one came from can be READ
        # OFF rather than guessed. If nothing the language already says
        # produces that correspondence, the correspondence is a new word — and
        # once it is a word, every hypothesis she can form afterwards may use
        # it.
        widened = _a_word_the_language_was_missing(pairs)
        if widened is None:
            return None
        meaning = induce_from(pairs)
        if meaning is None:
            return None
        answer = meaning.read(tuple(question.asked))
        if answer is None:
            return (
                "I had to invent a way of saying this, and it still does not "
                f"reach your case.\n\nNothing I could say described what your "
                f"examples do, so I worked out {widened} and added it "
                "to the language I make rules out of. It was read off what you "
                "showed me, and your case has parts it has never seen. Show me "
                "one more example covering them and it will."
            )
        admit(meaning.name, meaning)
        _keep_what_she_worked_out()
        return (
            f"{list(answer)}\n\n"
            "What your examples do sat outside the language I make rules out "
            "of, so no rule I could form would have described it. I worked "
            f"out {widened} and added it to that language. The rule is now "
            f"sayable: {meaning.name}. Everything I work out from here can "
            "use it."
        )
    fits = everything_that_fits(pairs)

    if len(fits) > 1:
        # Not settled, so nothing is learned. She keeps the readings, answers
        # what they agree on, and says what would tell them apart.
        name = hold_unsettled(meaning.name, fits)
        agreed = what_they_agree_on(name, tuple(question.asked))
        settle = _the_one_thing_worth_asking_him(
            {one.name: one for one in fits},
            len(question.asked),
            lambda reading, state: reading.read(state),
        )
        if agreed is not None:
            return (
                f"{list(agreed)}\n\n"
                f"{len(fits)} readings account for every example you showed me, "
                "and they all say the same thing about this one — so I can answer "
                "it without knowing which is right. I have not decided between "
                f"them.{settle}"
            )
        return (
            f"I cannot answer this one yet, and it is worth saying why.\n\n"
            f"{len(fits)} readings account for every example you showed me and "
            "they disagree about this case, so any answer I gave would be a "
            f"guess dressed as a rule.{settle}"
        )

    answer = meaning.read(tuple(question.asked))
    if answer is None:
        return None
    admit(meaning.name, meaning)
    try:
        from core.cognition.what_she_gave_meaning import keep

        keep()
    except (ImportError, OSError, RuntimeError, ValueError):
        pass  # no-op: an unkept meaning still answers this question
    return (
        f"{list(answer)}\n\n"
        "No rule I had could say this, so I worked out what the examples are "
        f"doing and gave it a meaning: {meaning.name}. It accounts for every "
        "example you showed me, including the ones I did not use to work it "
        "out, and I have kept it — ask me another of these and I will already "
        "know it."
    )


def _the_one_thing_worth_asking_him(
    readings: dict[str, Any], of_length: int, says: Any
) -> str:
    """The single example to ask him for, chosen against the whole field.

    Naming a state that parts two of sixteen readings and leaves fourteen
    standing is asking him to do the work twice. Both places she says "show me
    this one" used to pick a state off a pair and stop there. The act worth
    asking for is the one that settles the most of what is open, and saying how
    much it settles is what makes it worth his while to answer.
    """
    from core.cognition.the_experiment_that_settles_it import (
        every_act_that_settles_a_sequence,
        what_to_try,
    )

    if len(readings) < 2:
        return ""
    try:
        best = what_to_try(
            readings,
            every_act_that_settles_a_sequence(of_length),
            predicts=says,
        )
    except Exception as exc:  # noqa: BLE001 - recorded below, not swallowed
        record_degradation(
            "sequence_induction",
            exc,
            action="asked for no example, so the reading stays undecided",
        )
        return ""
    if best is None:
        return ""
    left = len(readings) - best.tells_apart
    beyond = "" if left <= 0 else f", leaving {left} still to separate"
    shown = sorted((str(list(answer)), name) for name, answer in best.expects.items())
    return (
        f" Show me what {list(best.do)} becomes and I will know which: it splits "
        f"them {best.tells_apart} ways{beyond}. "
        f"{shown[0][1]} says {shown[0][0]}; {shown[-1][1]} says {shown[-1][0]}."
    )


def meaning_lengths(pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]]) -> set[int]:
    """The lengths a set of examples was given at."""
    return {len(tuple(before)) for before, _after in pairs}


def _keep_what_she_worked_out() -> None:
    """Write down meanings and words alike, or carry on without."""
    try:
        from core.cognition.what_she_gave_meaning import keep

        keep()
    except (ImportError, OSError, RuntimeError, ValueError):
        pass  # no-op: an unkept meaning still answers this question


@dataclass(frozen=True, slots=True)
class _Situation:
    """What every developmental action is handed: the cases, and the test."""

    pairs: tuple[tuple[tuple[Any, ...], tuple[Any, ...]], ...]
    sayable: Any


def _a_kind_of_thing(situation: _Situation) -> str | None:
    """A kind of thing before a new word, because it is the answer that holds.

    When the failure splits the cases in two, what is missing is usually a
    DISTINCTION and not an operation. Reading a correspondence off the examples
    instead gives a word that works at the length it was read at and refuses
    every other — right about these cases and silent about the next one. A
    distinction is right about all of them, which is what makes it worth
    having.
    """
    from core.cognition.a_kind_of_thing_she_named import (
        KINDS_OF_THING,
        a_kind_of_thing_she_named,
        a_way_of_building_over,
    )
    from core.cognition.an_invented_kind import WAYS_TO_BUILD

    named = a_kind_of_thing_she_named(situation.pairs)
    if named is None:
        return None
    maker, _over = a_way_of_building_over(named)
    called = f"a way of saying it over {named.tells.name}"
    if called in WAYS_TO_BUILD:
        return None
    WAYS_TO_BUILD[called] = maker
    if situation.sayable():
        return (
            f"a kind of thing I had no name for ({named.tells.name}), "
            "and a way of saying things over it"
        )
    WAYS_TO_BUILD.pop(called, None)
    KINDS_OF_THING.pop(named.name, None)
    return None


def _a_way_of_saying_where_a_value_comes_from(situation: _Situation) -> str | None:
    """Everything constructible, not merely the words written down.

    A candidate that some composition already produces is new spelling for an
    old meaning, and admitting it would report growth that did not happen.
    """
    from core.cognition.an_invented_kind import WHERE_FROM, addressings
    from core.cognition.widening_the_language import (
        an_addressing_nobody_wrote,
        widen_with_addressing,
    )

    found = an_addressing_nobody_wrote(situation.pairs, already=addressings())
    if found is None:
        return None
    name = f"the way these move ({len(WHERE_FROM)})"
    said = widen_with_addressing(name, found)
    return f"a new way of saying where a value comes from ({said})" if said else None


def _a_way_of_combining_two_values(situation: _Situation) -> str | None:
    """Nothing was MOVED, so no addressing describes it.

    The values were made rather than taken. Then what is missing is a way of
    combining a pair, and that is derivable too — given where the two came
    from, what was done with them is whatever the result was.
    """
    from core.cognition.an_invented_kind import WHAT_OF_IT, addressings
    from core.cognition.widening_the_language import (
        an_operation_nobody_wrote,
        widen_with_operation,
    )

    closure = addressings()
    for first in list(closure.values()):
        for second in list(closure.values()):
            done = an_operation_nobody_wrote(
                situation.pairs, first, second, already=list(WHAT_OF_IT.values())
            )
            if done is None:
                continue
            name = f"what was done with these ({len(WHAT_OF_IT)})"
            said = widen_with_operation(name, done)
            return f"a new way of combining two values ({said})" if said else None
    return None


def _a_way_of_making_words_she_built(situation: _Situation) -> str | None:
    """Build one before reaching for one that was written.

    Activating a constructor somebody already wrote enlarges what she uses and
    never what she has; a recipe she composes can describe something the source
    does not contain.
    """
    from core.cognition.a_constructor_she_built import a_constructor_she_built

    built = a_constructor_she_built(situation.pairs, now_sayable=situation.sayable)
    if built is None:
        return None
    return f"a way of making words that I built rather than had ({built.name})"


def _a_way_of_building_words_she_wrote(situation: _Situation) -> str | None:
    """A way of building she WRITES, rather than one she is handed.

    This used to offer two makers by name — "one after another" and "twice
    over" — so the levels had no ceiling and the thing supplying candidates for
    them did. Whatever she reached was inside the closure of composition,
    inversion and iteration, and a family needing anything else was unreachable
    however long she looked. A way of building is now a TERM with a hole in it,
    in the same algebra a word is a term in, so there is no list to be at the
    end of.

    Only where the LANGUAGE is what failed. Nothing fitting is not evidence
    that a word is missing: it is equally what a search that went badly looks
    like, and writing a new way of building words in answer to that is an
    expensive way of being wrong.
    """
    from core.cognition.one_algebra import a_maker_she_wrote
    from core.cognition.what_the_failures_have_in_common import why_nothing_fits

    why = why_nothing_fits(situation.pairs)
    if not why.is_the_language:
        logger.info("not reaching for a new word: %s", why.describes())
        return None
    wrote = a_maker_she_wrote(situation.pairs, now_sayable=situation.sayable)
    if wrote is None:
        return None
    return f"a way of building words that I wrote ({wrote.name})"


def _a_way_of_computing_it(situation: _Situation) -> str | None:
    """A way of COMPUTING she writes, standing where a head stands.

    Everything below this is a term in the positional algebra, and
    core/cognition/what_the_old_language_cannot_say.py proves what those terms
    are bounded by: a polynomial in the length of the state whose degree is the
    length of the term, uniform over every vocabulary she can build. A family
    past that is not short of words.

    Refused unless it makes the family sayable, and taken out again when it
    does not, because a head is a branch at every step of every search from now
    on and that is the most expensive thing she can add.
    """
    return _a_way_of_computing(situation.pairs, situation.sayable)


def _a_rule_whose_shape_is_its_own(situation: _Situation) -> str | None:
    """A rule whose SHAPE is its own.

    Everything else, including a head she writes, fits inside one sentence
    somebody wrote — `after[i] = what(before[g(i,n)], before[h(i,n)])`. Two
    sources, one operation, both value-blind. A family wanting three sources,
    or an operation depending on where it is, or a value that was never in the
    state, is unsayable however wide the vocabulary gets, and
    language_limits.certify already refuses to rule on the last of those
    because no rule about where a cell came from can produce it.
    """
    return _a_rule_with_no_shape(situation.pairs)


def _the_ways_of_making_words_that_were_written_down(
    situation: _Situation,
) -> str | None:
    """The ones that were written down, because a way she can reach by writing
    is worth more than one she can only be handed."""
    from core.cognition.growing_at_any_level import grow_until_sayable, twice_over
    from core.cognition.widening_the_language import one_after_another

    kept = grow_until_sayable(
        [
            (1, "one after another", one_after_another),
            (2, "twice over", twice_over),
        ],
        now_sayable=situation.sayable,
    )
    if not kept:
        return None
    highest = max(one.level for one in kept)
    if highest >= 2:
        return (
            "a new way of making ways of making words ("
            + ", ".join(one.name for one in kept)
            + ")"
        )
    return f"a new way of MAKING words rather than a word ({kept[0].name})"


#: The eight, in the order they were written. That order still decides what
#: gets tried on a cold record, because a first occasion has nothing to read;
#: from the second onward it is decided by what each one has been worth. What
#: changed is not the order but the kind of thing that fixes it.
_THE_EIGHT: tuple[tuple[str, str, str, Any], ...] = (
    (
        "a kind of thing she had no name for",
        "the words",
        "a kind of thing",
        _a_kind_of_thing,
    ),
    (
        "a way of saying where a value comes from",
        "the words",
        "a way of saying",
        _a_way_of_saying_where_a_value_comes_from,
    ),
    (
        "a way of combining two values",
        "the words",
        "a way of combining",
        _a_way_of_combining_two_values,
    ),
    (
        "a way of making words she built",
        "the ways of building words",
        "a way of making words",
        _a_way_of_making_words_she_built,
    ),
    (
        "a way of building words she wrote",
        "the ways of building words",
        "a way of building words",
        _a_way_of_building_words_she_wrote,
    ),
    (
        "a way of computing",
        "the ways of computing",
        "a way of computing",
        _a_way_of_computing_it,
    ),
    (
        "a rule whose shape is its own",
        "the shapes a rule can have",
        "a rule with no shape",
        _a_rule_whose_shape_is_its_own,
    ),
    (
        "the ways of making words that were written down",
        "the ways of building words",
        "a way of making words",
        _the_ways_of_making_words_that_were_written_down,
    ),
)


def _register_what_she_could_do() -> None:
    """Put everything she could do in the registry, once.

    The eight that widen a language, the two that change how she searches and
    how she decides, and the three that change what she is made of. They are
    ranked together and none of them is tried first for being written first;
    what decides is what the record says each is worth.
    """
    from core.cognition.she_improves_her_own_deciding import (
        offer_what_she_can_do_about_herself,
    )
    from core.cognition.what_she_could_do_next import (
        WHAT_SHE_COULD_DO,
        what_she_could_do,
    )
    from core.cognition.what_she_does_about_herself import (
        offer_what_she_can_do_about_what_she_is_made_of,
    )

    for name, over, kind, do_it in _THE_EIGHT:
        if name not in WHAT_SHE_COULD_DO:
            what_she_could_do(name, over=over, kind=kind, do_it=do_it)
    # Bounded small, because these run on the path that answers a question and
    # a search that finds nothing has still spent the time. What they may
    # spend beyond this is decided by the ceiling, which is read off the
    # family rather than set here.
    offer_what_she_can_do_about_herself(within=4.0)
    offer_what_she_can_do_about_what_she_is_made_of()


def _what_family_this_is(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> str:
    """A key that recurs when the same shape of question recurs.

    The lengths asked about, and whether the values were moved or made. Two
    occasions of the same family share both; two families rarely do. Nothing
    about the family's NAME is in it, because a name would let the record be
    told what to count as recurrence rather than read it.
    """
    lengths = sorted({len(before) for before, _after in pairs})
    made = any(
        set(after) - set(before) for before, after in pairs
    )
    return f"{'made' if made else 'moved'} over {lengths}"


def _a_word_the_language_was_missing(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> str | None:
    """Decide what to do about a language that could not say this, and do it.

    This used to be a ladder: eight rungs, each running because the one above
    it returned nothing. That is not a decision, and the ladder was itself a
    hand-written account of what development can be.

    Now the eight are entries, and which one runs is whichever is worth the
    most — occasions ahead times what a change of that kind has saved before,
    less what it costs, less what it costs when it fails, all four read off
    `the_record_of_her_own_work`. On a cold record nothing has a history, so
    everything is unpriced and the order is the order they were written, which
    is exactly what this did before. From the second occasion it is decided by
    measurement.

    An action that is chosen and returns nothing is taken out of the running
    and the choice is made again, so a failure narrows the field rather than
    advancing a counter.
    """
    from core.cognition.an_invented_kind import (
        how_many_were_walked,
        induce_from,
        start_counting_again,
    )
    from core.cognition.she_decides_to_develop import what_to_do_next
    from core.cognition.the_record_of_her_own_work import note_an_episode
    from core.cognition.what_she_could_do_next import the_actions_she_has

    _register_what_she_could_do()
    sayable = lambda: induce_from(pairs) is not None
    situation = _Situation(
        pairs=tuple((tuple(before), tuple(after)) for before, after in pairs),
        sayable=sayable,
    )
    family = _what_family_this_is(pairs)

    # What the ordinary search spent failing on this. Measured, not estimated:
    # a search that finds nothing walks every meaning there is.
    start_counting_again()
    sayable()
    costs_now = max(1, how_many_were_walked())

    left = list(the_actions_she_has())
    while left:
        # The occasion is lost without a change: nothing she can say accounts
        # for these, which is why this function was called at all.
        decided = what_to_do_next(
            family, costs_now=costs_now, among=left, this_one_is_lost=True
        )
        if decided.action is None:
            logger.info("she is not developing: %s", decided.grounds)
            note_an_episode(family, route=None, walked=costs_now)
            return None
        try:
            said = decided.action.do_it(situation)
        except Exception:  # noqa: BLE001 - a failed action is a result
            logger.info("%s raised", decided.action.name, exc_info=True)
            said = None
        note_an_episode(
            family,
            route=decided.action.name if said else None,
            walked=decided.worth.cost if decided.worth else costs_now,
            admitted=decided.action.kind if said else None,
        )
        if said:
            logger.info(
                "she chose %s (%s) and it gave %s",
                decided.action.name,
                decided.because,
                said,
            )
            return said
        left = [one for one in left if one.name != decided.action.name]
    return None


def _the_body_inside(head_body: Any) -> Any:
    """A head's body with its binders off, so it can be a leaf in another."""
    inside = head_body
    while getattr(inside, "head", "") == "given a thing":
        inside = inside.parts[0]
    return inside


def _which_kind_of_growth_this_head_is(
    name: str,
    found: Any,
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> str:
    """Shorter name, longer reach, or a new distinction — decided, not assumed."""
    from core.cognition.an_invented_kind import addressings
    from core.cognition.one_algebra import DERIVED_HEADS, Head
    from core.cognition.one_algebra import _where_each_came_from  # noqa: PLC2701
    from core.cognition.what_an_invention_buys import the_horizon_of
    from core.cognition.which_kind_of_growth import UNDECIDED, which_kind_of_growth

    wanted = _where_each_came_from(pairs)
    if not wanted:
        return UNDECIDED

    def says_it(word: Any) -> bool:
        for size, places in wanted.items():
            if size <= 0:
                return False
            for at in range(size):
                try:
                    if int(word(at, size)) % size != places[at]:
                        return False
                except (ArithmeticError, IndexError, RecursionError, TypeError,
                        ValueError):
                    return False
        return True

    taken: Head | None = DERIVED_HEADS.pop(name, None)
    try:
        given = {
            one: word
            for one, word in addressings().items()
            if name not in one
        }
        return which_kind_of_growth(
            says_it,
            the_old_language=given,
            horizon=the_horizon_of(2),
            within=_HOW_LONG_A_CLASSIFICATION_GETS,
        ).kind
    finally:
        if taken is not None:
            DERIVED_HEADS[name] = taken


#: What deciding the kind may spend. It is a search over the language without
#: the head, and it runs once per admission rather than once per candidate.
_HOW_LONG_A_CLASSIFICATION_GETS = 8.0


def _a_way_of_computing(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    sayable: Any,
) -> str | None:
    """Write a head, install it, and keep it only if the family becomes sayable."""
    from core.cognition.a_way_of_computing_she_wrote import (
        a_way_of_computing_she_wrote,
    )
    from core.cognition.an_invented_kind import addressings
    from core.cognition.one_algebra import (
        DERIVED_HEADS,
        Head,
        Term,
        forget_the_head,
        the_head_she_wrote,
    )
    from core.cognition.which_kind_of_growth import A_SHORTER_NAME
    from core.cognition.widening_the_language import widen_with_addressing

    found = a_way_of_computing_she_wrote(
        pairs,
        now_sayable=sayable,
        # What she has already written, offered as leaves. A head unreachable
        # today because its pieces are missing is short tomorrow because they
        # are not, and that is the whole of what accumulating means here.
        #
        # The binders come off first. A head is stored closed over the six
        # things a head is given, and a closed head handed to the search as a
        # leaf is a function of six arguments where a number is wanted — so it
        # never fits anywhere and the library bought nothing. Measured: with
        # the binders on, a family needing a piece she already had came back
        # unsolved; with them off, the same family came back solved.
        already=tuple(_the_body_inside(one.body) for one in DERIVED_HEADS.values()),
    )
    if found is None:
        return None
    name = f"a way of computing she wrote ({len(DERIVED_HEADS)})"
    # What it costs the search, measured on both sides before it goes in.
    #
    # A word is one more thing to put in a hole. A head is one more shape at
    # every node of every term, so it multiplies rather than adds and it is
    # the most expensive thing she can admit. The maker search above walked
    # the positional space at this depth and found nothing; that is what the
    # head has to be cheaper than.
    from core.cognition.keeping_the_language_small import (
        what_a_head_costs_the_search,
    )
    from core.cognition.one_algebra import every_term

    def _how_many_terms() -> int:
        counted = 0
        for counted, _ in enumerate(  # noqa: B007
            every_term((0, 1, 2), holes=2, deepest=2), start=1
        ):
            pass
        return counted

    without = _how_many_terms()
    the_head_she_wrote(name, 2, found.body)
    worth = what_a_head_costs_the_search(
        name,
        without=without,
        with_it=_how_many_terms(),
        found_at=found.found_at,
        walked_without_finding=without,
    )
    if not worth.pays:
        logger.info("not keeping %s — %s", name, worth.describes())
        forget_the_head(name)
        return None

    # Which of the three things "the language grew" means, on this head.
    #
    # The control ChatGPT's response named and this codebase already had the
    # machinery for: take the name away and look for the behaviour in the
    # language without it. A head that merely spells something the positional
    # terms already say is a shorter name, and a shorter name is not worth a
    # shape at every node of every term.
    #
    # The head has to come out for the search, because every_term offers the
    # heads she wrote and the classifier would otherwise find this one and
    # report that the old language could say it all along.
    kind = _which_kind_of_growth_this_head_is(name, found, pairs)
    if kind == A_SHORTER_NAME:
        logger.info("not keeping %s — it is a shorter name for something sayable", name)
        forget_the_head(name)
        return None
    DERIVED_HEADS[name] = Head(
        name=name, takes=2, body=found.body, kind=kind
    )
    # A head is only reachable through a word written over it, so the word it
    # was fitted on goes in with it.
    every = addressings()
    over = Term(name, parts=(Term("hole", value=0), Term("hole", value=1)))
    from core.cognition.one_algebra import Made

    word = Made(
        term=over,
        words=tuple(every[one] for one in found.over if one in every),
        built_from=tuple(found.over),
    )
    if len(word.words) != 2:
        forget_the_head(name)
        return None
    said = widen_with_addressing(f"{name}, over {' and '.join(found.over)}", word)
    if said and sayable():
        return f"a way of COMPUTING that I wrote rather than had ({found.describes()})"
    forget_the_head(name)
    return None


def _a_rule_with_no_shape(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> str | None:
    """Write a rule with no fixed shape, and keep it where it can be used."""
    from core.cognition.a_rule_with_no_shape import (
        RULES_WITH_NO_SHAPE,
        a_rule_she_wrote,
    )

    def already_said() -> bool:
        return any(
            rule.read(before) == tuple(after)
            for rule in RULES_WITH_NO_SHAPE.values()
            for before, after in pairs
        )

    found = a_rule_she_wrote(pairs, now_sayable=already_said)
    if found is None:
        return None
    name = f"a rule she wrote ({len(RULES_WITH_NO_SHAPE)})"
    RULES_WITH_NO_SHAPE[name] = found
    made = " that makes values that were never there" if found.makes_new_values else ""
    return f"a rule with no shape but its own{made} ({found.describes()})"


def read_sequence_question(text: Any) -> SequenceQuestion | None:
    """The examples and the question, or None when this is not one.

    Structural: an odd number of sequences, at least three, the last being the
    one asked about and the rest pairing off. Nothing here reads the words
    around them beyond checking that a pair is joined by something meaning
    "becomes", so a list of unrelated sequences is not mistaken for examples.
    """

    body = str(text or "")
    if not body:
        return None
    found: list[tuple[int, tuple[Any, ...]]] = []
    covered: list[tuple[int, int]] = []
    for hit in _SEQUENCE.finditer(body):
        cells = _cells(hit.group(1))
        if cells is not None:
            found.append((hit.start(), cells))
            covered.append((hit.start(), hit.end()))
    for hit in _BARE_RUN.finditer(body):
        if any(start <= hit.start() < end for start, end in covered):
            # Already read as a bracketed run. Counting it twice would make an
            # even number of runs out of an odd one and lose the question.
            continue
        cells = _cells(hit.group(1).replace(" ", ", "))
        if cells is not None:
            found.append((hit.start(), cells))
    found.sort()
    if len(found) < 3 or len(found) % 2 == 0:
        return None
    *pairs, last = found
    shown: list[Transition] = []
    for index in range(0, len(pairs), 2):
        (start_a, before), (start_b, after) = pairs[index], pairs[index + 1]
        between = body[start_a:start_b]
        if not _BECOMES.search(between):
            return None
        if len(after) > len(before):
            # Cells appearing from nowhere is not something anything here can
            # say. Fewer is fine now and was not: every filter was thrown out
            # by this line before it reached inference, so the one family that
            # CHANGES the length could never be read as a question about
            # length.
            return None
        shown.append(Transition(before, after))
    if not shown:
        return None
    return SequenceQuestion(shown=tuple(shown), asked=last[1])


def _language_path() -> Path | None:
    try:
        from core.runtime.state_ownership import state_root

        return Path(state_root()) / "relation_language.json"
    except (ImportError, AttributeError, OSError, TypeError, ValueError):
        return None


def _language() -> RelationLanguage:
    """The shapes earlier turns worked out, or an empty language."""

    target = _language_path()
    if target is None:
        return RelationLanguage()
    return RelationLanguage.load(target)


def answer_sequence_question(text: Any) -> str:
    """The answer, what it cost, and a decision about whether that was too much.

    The answer is whatever `_the_sequence_answer` gives and is not changed here.
    What is here is the half of development a failure-driven ladder cannot
    have: the search WORKED, and it may still have been dear enough to be worth
    doing something about. Every answering path is wrapped, not only the one
    that induces a meaning, because what a change is worth is read off what
    answering costs and most answers do not come from that path.
    """
    from core.cognition.an_invented_kind import (
        how_many_were_walked,
        start_counting_again,
    )
    from core.cognition.the_record_of_her_own_work import note_an_episode

    question = read_sequence_question(text)
    if question is None:
        return ""
    start_counting_again()
    said = _the_sequence_answer(text)
    walked = max(0, how_many_were_walked())
    if not said:
        return said
    pairs = [(one.before, one.after) for one in question.shown]
    family = _what_family_this_is(pairs)
    note_an_episode(family, route="an answer", walked=walked, about=pairs)
    _she_may_improve_a_working_answer(pairs, family, walked)
    return said


def _the_sequence_answer(text: Any) -> str:
    """The answer and the rule behind it, or "" when there is nothing to say.

    The shape is kept afterwards, so a later question of the same kind is
    settled from fewer examples. A library beside the live path cannot do that.
    """

    question = read_sequence_question(text)
    if question is None:
        return ""
    language = _language()
    found = language.explain(list(question.shown))

    # A shape already learned, where the observations do not pin the positional
    # answer.
    #
    # LIVE, 2026-08-28: a turn taught "ascending order", and the next one,
    # showing a single example, was answered "position i takes from i+1 (mod
    # n)". The answer was right by luck — a rotation and an ordering agree on
    # that one state — and the ordering that had just been learned was never
    # consulted, because the positional path had found something and something
    # was enough.
    #
    # A form fitting one example is the thin evidence the probe exists to flag.
    # Where a rival survives, a shape that has accounted for a world before
    # beats one that has accounted for this one.
    #
    # Thin means what it says: one example, or more than one positional form
    # still fitting. The rival here is the ORDERING, and the probe only ever
    # compared positional forms to each other — so it saw no rival on a world
    # whose only rival was of the other kind, and reported the evidence as
    # settled when it was one example.
    thin = (
        len(question.shown) < 2
        or discriminating_probe(list(question.shown), known_forms=language.forms)
        is not None
    )
    known = language.order_that_explains(list(question.shown))
    if known is not None and (found is None or thin):
        answer = known.apply(tuple(question.asked))
        if answer is not None:
            return (
                f"{list(answer)}\n\n"
                f"The rule, from a shape worked out earlier: {known.describe()}."
            )

    if found is None:
        # Two failures wore the same face. A world one example short of being
        # settled and a world no rule of this shape can ever say both returned
        # nothing, so neither could be answered honestly and neither could be
        # acted on.
        verdict = certify(list(question.shown))
        if verdict.proven_outside:
            # The proof says a rule reading only positions cannot do this. That
            # is the one place it is right to look at the cells: a wider net is
            # offered where the language is PROVEN to fail, never beside it. A
            # mirror is explained by descending order just as well, and letting
            # that compete would lose the simpler answer that was already
            # right.
            # Something already known first. That is what makes the second
            # question of a kind cheaper than the first.
            ordering = solve_ordering(list(question.shown))
            if ordering is None or ordering.apply(tuple(question.asked)) is None:
                # An ordering and a move, where neither alone says it.
                #
                # "Sorted, then rotated" is proved outside the positional
                # language — the sources genuinely contradict — and the
                # ordering alone cannot say it either, because the cells do not
                # come out in the order the values carry. The two axes were
                # solved separately and had no way to meet.
                composed = solve_ordering_then_move(
                    list(question.shown), _index_forms(len(question.asked))
                )
                if composed is not None:
                    answer = composed.apply(tuple(question.asked))
                    if answer is not None:
                        language.admit_order(composed)
                        target = _language_path()
                        if target is not None:
                            language.path = target
                            language.save()
                        return (
                            f"{list(answer)}\n\n"
                            f"The rule, worked out from the examples: "
                            f"{composed.describe()}."
                        )
            if ordering is not None:
                answer = ordering.apply(tuple(question.asked))
                if answer is not None:
                    language.admit_order(ordering)
                    target = _language_path()
                    if target is not None:
                        language.path = target
                        language.save()
                    return (
                        f"{list(answer)}\n\n"
                        f"The rule, worked out from the examples: "
                        f"{ordering.describe()}."
                    )
                # The rule was worked out and this case is outside what it
                # covers. Saying no rule exists would be the wrong reason, and
                # a wrong reason is worse than no reason: it sends the person
                # looking for better examples of the wrong thing.
                return (
                    "I worked the rule out — "
                    f"{ordering.describe()} — and I still cannot answer this "
                    "one.\n\n"
                    "The order came from the cells you showed me, and "
                    f"{list(question.asked)} holds cells that were not among "
                    "them, so I have nothing that says where they go. Ask me "
                    "about cells from your examples and I can, or show me one "
                    "more example using these."
                )
            # Before saying she cannot: work out a MEANING from the examples.
            #
            # Everything above searches for a member of a language whose
            # evaluation rules were written down. When the proof says no member
            # of it can do this, the honest next move is not an apology — it is
            # to ask whether the examples themselves describe an operation, and
            # if they do, to give that operation a name and an executable body
            # and go on using it.
            #
            # This is the only production caller of that, and it belongs here:
            # at the point where the language is PROVEN insufficient, never
            # beside a language that still might explain it. A wider net
            # offered earlier would lose the simpler answer that was already
            # right, which is the same reason the ordering search sits behind
            # the same proof.
            said = _a_meaning_worked_out(question)
            if said is not None:
                return said

            return (
                "I cannot work this one out, and I can say why rather than "
                "just that.\n\n"
                f"{verdict.reason.capitalize()}.\n\n"
                "Every rule I can form here says where a cell comes from using "
                "its position and the length, never what the cells hold. "
                "Composing those only ever makes another one of them, so no "
                "amount of looking would find it — the rule you have in mind "
                "reads the values themselves, and that is a kind of rule I "
                "have no way to write."
            )
        # Nothing found, and nothing proved either way.
        #
        # The third state, and the commonest one. The positional proof needs
        # the cells to stay put to say anything, so a family that CHANGES the
        # values is neither settled nor refuted by it — and this exit returned
        # an empty string, which the caller shows as nothing at all.
        #
        # "I have not proved I cannot" is not a reason to stop looking. It is
        # the reason to look somewhere the proof does not reach: at what the
        # examples are doing to the values themselves.
        said = _a_meaning_worked_out(question)
        if said is not None:
            return said
        return ""
    try:
        result = tuple(found.apply(tuple(question.asked)))
    except Exception:  # noqa: BLE001 - a relation that throws has not answered
        return ""
    if len(result) > len(question.asked):
        return ""

    # What the evidence did not settle is not learned.
    #
    # A rival that fits everything shown and disagrees about what comes next
    # means the observations chose no rule. Saying so and then admitting one of
    # them anyway says two different things, and the admitted one is what
    # steers the next question — so the saying was decoration. This is the same
    # discipline the induced-meaning path follows a few hundred lines up.
    probe = discriminating_probe(list(question.shown), known_forms=language.forms)
    # How many fit, not whether one does.
    #
    # The probe reports every rival and the reply named the first of them, so a
    # question with four surviving rules was answered "X fits just as well" —
    # true of X, and quietly false about the size of what she had not decided.
    # The count is the whole fact about the evidence, and it is free.
    rivals = [
        text for text, _rule in (probe.rivals if probe is not None else ()) if text != found.form
    ]
    rival_form = rivals[0] if rivals else ""
    rival_says = (
        _what_a_form_says(rival_form, tuple(question.asked), language)
        if rival_form
        else None
    )
    also = "" if len(rivals) < 2 else f", and {len(rivals) - 1} more"

    unsettled = bool(rival_form)
    if not unsettled:
        language.admit(found)
        language.refactor()
        target = _language_path()
        if target is not None:
            language.path = target
            language.save()

    if unsettled and rival_says is not None and tuple(rival_says) != tuple(result):
        # They disagree about THIS case. Any answer would be a guess dressed as
        # a rule.
        return (
            "I cannot answer this one yet, and it is worth saying why.\n\n"
            f"{found.form.capitalize()} and {rival_form}{also} account for every "
            "example you showed me, and they disagree about this case: one says "
            f"{list(result)} and the other {list(rival_says)}."
            f"{_which_example_parts_the_forms(probe, found, question, language)}"
        )

    said = (
        f"{list(result)}\n\n"
        f"The rule, worked out from the examples: {found.form}."
    )

    # Whether anything else fits equally well is a fact about the evidence, not
    # a hedge about the answer. On thin observations the rule was stated with
    # the same confidence either way: one example of (1,2,3) becoming (3,2,1)
    # is a mirror and is just as much an exchange of the ends, and only the
    # first was ever said.
    if unsettled:
        said += (
            f"\n\n{rival_form.capitalize()}{also} fits everything you showed just "
            f"as well, and says the same about this one — so I could answer it "
            f"without deciding between them, and I have not."
            f"{_which_example_parts_the_forms(probe, found, question, language)}"
        )
    return said


def _which_example_parts_the_forms(
    probe: Any, found: Any, question: Any, language: Any
) -> str:
    """The example to ask for, weighed against every rival form rather than one.

    The probe hands back a state that parts the two forms it happened to
    compare. Where five forms survive that state settles one bit and leaves the
    rest, and she would have to ask again.
    """
    forms = {found.form: found.form}
    for text, _rule in getattr(probe, "rivals", ()) or ():
        forms[text] = text
    return _the_one_thing_worth_asking_him(
        forms,
        len(question.asked),
        lambda form, state: _what_a_form_says(form, state, language),
    )


def _what_a_form_says(
    form: str, cells: tuple[Any, ...], language: Any
) -> tuple[Any, ...] | None:
    """What a rule of that form would make of these cells, or nothing."""
    from core.cognition.primitive_invention import rule_for_description

    rule = (getattr(language, "rules", {}) or {}).get(form) or rule_for_description(form)
    if rule is None:
        return None
    try:
        size = len(cells)
        return tuple(cells[rule(index, size)] for index in range(size))
    except (IndexError, TypeError, ValueError, AttributeError):
        return None
