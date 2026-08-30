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
import re
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.cognition.language_limits import certify
from core.cognition.primitive_invention import (
    Transition,
    _index_forms,
    discriminating_probe,
)
from core.cognition.relation_language import RelationLanguage
from core.cognition.value_order import solve_ordering, solve_ordering_then_move

__all__ = ["SequenceQuestion", "answer_sequence_question", "read_sequence_question"]

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
        other = next((one for one in fits if one.name != meaning.name), fits[0])
        telling = what_would_tell_them_apart(
            meaning, other, of_length=len(question.asked)
        )
        settle = ""
        if telling is not None:
            settle = (
                f" Show me what {list(telling)} becomes and I will know which: "
                f"{meaning.name} says {list(meaning.read(telling) or ())} and "
                f"{other.name} says {list(other.read(telling) or ())}."
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


def _a_word_the_language_was_missing(
    pairs: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> str | None:
    """Derive a way of saying what these do, and add it to the language.

    Returns what it is called, or nothing when the examples do not determine
    one — values that repeat have no single source, and two examples of one
    length that disagree describe no correspondence at all.
    """
    from core.cognition.an_invented_kind import WHERE_FROM, addressings
    from core.cognition.widening_the_language import (
        an_addressing_nobody_wrote,
        widen_with_addressing,
    )

    # Everything constructible, not merely the words written down. A candidate
    # that some composition already produces is new spelling for an old
    # meaning, and admitting it would report growth that did not happen.
    closure = addressings()
    found = an_addressing_nobody_wrote(pairs, already=closure)
    if found is not None:
        name = f"the way these move ({len(WHERE_FROM)})"
        said = widen_with_addressing(name, found)
        return f"a new way of saying where a value comes from ({said})" if said else None

    # Nothing was MOVED, so no addressing describes it: the values were made
    # rather than taken. Then what is missing is a way of combining a pair, and
    # that is derivable too — given where the two came from, what was done with
    # them is whatever the result was.
    from core.cognition.an_invented_kind import WHAT_OF_IT
    from core.cognition.widening_the_language import (
        an_operation_nobody_wrote,
        widen_with_operation,
    )

    for first in list(closure.values()):
        for second in list(closure.values()):
            done = an_operation_nobody_wrote(
                pairs, first, second, already=list(WHAT_OF_IT.values())
            )
            if done is None:
                continue
            name = f"what was done with these ({len(WHAT_OF_IT)})"
            said = widen_with_operation(name, done)
            return f"a new way of combining two values ({said})" if said else None

    # No new WORD is enough. Then what is missing is a way of MAKING words.
    #
    # Tried last, because it enlarges the search enormously and every
    # hypothesis it adds is another chance for a coincidence to win. But it is
    # the only one of the three that helps with a family she has not met: a
    # derived word is read off what she was shown and says nothing beyond it,
    # while a way of building takes every word she has — including the derived
    # ones — and makes more out of them.
    from core.cognition.an_invented_kind import induce_from
    from core.cognition.a_constructor_she_built import a_constructor_she_built
    from core.cognition.growing_at_any_level import grow_until_sayable, twice_over
    from core.cognition.widening_the_language import one_after_another

    sayable = lambda: induce_from(pairs) is not None

    # Build one before reaching for one that was written. Activating a
    # constructor somebody already wrote enlarges what she uses and never what
    # she has; a recipe she composes can describe something the source does not
    # contain.
    built = a_constructor_she_built(pairs, now_sayable=sayable)
    if built is not None:
        return f"a way of making words that I built rather than had ({built.name})"

    # A way of building she WRITES, rather than one she is handed.
    #
    # This used to offer two makers by name — "one after another" and "twice
    # over" — so the levels had no ceiling and the thing supplying candidates
    # for them did. Whatever she reached was inside the closure of
    # composition, inversion and iteration, and a family needing anything else
    # was unreachable however long she looked.
    #
    # A way of building is now a TERM with a hole in it, in the same algebra a
    # word is a term in, so there is no list to be at the end of. Measured: the
    # first family that branches on the size of the thing is served by a maker
    # she wrote that is provably outside the closure of those three.
    from core.cognition.one_algebra import a_maker_she_wrote

    wrote = a_maker_she_wrote(pairs, now_sayable=sayable)
    if wrote is not None:
        return f"a way of building words that I wrote ({wrote.name})"

    # And the ones that were written down, tried after, because a way she can
    # reach by writing is worth more than one she can only be handed.
    kept = grow_until_sayable(
        [
            (1, "one after another", one_after_another),
            (2, "twice over", twice_over),
        ],
        now_sayable=sayable,
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
    rival_form = ""
    rival_says: tuple[Any, ...] | None = None
    if probe is not None:
        for text, _rule in probe.rivals:
            if text == found.form:
                continue
            rival_form = text
            rival_says = _what_a_form_says(text, tuple(question.asked), language)
            break

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
            f"{found.form.capitalize()} and {rival_form} both account for every "
            "example you showed me, and they disagree about this case: one says "
            f"{list(result)} and the other {list(rival_says)}. "
            f"Show me what {list(probe.state)} becomes and I will know which."
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
            f"\n\n{rival_form.capitalize()} fits everything you showed just as "
            f"well, and says the same about this one — so I could answer it "
            f"without deciding between them, and I have not. "
            f"{list(probe.state)} would tell them apart."
        )
    return said


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
