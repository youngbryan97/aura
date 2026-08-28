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
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.cognition.language_limits import certify
from core.cognition.primitive_invention import (
    Transition,
    discriminating_probe,
    invent_relation,
)
from core.cognition.relation_language import RelationLanguage
from core.cognition.value_order import solve_ordering

__all__ = ["SequenceQuestion", "answer_sequence_question", "read_sequence_question"]

#: A bracketed run of comma-separated cells. Deliberately not a bare comma
#: sequence: prose is full of commas, and a wrong reading here would answer a
#: question nobody asked.
_SEQUENCE = re.compile(r"[\[(]\s*([^\[\]()]{1,300}?)\s*[\])]")

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
    for hit in _SEQUENCE.finditer(body):
        cells = _cells(hit.group(1))
        if cells is not None:
            found.append((hit.start(), cells))
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
    if found is None:
        # Two failures wore the same face. A world one example short of being
        # settled and a world no rule of this shape can ever say both returned
        # nothing, so neither could be answered honestly and neither could be
        # acted on.
        verdict = certify(list(question.shown))
        # An ordering already learned is consulted whenever the positional
        # language came up empty, not only where it is proved impossible. The
        # proof needs two states of one length; a person showing one example of
        # something they have shown before should not have to.
        known = language.order_that_explains(list(question.shown))
        if known is not None:
            answer = known.apply(tuple(question.asked))
            if answer is not None:
                return (
                    f"{list(answer)}\n\n"
                    f"The rule, from a shape worked out earlier: "
                    f"{known.describe()}."
                )
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
        return ""
    try:
        result = tuple(found.apply(tuple(question.asked)))
    except Exception:  # noqa: BLE001 - a relation that throws has not answered
        return ""
    if len(result) > len(question.asked):
        return ""

    language.admit(found)
    language.refactor()
    target = _language_path()
    if target is not None:
        language.path = target
        language.save()

    said = (
        f"{list(result)}\n\n"
        f"The rule, worked out from the examples: {found.form}."
    )

    # Whether anything else fits equally well is a fact about the evidence, not
    # a hedge about the answer. On thin observations the rule was stated with
    # the same confidence either way: one example of (1,2,3) becoming (3,2,1)
    # is a mirror and is just as much an exchange of the ends, and only the
    # first was ever said.
    #
    # Named only when a rival exists AND disagrees somewhere reachable, so a
    # world the observations actually pin says nothing extra.
    probe = discriminating_probe(list(question.shown), known_forms=language.forms)
    if probe is not None:
        rival = next(
            (text for text, _r in probe.rivals if text != found.form), None
        )
        if rival:
            said += (
                f"\n\n{rival.capitalize()} fits everything you showed just as "
                f"well. {list(probe.state)} would tell them apart."
            )
    return said
