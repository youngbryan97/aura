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

from core.cognition.primitive_invention import Transition, invent_relation
from core.cognition.relation_language import RelationLanguage

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
        if len(before) != len(after):
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
    settled from fewer examples — which is the whole point of the library being
    in the live path rather than beside it.
    """

    question = read_sequence_question(text)
    if question is None:
        return ""
    language = _language()
    found = language.explain(list(question.shown))
    if found is None:
        return ""
    try:
        result = tuple(found.apply(tuple(question.asked)))
    except Exception:  # noqa: BLE001 - a relation that throws has not answered
        return ""
    if len(result) != len(question.asked):
        return ""

    language.admit(found)
    language.refactor()
    target = _language_path()
    if target is not None:
        language.path = target
        language.save()

    return (
        f"{list(result)}\n\n"
        f"The rule, worked out from the examples: {found.form}."
    )
