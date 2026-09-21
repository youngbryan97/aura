"""Measured sequence reach through retained meanings and the existing inducer.

One candidate must explain every pair. Reuse and induction report both the
executable witness and work spent; an exhausted search is not a cheap solution.
This measures agreement with supplied examples, not arbitrary-domain transfer.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from core.verify.invariants import Violation, invariant


@dataclass(frozen=True)
class SequenceMeaning:
    name: str
    reader: Callable[[tuple[Any, ...]], Any]

    def read(self, cells: Sequence[Any]) -> tuple[Any, ...] | None:
        try:
            answer = self.reader(tuple(cells))
        except (ArithmeticError, TypeError, ValueError, RuntimeError, RecursionError):
            return None
        return tuple(answer) if isinstance(answer, (tuple, list)) else None


@dataclass(frozen=True)
class SequenceReach:
    meanings: tuple[SequenceMeaning, ...]
    walked: int

    @property
    def solved(self) -> bool:
        return bool(self.meanings)

    def agreed_on(self, cells: Sequence[Any]) -> tuple[Any, ...] | None:
        answers = [meaning.read(cells) for meaning in self.meanings]
        if not answers or answers[0] is None:
            return None
        return answers[0] if all(answer == answers[0] for answer in answers) else None


def retained_sequence_reach(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> SequenceReach:
    """Check installed meanings using their existing execution contracts."""
    from core.cognition.a_rule_with_no_shape import RULES_WITH_NO_SHAPE
    from core.cognition.an_invented_kind import KINDS
    from core.cognition.an_operator_she_invents import the_kernel
    from core.cognition.operator_invention import _the_term_as_a_function
    from core.cognition.the_floor_she_stands_on import Code
    from core.cognition.the_record_of_her_own_work import note_a_step

    pairs = tuple((tuple(before), tuple(after)) for before, after in transitions)
    if not pairs:
        return SequenceReach((), 0)
    readers = [SequenceMeaning(name, meaning.read) for name, meaning in list(KINDS.items())]
    readers.extend(SequenceMeaning(name, rule.read) for name, rule in list(RULES_WITH_NO_SHAPE.items()))
    readers.extend(
        SequenceMeaning(name, _the_term_as_a_function(operator.term))
        for name, operator in the_kernel().operators().items()
        if operator.invented and isinstance(operator.term, Code)
    )
    matching = []
    for meaning in readers:
        note_a_step()
        if all(meaning.read(before) == after for before, after in pairs):
            matching.append(meaning)
    return SequenceReach(tuple(matching), len(readers))


def measure_sequence_reach(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> SequenceReach:
    """Reuse retained meanings first, otherwise run the existing induction."""
    from core.cognition.an_invented_kind import (
        how_many_were_walked,
        induce_from,
        start_counting_again,
    )

    start_counting_again()
    retained = retained_sequence_reach(transitions)
    if retained.solved or not transitions:
        return retained
    meaning = induce_from(transitions)
    meanings = () if meaning is None else (SequenceMeaning(meaning.name, meaning.read),)
    return SequenceReach(meanings, how_many_were_walked())


def reach_utility(measurements: Sequence[tuple[int, bool]]) -> float:
    """Solved count dominates all possible search-cost savings on this cohort.

    U = solved + sum(solved / (1 + cost)) / (n + 1). The entire cost term
    lies below one, so losing a solved family cannot increase U.
    """
    if any(cost < 0 for cost, _ in measurements):
        raise ValueError("search work cannot be negative")
    return sum(solved for _, solved in measurements) + sum(
        1 / (1 + cost) for cost, solved in measurements if solved
    ) / (len(measurements) + 1)


@invariant("cognition.sequence_reach_correctness_first", scope="cognition", owner=__name__)
def reach_preserves_correctness() -> list[Any]:
    """No finite search saving outweighs a solved-family loss in the canary."""
    for size in (1, 2, 10, 100):
        if reach_utility([(10**12, True)] * size) <= reach_utility(
            [(0, True)] * (size - 1) + [(0, False)]
        ):
            return [Violation(subject="sequence reach", message="cost savings outweighed a solved-family loss")]
    return []
