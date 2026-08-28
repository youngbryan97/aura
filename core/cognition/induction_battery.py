"""A score for induction with no language model in the process.

The standing criticism is that when the resident foundation model is taken away,
what remains gets sharply narrower. That is a claim about a number, and there
was no number. This produces one.

Every problem here is a world: some transitions, and a held-out transition the
answer has to predict. Solving one means working out the relation from the
observations and getting the held-out case right. Nothing in the path consults
a model, an embedding, or a stored answer — it is arithmetic on tuples.

Held out from the author
------------------------
The generator composes shape x representation x length, and the representations
are deliberately ones the mechanism was not written against: strings, colours,
records, and grids whose cells are themselves tuples. The mechanism was written
for sequences of integers. If the shapes are structural rather than numeric
then the representation should not matter, and if that is wrong the score says
so rather than the docstring.

The battery is frozen in the sense that matters: the generator is deterministic
from a seed, the problems do not change when the mechanism does, and failures
are counted rather than removed.
"""

from __future__ import annotations

import hashlib
import random
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from typing import Any

from core.cognition.language_limits import certify
from core.cognition.primitive_invention import Transition, invent_relation
from core.cognition.relation_language import RelationLanguage
from core.cognition.value_order import solve_ordering

__all__ = [
    "Problem",
    "Report",
    "battery_fingerprint",
    "generate_battery",
    "score_battery",
]


@dataclass(frozen=True)
class Problem:
    """Observations of one world, and a case the answer must predict."""

    name: str
    shown: tuple[Transition, ...]
    held_out: Transition
    shape: str
    representation: str


@dataclass
class Report:
    """What the mechanism scored, and on what."""

    solved: int = 0
    attempted: int = 0
    #: Solved among the shapes the language can express at all. Reported apart
    #: from the total, because a shape it cannot say is not a shape it failed.
    solved_expressible: int = 0
    attempted_expressible: int = 0
    by_shape: dict[str, tuple[int, int]] = field(default_factory=dict)
    by_representation: dict[str, tuple[int, int]] = field(default_factory=dict)
    missed: list[str] = field(default_factory=list)

    @property
    def score(self) -> float:
        return (self.solved / self.attempted) if self.attempted else 0.0

    @property
    def score_expressible(self) -> float:
        if not self.attempted_expressible:
            return 0.0
        return self.solved_expressible / self.attempted_expressible

    def line(self) -> str:
        return f"{self.solved}/{self.attempted} ({self.score:.0%})"


# ------------------------------------------------------------- the material
#
# Values of a kind the mechanism was not written for. Order matters and
# nothing else does, which is the point: a structural shape should not care
# what the cells contain.

_ALPHABET = ("alpha", "bravo", "charlie", "delta", "echo", "foxtrot", "golf", "hotel")
_COLOURS = ("red", "amber", "green", "blue", "violet", "grey", "white", "black")


def _integers(n: int, seed: int) -> tuple[Any, ...]:
    start = (seed % 17) * 3
    return tuple(start + step for step in range(n))


def _words(n: int, seed: int) -> tuple[Any, ...]:
    return tuple(_ALPHABET[(seed + i) % len(_ALPHABET)] for i in range(n))


def _colours(n: int, seed: int) -> tuple[Any, ...]:
    return tuple(_COLOURS[(seed * 3 + i) % len(_COLOURS)] for i in range(n))


def _records(n: int, seed: int) -> tuple[Any, ...]:
    return tuple(
        (_ALPHABET[(seed + i) % len(_ALPHABET)], (seed + i) % 5) for i in range(n)
    )


def _rows(n: int, seed: int) -> tuple[Any, ...]:
    """A grid, as rows. Each cell is itself a tuple, so the state is nested."""

    return tuple(tuple((seed + i + j) % 7 for j in range(3)) for i in range(n))


_REPRESENTATIONS: dict[str, Callable[[int, int], tuple[Any, ...]]] = {
    "integers": _integers,
    "words": _words,
    "colours": _colours,
    "records": _records,
    "grid rows": _rows,
}


# --------------------------------------------------------------- the shapes
#
# Written as what they DO to a state, so the generator and the mechanism share
# no code: the mechanism has to work them out from before-and-after alone.


def _mirror(state: tuple[Any, ...]) -> tuple[Any, ...]:
    return tuple(reversed(state))


def _rotate(k: int) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    def apply(state: tuple[Any, ...]) -> tuple[Any, ...]:
        if not state:
            return state
        step = k % len(state)
        return state[step:] + state[:step]

    return apply


def _exchange(i: int, j: int) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    def apply(state: tuple[Any, ...]) -> tuple[Any, ...]:
        if max(i, j) >= len(state):
            return state
        row = list(state)
        row[i], row[j] = row[j], row[i]
        return tuple(row)

    return apply


def _identity(state: tuple[Any, ...]) -> tuple[Any, ...]:
    return state


def _then(
    first: Callable[[tuple[Any, ...]], tuple[Any, ...]],
    second: Callable[[tuple[Any, ...]], tuple[Any, ...]],
) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    """One shape applied after another, which is a shape the solver has to find."""

    def apply(state: tuple[Any, ...]) -> tuple[Any, ...]:
        return second(first(state))

    return apply


def _every_other(state: tuple[Any, ...]) -> tuple[Any, ...]:
    """Odd positions before even ones. Not a form the solver can express."""

    return state[1::2] + state[0::2]


def _sorted_by_repr(state: tuple[Any, ...]) -> tuple[Any, ...]:
    """Reordered by the cells themselves, which is not a rule over positions."""

    return tuple(sorted(state, key=repr))


#: What the battery asks. The first six are shapes the solver can express and
#: the score on them is the score on its own ground. The compositions are
#: shapes it must find without either half being given. The last two are
#: outside what it can express at all, and are here so the number means
#: something: a battery a mechanism cannot fail is not a measurement of it.
def _sorted_by_a_secret_key(state: tuple[Any, ...]) -> tuple[Any, ...]:
    """Ordered by a property of each cell that nothing can recover.

    The battery needs a shape it cannot do or it measures nothing. Ordering by
    the values themselves stopped being that once the ordering could be solved
    for, so here is one that is still out of reach: the key is a hash of the
    cell, so it is a genuine function of the value — the right KIND of rule —
    and it is not the order the values carry.

    The score on this is not expected to be zero, and saying it should be would
    be the dishonest version. Where the held-out state happens to hold only
    values the examples showed, a table over those values genuinely answers it,
    and a table over values that were actually observed is a real thing that
    was really learned. What cannot be done is the case that matters: a value
    never shown has no place in a learned order, and the mechanism refuses
    rather than putting it somewhere.

    So this measures how far a table reaches, and the part that is out of reach
    stays out of reach. Both numbers are recorded.
    """

    return tuple(sorted(state, key=lambda cell: hashlib.sha256(repr(cell).encode()).hexdigest()))


_SHAPES: dict[str, Callable[[tuple[Any, ...]], tuple[Any, ...]]] = {
    "mirror": _mirror,
    "rotate by one": _rotate(1),
    "rotate by two": _rotate(2),
    "exchange the ends": _exchange(0, -1),
    "exchange the middle": _exchange(1, 2),
    "identity": _identity,
    "mirror then rotate": _then(_mirror, _rotate(1)),
    "rotate then exchange the ends": _then(_rotate(2), _exchange(0, -1)),
    "odd positions first": _every_other,
    "reordered by the cells": _sorted_by_repr,
    "reordered by a secret key": _sorted_by_a_secret_key,
}

#: Shapes nothing in the solver can say. Named so a report can separate "got it
#: wrong" from "was never able to say it", which are different facts.
#:
#: "Odd positions first" was here and is not any more, and how it left is the
#: point. The basis had order, symmetry and adjacency and nothing for cells
#: BELONGING together — the one core-knowledge system of the four that applies
#: here which was missing — so the prediction was made in advance that adding
#: grouping would lift exactly this shape and nothing else. It lifted nothing at
#: first, because the form as written laid the even class down first and could
#: not say the other order; with that fixed the shape went 0/10 to 10/10 and no
#: other shape moved.
#:
#: What is left needs cells to be objects with properties that can be ordered,
#: which is objecthood in a second sense and a different primitive.
#: Shapes no rule reading only positions can express. The proof in
#: core.cognition.language_limits fires on these and on nothing else.
BEYOND_THE_LANGUAGE = frozenset(
    {"reordered by the cells", "reordered by a secret key"}
)

#: Ordered by a genuine property of the cells that is not the order the cells
#: carry, so only a table over values actually seen can answer it.
#:
#: "Reordered by the cells" left the unreachable set on 2026-08-28: the
#: ordering behind it can be solved for from the observations rather than
#: searched for, so it is expressible now and it was not before. A battery
#: whose unreachable shape becomes reachable has stopped measuring anything,
#: so this went in with the same edit.
ONLY_A_TABLE_REACHES = frozenset({"reordered by a secret key"})

#: Shapes three transformations deep. Unreachable from the basis alone however
#: many observations are offered, and reachable once a two-deep shape has been
#: learned somewhere else.
#:
#: These exist because the ablation said so. With only the shapes above, turning
#: OFF the learned library and the prior changed the score by nothing: every
#: problem showed the same shape at two lengths, which pins it without help, so
#: the battery measured induction and not transfer while being described as
#: measuring both. A component that cannot change a score is not being measured
#: by it.
_DEEP: dict[str, Callable[[tuple[Any, ...]], tuple[Any, ...]]] = {
    "three deep: mirror, rotate, ends": _then(
        _then(_mirror, _rotate(1)), _exchange(0, -1)
    ),
    "three deep: rotate, ends, mirror": _then(
        _then(_rotate(2), _exchange(0, -1)), _mirror
    ),
}

#: What has to be learned first for a deep shape to be reachable at all.
_TEACHES: dict[str, Callable[[tuple[Any, ...]], tuple[Any, ...]]] = {
    "three deep: mirror, rotate, ends": _then(_mirror, _rotate(1)),
    "three deep: rotate, ends, mirror": _then(_rotate(2), _exchange(0, -1)),
}

NEEDS_A_LEARNED_FORM = frozenset(_DEEP)


def battery_fingerprint(*, seed: int = 20260828, per_cell: int = 2) -> str:
    """A hash of the problems themselves, so a changed battery is visible.

    The score is only evidence while the problems are fixed. Whoever owns the
    generator can raise the number by making the problems easier, and would not
    have to mean to: widening the solver's basis and widening its battery are
    two edits in the same file. The literature on measuring self-improvement is
    largely about this — held-out sets that leak, evaluators that can be gamed —
    so the floor records what it was measured on and not only what was scored.
    """

    import hashlib

    digest = hashlib.sha256()
    for problem in generate_battery(seed=seed, per_cell=per_cell):
        digest.update(problem.name.encode())
        digest.update(repr(problem.shown).encode())
        digest.update(repr(problem.held_out).encode())
    return digest.hexdigest()[:16]


def generate_battery(*, seed: int = 20260828, per_cell: int = 2) -> list[Problem]:
    """The frozen set: every shape, in every representation, at several lengths.

    Deterministic from the seed, so the battery is the same set of problems
    before and after any change to the mechanism.
    """

    rng = random.Random(seed)
    problems: list[Problem] = []
    for shape_name, shape in {**_SHAPES, **_DEEP}.items():
        for rep_name, build in _REPRESENTATIONS.items():
            for _ in range(per_cell):
                lengths = rng.sample([4, 5, 6, 7, 8], 3)
                shown = []
                for length in lengths[:2]:
                    before = build(length, rng.randrange(1000))
                    shown.append(Transition(before, shape(before)))
                # A second state at a length already shown.
                #
                # Every problem used to show its two states at two DIFFERENT
                # lengths, so each length was seen exactly once and there was
                # never anything to intersect. That makes a rule identifiable —
                # one length cannot separate a mirror from an exchange — and it
                # makes the language unrefutable: the proof that no value-blind
                # rule exists compares what one position had to take from on
                # two occasions, and there were never two occasions.
                #
                # The detector was correct and could not fire on a single one
                # of the 120, including all ten it was written for. Repetition
                # within a length is what makes a language refutable; variation
                # across lengths is what makes a rule identifiable, and a
                # battery needs both or it measures only one of them.
                for _repeat in range(2):
                    again = build(lengths[0], rng.randrange(1000))
                    shown.append(Transition(again, shape(again)))
                held_before = build(lengths[2], rng.randrange(1000))
                problems.append(
                    Problem(
                        # The index is in the name because the lengths are not
                        # enough to tell two problems apart: the generator drew
                        # the same triple twice and produced 120 problems under
                        # 119 names, so Report.missed lost one of them into the
                        # other and nobody could have noticed from the score.
                        name=f"{shape_name} / {rep_name} / {lengths} #{len(problems)}",
                        shown=tuple(shown),
                        held_out=Transition(held_before, shape(held_before)),
                        shape=shape_name,
                        representation=rep_name,
                    )
                )
    return problems


def teach_the_language(
    problems: Sequence[Problem],
    *,
    language: RelationLanguage,
    without: frozenset[str] = frozenset(),
) -> int:
    """Show the language the shapes the deep problems are built from.

    Taught on worlds of its own, at lengths the deep problems do not use, so
    what carries across is the shape and not the instance.

    ``without`` is honoured here as well as at scoring time. Teaching a shape
    with a part switched on and then scoring with it switched off measures
    neither condition, and read as an ablation it flatters whichever side did
    the teaching.
    """

    learned = 0
    for name, build in _TEACHES.items():
        if not any(problem.shape == name for problem in problems):
            continue
        world = [
            Transition(tuple(range(n)), build(tuple(range(n)))) for n in (9, 11, 13)
        ]
        found = invent_relation(world, without=without)
        if found is not None:
            language.admit(found)
            learned += 1
    return learned


def _solve(
    problem: Problem,
    language: RelationLanguage | None,
    *,
    without: frozenset[str] = frozenset(),
) -> bool:
    """Work out the relation and predict the case that was held back.

    ``without`` names parts of the mechanism to switch off, so the contribution
    of each can be measured rather than assumed. A researcher asks for this
    first and the numbers are cheap.
    """

    if language is not None:
        found = language.explain(list(problem.shown), without=without)
    else:
        found = invent_relation(list(problem.shown), without=without)
    if found is None:
        if "value_order" in without:
            return False
        # Only where a rule reading positions is PROVEN not to exist. Offered
        # beside the index language it would explain a mirror as descending
        # order, which fits every observation and is the wrong answer.
        if not certify(list(problem.shown)).proven_outside:
            return False
        ordering = solve_ordering(list(problem.shown))
        if ordering is None:
            return False
        answered = ordering.apply(problem.held_out.before)
        return answered is not None and tuple(answered) == tuple(
            problem.held_out.after
        )
    try:
        return tuple(found.apply(problem.held_out.before)) == tuple(
            problem.held_out.after
        )
    except Exception:  # noqa: BLE001 - a relation that throws did not predict it
        return False


def score_battery(
    problems: Sequence[Problem] | None = None,
    *,
    language: RelationLanguage | None = None,
    without: frozenset[str] = frozenset(),
) -> Report:
    """Run the battery and report the score, with the misses named."""

    material = list(problems if problems is not None else generate_battery())
    report = Report()
    for problem in material:
        got = _solve(problem, language, without=without)
        report.attempted += 1
        report.solved += int(got)
        if problem.shape not in BEYOND_THE_LANGUAGE:
            report.attempted_expressible += 1
            report.solved_expressible += int(got)
        for bucket, key in (
            (report.by_shape, problem.shape),
            (report.by_representation, problem.representation),
        ):
            solved, seen = bucket.get(key, (0, 0))
            bucket[key] = (solved + int(got), seen + 1)
        if not got:
            report.missed.append(problem.name)
    return report


def learning_curve(
    problems: Sequence[Problem] | None = None,
    *,
    seed: int = 20260828,
) -> Iterator[tuple[int, bool]]:
    """Whether each problem is solved, as the language accumulates shapes.

    The higher-order claim: a mechanism that has learned shapes should settle a
    NEW world in fewer observations. This yields the raw sequence so a caller
    can measure that rather than be told it.
    """

    material = list(problems if problems is not None else generate_battery(seed=seed))
    language = RelationLanguage()
    for index, problem in enumerate(material):
        got = _solve(problem, language)
        if got:
            language.admit(invent_relation(list(problem.shown)))
        yield index, got
