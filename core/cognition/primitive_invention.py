"""When the hypothesis language cannot express what happened, extend it.

Aura can search a hypothesis family for the member that explains the world. Put
a world outside the family in front of her and she refuses, correctly, because
nothing in the language she has can say what happened. Adding the missing
member by hand answers that one world and nothing else.

This is the other move: notice that no hypothesis fits, then work out what
relation WOULD fit, from the observations themselves, and admit it to the
language so later problems can compose with it.

What keeps it from being "add swap"
-----------------------------------
The candidates are not named operators. Nothing here knows the word swap, or
rotate, or reverse. The mechanism solves for a correspondence between the state
before and the state after — which position each value came from, or what was
done to each value — and then asks whether that correspondence has a closed
form over indices. A transposition, a rotation and a reversal all fall out of
the same solve; so does anything else expressible that way. The vocabulary sits
below domain ontology, at the level of structure on finite states, which is
where a person's sense of "the same thing moved" sits too.

What it will not do
-------------------
A relation is admitted only if it explains transitions it was not built from.
An explicit permutation that fits the training states and has no closed form is
reported as such and does not generalise past that state length, because it
genuinely does not. Insufficiency that cannot be repaired is reported as
insufficiency rather than dressed up.

None of this consults a language model. The point of the exercise is that the
representation is formed here.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = [
    "InventedRelation",
    "Transition",
    "explains",
    "invent_relation",
    "language_is_sufficient",
]


@dataclass(frozen=True)
class Transition:
    """One observation: the world before, and the world after."""

    before: tuple[Any, ...]
    after: tuple[Any, ...]


@dataclass(frozen=True)
class InventedRelation:
    """A relation the language did not have, with what it was learned from.

    ``form`` is the closed form over indices when there is one, and the literal
    correspondence when there is not. ``generalises`` says which of those it
    is, because a relation that only fits one state length is a weaker thing
    and must not be reported as the same kind of finding.
    """

    kind: str
    form: str
    generalises: bool
    apply: Callable[[tuple[Any, ...]], tuple[Any, ...]]
    #: The shape this relation belongs to, as opposed to this instance of it.
    #: Transfer runs on families: a world that exchanges positions 0 and 2 is a
    #: different relation from one that exchanges 1 and 3, and the same shape.
    family: str = ""
    learned_from: int = 0
    held_out_checked: int = 0
    detail: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"{self.kind}: {self.form}"


def explains(
    operator: Callable[[tuple[Any, ...]], tuple[Any, ...]],
    transitions: Iterable[Transition],
) -> bool:
    """Whether one operator reproduces every one of these transitions."""

    seen = False
    for transition in transitions:
        seen = True
        try:
            produced = tuple(operator(tuple(transition.before)))
        except Exception:  # noqa: BLE001 - an operator that throws does not explain
            return False
        if produced != tuple(transition.after):
            return False
    return seen


def language_is_sufficient(
    operators: Iterable[Callable[[tuple[Any, ...]], tuple[Any, ...]]],
    transitions: Iterable[Transition],
) -> bool:
    """Whether anything in the current language already accounts for these."""

    observed = list(transitions)
    if not observed:
        return True
    return any(explains(operator, observed) for operator in operators)


def _positional_source(before: Sequence[Any], after: Sequence[Any]) -> tuple[int, ...] | None:
    """For each position after, which position it took its value from.

    Identity is preferred wherever it holds, so the correspondence names the
    smallest movement that accounts for the change rather than an arbitrary one
    among many. Returns None when some value in ``after`` is not in ``before``
    at all — that is not a rearrangement and a different question applies.
    """

    if len(before) != len(after):
        return None
    unused: dict[Any, list[int]] = {}
    for index, value in enumerate(before):
        unused.setdefault(value, []).append(index)
    source: list[int | None] = [None] * len(after)
    # Identity first, so an unchanged position claims itself.
    for index, value in enumerate(after):
        candidates = unused.get(value)
        if candidates and index in candidates:
            candidates.remove(index)
            source[index] = index
    for index, value in enumerate(after):
        if source[index] is not None:
            continue
        candidates = unused.get(value)
        if not candidates:
            return None
        source[index] = candidates.pop(0)
    return tuple(int(item) for item in source if item is not None) or None


def _closed_forms_over_indices(
    source: Sequence[int],
) -> list[tuple[str, str, Callable[[int, int], int]]]:
    """Every rule for "position i took from f(i)" that fits, not just the first.

    Tried as forms rather than as names: an offset that wraps, the mirror, and
    a correspondence that exchanges positions pairwise while leaving the rest
    alone. A transposition is the two-position case of the last and is not
    special-cased.

    More than one shape usually fits a single observation — (1,2) -> (2,1) is a
    pairwise exchange AND a mirror AND an offset of one — and which of them is
    the right reading is not decidable from that observation. Returning only
    the first made the choice silently, by the order the branches happen to be
    written in. Returning all of them lets a prior over shapes decide, and lets
    a second observation decide when there is no prior.
    """

    size = len(source)
    fitting: list[tuple[str, str, Callable[[int, int], int]]] = []
    if size == 0:
        return fitting
    if all(source[i] == i for i in range(size)):
        fitting.append(("identity", "identity", lambda i, _n: i))
    offset = (source[0] - 0) % size
    if all(source[i] == (i + offset) % size for i in range(size)):
        fitting.append(
            (
                "offset",
                f"position i takes from i+{offset} (mod n)",
                lambda i, n, _o=offset: (i + _o) % n,
            )
        )
    if all(source[i] == size - 1 - i for i in range(size)):
        fitting.append(("mirror", "position i takes from n-1-i", lambda i, n: n - 1 - i))
    moved = [i for i in range(size) if source[i] != i]
    if moved and all(source[source[i]] == i for i in moved):
        pairs = sorted({tuple(sorted((i, source[i]))) for i in moved})
        exchanged = ", ".join(f"{a}<->{b}" for a, b in pairs)
        mapping = {i: source[i] for i in moved}
        fitting.append(
            (
                "pairwise exchange",
                f"positions exchange in pairs ({exchanged})",
                lambda i, _n, _m=mapping: _m.get(i, i),
            )
        )
    return fitting


def _value_map(
    transitions: Sequence[Transition],
) -> tuple[str, str, Callable[[Any], Any]] | None:
    """A rule for what was done to each value, when positions did not move."""

    pairs: list[tuple[Any, Any]] = []
    for transition in transitions:
        if len(transition.before) != len(transition.after):
            return None
        pairs.extend(zip(transition.before, transition.after, strict=False))
    changed = [(a, b) for a, b in pairs if a != b]
    if not changed:
        return None
    outputs = {b for _a, b in changed}
    if len(outputs) == 1:
        only = next(iter(outputs))
        return "constant", f"every changed value becomes {only!r}", lambda _x, _c=only: _c
    try:
        offsets = {b - a for a, b in changed}  # type: ignore[operator]
    except TypeError:
        offsets = set()
    if len(offsets) == 1:
        delta = next(iter(offsets))
        return "value offset", f"every value gains {delta}", lambda x, _d=delta: x + _d
    substitution: dict[Any, Any] = {}
    for a, b in pairs:
        if a in substitution and substitution[a] != b:
            return None
        substitution[a] = b
    # A table with one entry per observation is a transcript, not a relation.
    # It reproduces everything it was shown and predicts nothing it was not,
    # which is how pure noise came back "explained" by lookup.
    #
    # An abstraction has to be smaller than what it accounts for. The smallest
    # form of that: some value has to have been seen twice and behaved the same
    # way both times. Then the table is a claim about that value rather than a
    # record of one occasion.
    if len(substitution) >= len(pairs):
        return None
    shown = ", ".join(f"{a!r}->{b!r}" for a, b in sorted(changed, key=repr)[:4])
    return (
        "substitution table",
        f"each value is replaced by its own counterpart ({shown})",
        lambda x, _s=dict(substitution): _s.get(x, x),
    )


def _permutation_operator(
    rule: Callable[[int, int], int],
) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    def operator(state: tuple[Any, ...]) -> tuple[Any, ...]:
        size = len(state)
        return tuple(state[rule(index, size)] for index in range(size))

    return operator


def _value_operator(rule: Callable[[Any], Any]) -> Callable[[tuple[Any, ...]], tuple[Any, ...]]:
    def operator(state: tuple[Any, ...]) -> tuple[Any, ...]:
        return tuple(rule(value) for value in state)

    return operator


def invent_relation(
    transitions: Sequence[Transition],
    *,
    held_out: Sequence[Transition] = (),
    prefer: dict[str, int] | None = None,
) -> InventedRelation | None:
    """Work out the relation these transitions need, or return None.

    ``held_out`` is the discipline: a relation that explains only what it was
    built from has not been shown to be a relation at all. Passing none is
    allowed and is recorded, so a caller can tell an unvalidated finding from a
    validated one.

    ``prefer`` is a count per shape, from worlds already accounted for. It only
    ever decides between shapes that fit the observations equally well, so it
    can make an answer arrive sooner and cannot make a wrong answer pass.
    """

    observed = [
        Transition(tuple(item.before), tuple(item.after)) for item in transitions if item is not None
    ]
    if not observed:
        return None

    # Did anything move, or did the values themselves change?
    correspondences = [
        _positional_source(item.before, item.after) for item in observed
    ]
    if all(item is not None for item in correspondences):
        # Compared by the form itself. Comparing (description, rule) pairs put
        # a fresh closure in every element, so no two transitions ever agreed
        # and every world came back as a rule for one state length.
        fitted = [_closed_forms_over_indices(item) for item in correspondences if item]
        # A shape has to fit EVERY observation to be a candidate. With one
        # observation several will; with two of different lengths, usually one.
        shared: dict[str, tuple[str, Callable[[int, int], int]]] = {}
        if fitted and all(fitted):
            common = set.intersection(
                *({family for family, _d, _r in options} for options in fitted)
            )
            for family, description, rule in fitted[0]:
                if family in common:
                    shared[family] = (description, rule)
        first = None
        if shared:
            # The prior chooses among shapes the observations do not separate.
            # With no prior this is the order the shapes are generated in,
            # which is what the measurement compares against.
            chosen = max(
                shared,
                key=lambda name: (int((prefer or {}).get(name, 0)), -list(shared).index(name)),
            )
            first = (chosen, *shared[chosen])
        if first is not None:
            family, description, rule = first
            operator = _permutation_operator(rule)
            if explains(operator, observed):
                relation = InventedRelation(
                    kind="rearrangement",
                    form=description,
                    generalises=True,
                    apply=operator,
                    learned_from=len(observed),
                    held_out_checked=len(held_out),
                    family=family,
                    detail={"correspondences": [list(c) for c in correspondences if c]},
                )
                if not held_out or explains(operator, held_out):
                    return relation
                return None
        # No closed form: the correspondence is still real, and still explains
        # these states. It is reported as what it is — a rule for this length.
        one_length = {len(item.before) for item in observed}
        if len(one_length) == 1 and correspondences[0] is not None:
            fixed = correspondences[0]
            if all(item == fixed for item in correspondences):
                operator = _permutation_operator(lambda i, _n, _f=fixed: _f[i])
                if explains(operator, observed) and (
                    not held_out or explains(operator, held_out)
                ):
                    return InventedRelation(
                        kind="rearrangement",
                        form=f"positions take from {list(fixed)}",
                        generalises=False,
                        apply=operator,
                        learned_from=len(observed),
                        held_out_checked=len(held_out),
                        detail={"length": next(iter(one_length))},
                    )

    mapped = _value_map(observed)
    if mapped is not None:
        family, description, rule = mapped
        operator = _value_operator(rule)
        if explains(operator, observed) and (not held_out or explains(operator, held_out)):
            return InventedRelation(
                kind="substitution",
                family=family,
                form=description,
                # A table applies only to values it has seen; an offset or a
                # constant applies to any. The difference is the whole of what
                # "generalises" means here and it must not be flattened.
                generalises=family != "substitution table",
                apply=operator,
                learned_from=len(observed),
                held_out_checked=len(held_out),
            )
    return None
