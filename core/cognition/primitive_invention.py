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
    #: The rule over indices, when there is one, so a language can offer this
    #: shape to the next world as a member rather than as a preference.
    index_rule: Callable[[int, int], int] | None = None
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


def _possible_sources(
    before: Sequence[Any], after: Sequence[Any]
) -> tuple[tuple[int, ...], ...] | None:
    """For each position after, EVERY position it could have taken its value from.

    Not one correspondence but all of them. Committing to a single one needs a
    tie-break when values repeat, and the tie-break was "prefer identity",
    which is a guess: a grid whose rows repeat had positions assigned to
    themselves and the shape that actually held could not be seen through it.
    Keeping the candidates lets a form be tested exactly — it holds if its
    answer is among the possibilities at every position — and no choice is
    made that the data does not force.

    None when some value in ``after`` does not occur in ``before``: that is not
    a rearrangement and a different question applies.
    """

    if len(before) != len(after):
        return None
    where: dict[Any, list[int]] = {}
    for index, value in enumerate(before):
        try:
            where.setdefault(value, []).append(index)
        except TypeError:  # an unhashable cell is not a value we can trace
            return None
    options: list[tuple[int, ...]] = []
    for value in after:
        try:
            found = where.get(value)
        except TypeError:
            return None
        if not found:
            return None
        options.append(tuple(found))
    return tuple(options)


def _a_consistent_source(options: Sequence[Sequence[int]]) -> tuple[int, ...] | None:
    """One correspondence the possibilities allow, identity where it can be."""

    taken: set[int] = set()
    chosen: list[int] = []
    for index, candidates in enumerate(options):
        if index in candidates and index not in taken:
            taken.add(index)
            chosen.append(index)
            continue
        chosen.append(-1)
    for index, candidates in enumerate(options):
        if chosen[index] != -1:
            continue
        free = [item for item in candidates if item not in taken]
        if not free:
            return None
        taken.add(free[0])
        chosen[index] = free[0]
    return tuple(chosen)


def _index_forms(size: int) -> list[tuple[str, str, Callable[[int, int], int]]]:
    """Every shape of "position i takes from f(i)" this can express, at this size.

    Generated from the size rather than listed as named operators. The offsets
    are every offset a state of this length has; the exchanges are every pair
    of positions, and every pair expressed relative to the ends so that "the
    ends exchange" means the same thing at length four and length eight.

    Length-relative pairs are here because an absolute one cannot say it. A
    world exchanging its first and last cells produced {0<->3} at length four,
    which is false at length eight, and the whole shape scored zero.
    """

    forms: list[tuple[str, str, Callable[[int, int], int]]] = [
        ("identity", "identity", lambda i, _n: i),
        ("mirror", "position i takes from n-1-i", lambda i, n: n - 1 - i),
    ]
    for step in range(1, max(2, size)):
        forms.append(
            (
                "offset",
                f"position i takes from i+{step} (mod n)",
                lambda i, n, _k=step: (i + _k) % n,
            )
        )
    for left in range(size):
        for right in range(left + 1, size):
            forms.append(
                (
                    "pairwise exchange",
                    f"positions exchange in pairs ({left}<->{right})",
                    lambda i, _n, _a=left, _b=right: (
                        _b if i == _a else (_a if i == _b else i)
                    ),
                )
            )
    for depth in range(max(1, size // 2)):
        forms.append(
            (
                "pairwise exchange",
                f"the cells {depth} in from each end exchange",
                lambda i, n, _d=depth: (
                    n - 1 - _d if i == _d else (_d if i == n - 1 - _d else i)
                ),
            )
        )
    return forms


def _forms_that_fit(
    options: Sequence[Sequence[int]],
    known: Sequence[tuple[str, str, Callable[[int, int], int]]] = (),
) -> list[tuple[str, str, Callable[[int, int], int]]]:
    """Every shape whose answer is among the possibilities at every position.

    Single shapes first, then one shape after another. A composition is a shape
    the observations never show either half of — "mirror then rotate" looks
    like neither a mirror nor a rotation — and without composing, twenty of a
    hundred battery problems were unreachable however many observations were
    offered.

    The simpler description is kept ahead of the compound one, so a world that
    IS a plain mirror is never explained as two things.
    """

    size = len(options)
    # Shapes worked out in earlier worlds are members of the language now, not
    # only a preference over it. That is what makes a NEW shape cheaper to
    # learn as more shapes are known: a composition of one learned form and one
    # base form is reachable, and was not before the first world taught it.
    singles = list(known) + _index_forms(size)
    fitting = [
        (family, description, rule)
        for family, description, rule in singles
        if _fits(rule, options, size)
    ]
    if fitting:
        return fitting
    for _fa, first_text, first in singles:
        for _fb, second_text, second in singles:
            def composed(i: int, n: int, _a=first, _b=second) -> int:
                return _a(_b(i, n), n)

            if _fits(composed, options, size):
                fitting.append(
                    (
                        "composition",
                        f"{second_text}, then {first_text}",
                        composed,
                    )
                )
    return fitting


def _fits(
    rule: Callable[[int, int], int],
    options: Sequence[Sequence[int]],
    size: int,
) -> bool:
    try:
        return all(rule(index, size) in options[index] for index in range(size))
    except (IndexError, TypeError, ZeroDivisionError):
        return False


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
    known_forms: Sequence[tuple[str, str, Callable[[int, int], int]]] = (),
) -> InventedRelation | None:
    """Work out the relation these transitions need, or return None.

    ``held_out`` is the discipline: a relation that explains only what it was
    built from has not been shown to be a relation at all. Passing none is
    allowed and is recorded, so a caller can tell an unvalidated finding from a
    validated one.

    ``prefer`` is a count per shape, from worlds already accounted for. It only
    ever decides between shapes that fit the observations equally well, so it
    can make an answer arrive sooner and cannot make a wrong answer pass.

    ``known_forms`` are shapes worked out in earlier worlds, offered as members
    of the language rather than as a preference over it. A shape reachable only
    as a composition involving one of them was not expressible before that
    world was seen, so what can be learned grows with what has been.
    """

    observed = [
        Transition(tuple(item.before), tuple(item.after)) for item in transitions if item is not None
    ]
    if not observed:
        return None

    # Did anything move, or did the values themselves change?
    possibilities = [_possible_sources(item.before, item.after) for item in observed]
    if all(item is not None for item in possibilities):
        fitted = [_forms_that_fit(item, known_forms or ()) for item in possibilities if item]
        # A shape has to fit EVERY observation. With one observation several
        # will; with two of different lengths, usually one.
        shared: dict[str, tuple[str, Callable[[int, int], int]]] = {}
        if fitted and all(fitted):
            common = set.intersection(
                *({description for _f, description, _r in options} for options in fitted)
            )
            for family, description, rule in fitted[0]:
                if description in common and description not in shared:
                    shared[description] = (family, rule)
        first = None
        if shared:
            # The prior chooses among shapes the observations do not separate.
            # With no prior this is the order the shapes are generated in,
            # which is what the measurement compares against.
            chosen = max(
                shared,
                key=lambda text: (
                    int((prefer or {}).get(shared[text][0], 0)),
                    -list(shared).index(text),
                ),
            )
            family, rule = shared[chosen]
            first = (family, chosen, rule)
        if first is not None:
            family, description, rule = first
            operator = _permutation_operator(rule)
            if explains(operator, observed):
                relation = InventedRelation(
                    kind="rearrangement",
                    form=description,
                    generalises=True,
                    apply=operator,
                    family=family,
                    learned_from=len(observed),
                    held_out_checked=len(held_out),
                    index_rule=rule,
                    detail={"fitting_shapes": sorted(shared)},
                )
                if not held_out or explains(operator, held_out):
                    return relation
                return None
        # No shape fits every observation. A single correspondence still
        # explains these states, and is reported as what it is: a rule for
        # this length.
        one_length = {len(item.before) for item in observed}
        if len(one_length) == 1 and possibilities[0] is not None:
            fixed = _a_consistent_source(possibilities[0])
            if fixed is not None and all(
                _a_consistent_source(item) == fixed for item in possibilities if item
            ):
                operator = _permutation_operator(lambda i, _n, _f=fixed: _f[i])
                if explains(operator, observed) and (
                    not held_out or explains(operator, held_out)
                ):
                    return InventedRelation(
                        kind="rearrangement",
                        form=f"positions take from {list(fixed)}",
                        generalises=False,
                        apply=operator,
                        family="fixed correspondence",
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
