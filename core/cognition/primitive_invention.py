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
    "IndexProgram",
    "rule_for_description",
    "explains",
    "invent_relation",
    "Probe",
    "discriminating_probe",
    "language_is_sufficient",
]


#: The kinds __call__ knows how to interpret. Read from one place so a kind
#: added to the interpreter and forgotten here cannot be written down and then
#: refused on the way back in.
_KINDS_THIS_BUILD_INTERPRETS = frozenset(
    {
        "identity",
        "mirror",
        "offset",
        "exchange",
        "ends",
        "grouping",
        "affine",
        "compose",
    }
)


@dataclass(frozen=True)
class IndexProgram:
    """A rule over positions, written down rather than closed over.

    Every shape here used to be a lambda. A lambda cannot be saved, so a
    library of learned shapes could persist how OFTEN each kind had worked and
    not the shapes themselves: after a restart the counts came back and the
    expanded language contracted to the basis it started from. What had been
    learned was the one thing that did not survive.

    So a shape is a small structured value that happens to be callable. It
    interprets itself, it compares by value, and it goes to JSON and back
    without losing anything. Callers that only wanted a function still get one.

    ``kind`` names the rule, ``args`` are its numbers, and ``parts`` are the
    programs it is built from — which is what makes a composition a value too,
    and what lets refactoring take one apart.
    """

    kind: str
    args: tuple[int, ...] = ()
    parts: tuple["IndexProgram", ...] = ()

    def __call__(self, index: int, size: int) -> int:
        kind = self.kind
        if kind == "identity":
            return index
        if kind == "mirror":
            return size - 1 - index
        if kind == "offset":
            step = self.args[0] if self.args else 0
            return (index + step) % size if size else index
        if kind == "exchange":
            first, second = self.args[0], self.args[1]
            if index == first:
                return second
            if index == second:
                return first
            return index
        if kind == "ends":
            depth = self.args[0] if self.args else 0
            far = size - 1 - depth
            if index == depth:
                return far
            if index == far:
                return depth
            return index
        if kind == "affine":
            # (a*i + b) mod m, with anything past m standing still.
            #
            # The family the other forms are members of: identity is (1,0,n),
            # mirror is (-1,-1,n), offset k is (1,k,n), and dealing six cells
            # into two classes is (2,0,n-1) — the shuffle a person had to add
            # by hand, and the classical riffle.
            # Stated RELATIVE to the length, the way "the cells d in from each
            # end" is. An absolute modulus cannot say a shape: dealing into two
            # classes is mod 5 at length six and mod 7 at length eight, so the
            # same shape seen at two lengths would be two different shapes and
            # would intersect to nothing. The same mistake the exchange forms
            # made, made again here, and found the same way — by the score.
            a, b, delta = self.args[0], self.args[1], self.args[2]
            m = size + delta
            if m < 2 or index >= m:
                return index
            return (a * index + b) % m
        if kind == "grouping":
            span = self.args[0] if self.args else 1
            first = self.args[1] if len(self.args) > 1 else 0
            return _grouped_source(index, size, span, first)
        if kind == "compose":
            # Innermost first: the parts are applied in the order they were
            # composed, which is the order refactoring reads them in.
            position = index
            for part in reversed(self.parts):
                position = part(position, size)
            return position
        # A kind this build does not know is not the identity.
        #
        # It returned `index` — and the identity FITS some worlds, so a program
        # written by a later build and read back by this one would be reported
        # as a relation that was found rather than one that could not be run.
        # A silent wrong answer is worse than a loud missing one.
        raise ValueError(f"no interpretation for index program kind {kind!r}")

    def to_json(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "args": list(self.args),
            "parts": [part.to_json() for part in self.parts],
        }

    @classmethod
    def from_json(cls, raw: Any) -> "IndexProgram | None":
        if not isinstance(raw, dict):
            return None
        kind = str(raw.get("kind") or "").strip()
        if kind not in _KINDS_THIS_BUILD_INTERPRETS:
            # A kind this build cannot run is not a program.
            #
            # This accepted any non-empty string, so anything written by a
            # later build — or by another kind of learned relation entirely —
            # came back as an IndexProgram that raised the first time it was
            # asked for a position. Refusing here makes it a relation that was
            # not recognised, which the reader above can say out loud, instead
            # of a program that fails somewhere else later.
            return None
        try:
            args = tuple(int(value) for value in (raw.get("args") or ()))
        except (TypeError, ValueError):
            return None
        parts: list[IndexProgram] = []
        for item in raw.get("parts") or ():
            built = cls.from_json(item)
            if built is None:
                return None
            parts.append(built)
        return cls(kind=kind, args=args, parts=tuple(parts))


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
    index_rule: IndexProgram | None = None
    #: The parts this shape is made of, innermost first. A library that only
    #: keeps whole winners can never find structure that several solutions
    #: share without any of them being it — which is the step that keeps
    #: DreamCoder's library growing and the one an accumulating library lacks.
    components: tuple[str, ...] = ()
    learned_from: int = 0
    held_out_checked: int = 0
    #: Whether the observations pick this out, or merely admit it.
    #:
    #: Two things make an answer unsettled. Several shapes fit everything
    #: shown and disagree about the case in hand — the search returned the
    #: first of them in preference order and said nothing about the rest. And
    #: too few observations to separate anything: one worked example almost
    #: never pins a rule down, and the module's own opening says so — a world
    #: exchanging its first and last cells produced {0<->3} at length four,
    #: which is false at length eight, and one observation cannot tell those
    #: apart.
    #:
    #: Several shapes can fit everything shown and disagree about the case in
    #: hand, and the search returned the first of them in preference order
    #: and said nothing. Shown one worked example, it answered — and was
    #: wrong five times in twelve, confidently, on evidence that settled
    #: nothing. That is not a wrong answer, it is an answer where a refusal
    #: was the correct one, and the caller had no way to tell the two apart.
    settled: bool = True
    #: The other shapes that fit everything shown and disagree with this one.
    #: What a question would have to separate.
    also_fits: tuple[str, ...] = ()
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

    forms: list[tuple[str, str, IndexProgram]] = [
        ("identity", "identity", IndexProgram("identity")),
        ("mirror", "position i takes from n-1-i", IndexProgram("mirror")),
    ]
    for step in range(1, max(2, size)):
        forms.append(
            (
                "offset",
                f"position i takes from i+{step} (mod n)",
                IndexProgram("offset", (step,)),
            )
        )
    for left in range(size):
        for right in range(left + 1, size):
            forms.append(
                (
                    "pairwise exchange",
                    f"positions exchange in pairs ({left}<->{right})",
                    IndexProgram("exchange", (left, right)),
                )
            )
    for depth in range(max(1, size // 2)):
        forms.append(
            (
                "pairwise exchange",
                f"the cells {depth} in from each end exchange",
                IndexProgram("ends", (depth,)),
            )
        )
    # Cells fall into groups, and the groups move together.
    #
    # The basis had order, symmetry and adjacency — geometry and number, in the
    # core-knowledge sense — and nothing for OBJECTHOOD: no way to say that some
    # cells belong together and travel as a set. That omission is not a taste
    # call, it is the one system of the four that applies here which was left
    # out, and it predicts exactly which battery shapes fail.
    #
    # Grouping by residue is the smallest form of it: positions belong to a
    # class by where they fall in a repeating count, and the classes are laid
    # out one after another. At k=2 that is "the odd ones, then the even ones".
    # Every span the state is long enough to have, rather than half of them.
    #
    # The range stopped at half the length, so "grouped every three" existed
    # at length six and did not exist at length five — and a shape has to be
    # in the basis at EVERY length shown before it can be shared across the
    # observations. A world that groups every three, shown at lengths five,
    # seven and nine, was therefore unreachable because of the five: the
    # shape was fine, one of the examples was short, and the answer was a
    # refusal. Twelve of forty sealed rules refused for exactly this.
    for span in range(2, max(3, size)):
        for first in range(span):
            forms.append(
                (
                    "grouping",
                    f"cells are grouped every {span}, the group at {first} first",
                    IndexProgram("grouping", (span, first)),
                )
            )
    return forms


def _grouped_source(index: int, size: int, span: int, first: int = 0) -> int:
    """Which position the cell at ``index`` comes from, when cells are grouped.

    Positions are dealt into ``span`` classes by residue, the classes are laid
    end to end, and this is the inverse: given a place in the result, which
    place in the original it took.

    ``first`` is which class leads, and it is a degree of freedom rather than a
    detail: the prediction that grouping would reach "odd positions first" was
    made and failed, because the form as first written laid the even class down
    first and there was no way to say the other one. A grouping with no say in
    which group leads is half a grouping.
    """

    if span <= 1 or size <= 0:
        return index
    classes = [(residue + first) % span for residue in range(span)]
    order = [
        position for residue in classes for position in range(residue, size, span)
    ]
    if index < 0 or index >= len(order):
        return index
    return order[index]


def _parts_of(
    known: Sequence[Any], description: str
) -> tuple[str, ...]:
    """The components a form is made of, from the library if it is a learned one."""

    for entry in known:
        if len(entry) >= 4 and entry[1] == description:
            return tuple(entry[3])
    return (description,)


def _affine_forms_that_fit(
    options: Sequence[Sequence[int]],
) -> list[tuple[str, str, IndexProgram]]:
    """The members of the affine family that land inside the possibilities.

    Fitted, not enumerated. Every other form in this module is a list somebody
    wrote down; this one is solved for from what was observed, which is why it
    reaches shapes nobody wrote down. At length six the authored positional
    forms reach fifteen permutations and this family reaches forty-four.

    The cost is a search over the modulus and the multiplier, which is O(n^2)
    in the length of the state and does not grow with how much the family can
    say. That is the whole argument for it: a basis a person extends one form
    at a time is a family somebody can fit in a loop.
    """

    size = len(options)
    if size < 2:
        return []

    def signed(value: int, modulus: int) -> int:
        """The residue written the short way round, so -1 is not n-1.

        This is what makes a member of the family mean the same thing at two
        lengths. A mirror is "minus one" at every length; it is n-1 at none of
        them twice.
        """

        return value if value <= modulus // 2 else value - modulus

    found: list[tuple[str, str, IndexProgram]] = []
    seen: set[tuple[int, ...]] = set()
    for delta in (0, -1, 1):
        modulus = size + delta
        if modulus < 2:
            continue
        # Anything at or past the modulus stands still, so it has to be
        # standing still in the observations too.
        if any(place not in options[place] for place in range(modulus, size)):
            continue
        for multiplier in range(modulus):
            for shift in range(modulus):
                a, b = signed(multiplier, modulus), signed(shift, modulus)
                rule = IndexProgram("affine", (a, b, delta))
                landing = tuple(rule(place, size) for place in range(size))
                if landing in seen:
                    continue
                if sorted(landing) != list(range(size)):
                    continue
                if any(
                    landing[place] not in options[place] for place in range(size)
                ):
                    continue
                seen.add(landing)
                where = "n" if not delta else f"n{delta:+d}"
                found.append(
                    (
                        "affine",
                        f"position i takes from {a}i{b:+d} (mod {where})",
                        rule,
                    )
                )
    return found


def _forms_that_fit(
    options: Sequence[Sequence[int]],
    known: Sequence[Any] = (),
    *,
    compose: bool = True,
    without: frozenset[str] = frozenset(),
    force_compose: bool = False,
    reach_for_the_family: bool = False,
    three_deep: bool = False,
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
    if "authored_positional" in without:
        # The transpositions are genuinely a different kind — identity with two
        # positions swapped is not an affine map — so they stay when the rest
        # of the authored basis is taken away. What this ablation removes is
        # exactly what the family claims to subsume.
        authored = [
            entry
            for entry in _index_forms(size)
            if entry[2].kind in {"exchange", "ends"}
        ]
    else:
        authored = _index_forms(size)
    family = (
        _affine_forms_that_fit(options)
        if reach_for_the_family and "affine" not in without
        else []
    )
    singles = [tuple(entry)[:3] for entry in known] + authored + family
    fitting = [
        (family, description, rule)
        for family, description, rule in singles
        if _fits(rule, options, size)
    ]
    if not compose or (fitting and not force_compose):
        return fitting
    # The two-deep terms, and how much of the answer each already has. Kept
    # as they are built rather than built and then sorted: the sort was over
    # every pair of singles — seven thousand at length nine — and scoring all
    # of them cost more than the search it was ordering. The battery went
    # from 1.1 seconds to over two minutes, and the clock below could not see
    # it, because the clock bounds the search and this was the setup.
    two: list[tuple[float, int, tuple[str, str, IndexProgram]]] = []
    scored = 0
    for _fa, first_text, first in singles:
        for _fb, second_text, second in singles:
            composed = IndexProgram("compose", (), (first, second))
            said = f"{second_text}, then {first_text}"
            if _fits(composed, options, size):
                fitting.append(("composition", said, composed))
                continue
            if three_deep and scored < MOST_PAIRS_SCORED:
                scored += 1
                right = _how_much_it_gets_right(composed, options, size)
                if right:
                    # The index keeps the order total, so two shapes right
                    # about equally much always come back in the same order.
                    two.append(
                        (-float(right), len(two), ("composition", said, composed))
                    )
    if fitting and not three_deep:
        return fitting
    # Three deep, when two found nothing here or when the caller has already
    # seen that what two found does not hold across every observation.
    #
    # Deciding this inside one observation was the same mistake the two-deep
    # rung was written to fix, one level up: a two-deep form fitting length
    # five on its own stopped the three-deep search that was the only thing
    # fitting five, seven and nine together, and the world scored zero with
    # the answer never generated. Eight of forty sealed rules, every one of
    # them three shapes in a row.
    #
    # Two was the whole of it, so a world that is three shapes one after
    # another was unreachable however many observations were offered — and
    # composed worlds are the ordinary case, not the exotic one: on sealed
    # rules composed from three primitives she refused a quarter of them for
    # this reason alone. Bounded, because this is cubic in the basis: the
    # third level is built from the two-deep terms that at least land inside
    # the possibilities somewhere, which is a far smaller set than all of
    # them.
    # Ordered by how much of the answer each two-deep term already has, so
    # the ones extended first are the ones with a chance of being half of it.
    # Taken in generation order instead, the cap fell on whichever four
    # hundred happened to be built first, and a world three shapes deep was
    # unreachable because its first two were late in a list.
    two.sort()
    landing = [entry for _right, _index, entry in two]
    tried = 0
    for _fa, first_text, first in singles:
        if tried >= MOST_THREE_DEEP_TRIED:
            break
        for _fb, second_text, second in landing[:MOST_TWO_DEEP_TO_EXTEND]:
            tried += 1
            composed = IndexProgram("compose", (), (first, second))
            if _fits(composed, options, size):
                fitting.append(
                    (
                        "composition",
                        f"{second_text}, then {first_text}",
                        composed,
                    )
                )
    return fitting


#: How many two-deep shapes the three-deep search will extend. Bounded
#: because the search is cubic in the basis, and ordered by how much of the
#: answer each already has, so the ones kept are the ones with a chance of
#: being half of it.
#:
#: How many two-deep shapes each single extends. The other half of the bound
#: below, and the one that decides which candidates are reached rather than
#: how many.
MOST_TWO_DEEP_TO_EXTEND = 400

#: Counted, not timed.
#:
#: This was a wall clock, and a wall clock makes the answer depend on how
#: busy the machine is: the same battery came back with eight deep shapes
#: reachable and then with seven, on identical code. Gate eighteen of the
#: proof gauntlet asks for the same answers on a re-run, and a clock cannot
#: give that. So the bound is a count of candidates, which is the same
#: number on any machine on any day.
#:
#: Both halves are bounded, which is why the first version was useless: the
#: ordering was computed over every pair of singles — seven thousand at
#: length nine — before the bound was consulted at all, and the battery went
#: from 1.1 seconds to over two minutes without the bound seeing it.
#: Set where the reach stops changing, measured on the battery:
#:
#:     pairs scored   candidates tried   deep shapes   battery   time
#:            2,500             30,000          8/20   114/130   15.0s
#:            1,200             12,000          8/20   114/130   10.6s
#:              600              6,000          7/20   113/130    7.6s
#:              300              3,000          7/20   113/130    6.1s
#:
#: Against 106 in 1.1s before the rung existed. So the rung buys eight
#: problems for about ten times the time, the curve is flat above twelve
#: hundred, and below it a problem goes. Nothing above these numbers is free
#: and nothing below them is safe.
MOST_PAIRS_SCORED = 1200
MOST_THREE_DEEP_TRIED = 12_000


def _how_much_it_gets_right(
    rule: Callable[[int, int], int],
    options: Sequence[Sequence[int]],
    size: int,
) -> int:
    """How many positions this shape is already right about.

    The ordering on what is worth extending, and the filter: a shape wrong
    everywhere is not half of an answer, and one right about most of it very
    likely is.
    """

    try:
        return sum(
            1 for index in range(size) if rule(index, size) in options[index]
        )
    except (IndexError, TypeError, ZeroDivisionError):
        return 0


def _fits(
    rule: Callable[[int, int], int],
    options: Sequence[Sequence[int]],
    size: int,
) -> bool:
    try:
        return all(rule(index, size) in options[index] for index in range(size))
    except (IndexError, TypeError, ZeroDivisionError):
        return False


def _affine_value_map(
    changed: Sequence[tuple[Any, Any]],
) -> tuple[str, str, Callable[[Any], Any]] | None:
    """``v -> a*v + b`` fitted from two pairs and checked against the rest.

    Exact arithmetic only. A slope that does not divide evenly is not a slope
    these cells have, and rounding it would be inventing a rule that nearly
    works — which is worse here than none, because nearly is indistinguishable
    from right on the examples it was fitted to.
    """

    pairs = [(a, b) for a, b in changed if isinstance(a, int) and isinstance(b, int)]
    if len(pairs) != len(changed) or len(pairs) < 2:
        return None
    apart = next(
        (
            (one, other)
            for index, one in enumerate(pairs)
            for other in pairs[index + 1 :]
            if one[0] != other[0]
        ),
        None,
    )
    if apart is None:
        return None
    (x1, y1), (x2, y2) = apart
    rise, run = y2 - y1, x2 - x1
    if run == 0 or rise % run:
        return None
    slope = rise // run
    if slope in (0, 1):
        # A constant or a plain offset, which are said better above.
        return None
    shift = y1 - slope * x1
    if any(slope * a + shift != b for a, b in pairs):
        return None
    said = f"every value becomes {slope} times itself"
    if shift > 0:
        said += f", plus {shift}"
    elif shift < 0:
        said += f", minus {abs(shift)}"
    return (
        "value scaling",
        said,
        lambda x, _s=slope, _b=shift: _s * x + _b,
    )


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
    # v -> a*v + b, solved for rather than listed.
    #
    # The value side had "becomes a constant" and "gains k" and nothing else,
    # so doubling every cell — as plain a rule as either of them — was silent.
    # Adding "times k" beside them would have been the third entry in a list
    # that has no end; these are three members of one family and the family is
    # two points and a check, the same argument as the positional one.
    fitted = _affine_value_map(changed)
    if fitted is not None:
        return fitted
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
    #
    # One value repeating is not enough of that. Asking only for FEWER entries
    # than pairs let a single fixed point — 5 becoming 5, seen twice — license
    # ten arbitrary entries beside it, and the table then fired as a fallback
    # and answered confidently wrong. Half is the honest reading of "smaller
    # than what it accounts for": every entry carries two observations on
    # average, not one plus a rounding error.
    if len(substitution) * 2 > len(pairs):
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


def rule_for_description(description: str) -> IndexProgram | None:
    """The rule a basis shape's description names, or None if it names none.

    Refactoring works over descriptions, because that is what a shared
    sub-sequence is made of, and the parts of a learned shape are mostly BASIS
    atoms rather than library entries. Without a way back from a description to
    its rule, a shared run could be found and never rebuilt.

    A generous size is used to generate the basis, so the pairwise forms that
    only exist at larger sizes are reachable; the rules themselves take the
    size as an argument and do not depend on the one used to make them.
    """

    wanted = str(description or "").strip()
    if not wanted:
        return None
    for size in (12, 8, 4):
        for _family, said, rule in _index_forms(size):
            if said == wanted:
                return rule
    return None


def _which_others_disagree(
    chosen: str,
    shared: dict[str, tuple[str, Any]],
    observed: Sequence[Transition],
    about: Sequence[int] = (),
) -> tuple[str, ...]:
    """Other shapes that fit everything shown and act differently anyway.

    Checked on lengths outside the observations as well as on them, because
    two shapes agreeing on the three rows shown is exactly the situation
    where they can differ on the fourth — which is the case being asked about.

    ``about`` is that case, when the caller knows it. Settledness is about
    the question being asked, and a window of plus two around the
    observations is a guess at where the question will be.

    Measured, and it changed nothing: on two hundred sealed rules from a
    fixed seed, 163 right and 2 confidently wrong either way. The five wrong
    answers that prompted this were a different draw of rules, not the window
    being too narrow, and reading them as evidence for the fix would have
    been reading the seed. It stays because it asks the right question rather
    than a nearby one, and its worth here is nought.

    What the remaining wrong answers are is a harder thing: every shape that
    fits agrees, and all of them are wrong. There is no disagreement to find,
    so no amount of checking for one helps — the true shape is not in the
    fitting set in a form anything can tell apart.
    """

    lengths = sorted({len(one.before) for one in observed})
    trying = sorted(
        {
            *lengths,
            *(one + 1 for one in lengths),
            *(one + 2 for one in lengths),
            *(int(one) for one in about if int(one) > 0),
        }
    )
    mine = _permutation_operator(shared[chosen][1])
    others: list[str] = []
    for description, (_family, rule) in shared.items():
        if description == chosen:
            continue
        theirs = _permutation_operator(rule)
        for size in trying:
            row = tuple(range(size))
            try:
                if tuple(mine(row)) != tuple(theirs(row)):
                    others.append(description)
                    break
            except (TypeError, ValueError, IndexError, ZeroDivisionError):
                continue
    return tuple(sorted(others))


def _components_of(description: str, known: Sequence[Any]) -> tuple[str, ...]:
    """The parts of this shape, innermost first, resolving learned ones.

    A composition is described "B, then A". Splitting on that and resolving
    each half through the library gives the flat sequence of parts, which is
    what a refactoring step needs: shared structure is a shared SUB-SEQUENCE,
    and a description string cannot be searched for one.
    """

    parts: list[str] = []
    for piece in str(description).split(", then "):
        piece = piece.strip()
        if not piece:
            continue
        resolved = _parts_of(known, piece)
        parts.extend(resolved if resolved != (piece,) else [piece])
    return tuple(parts)


@dataclass(frozen=True)
class Probe:
    """A case that would tell the surviving rules apart."""

    state: tuple[Any, ...]
    rivals: tuple[tuple[str, tuple[Any, ...]], ...]

    def asked(self) -> str:
        """The question, as a person would put it."""

        names = " or ".join(f"{text}" for text, _r in self.rivals[:2])
        return (
            f"More than one rule fits everything you have shown: {names}. "
            f"What does {list(self.state)} become? That case separates them."
        )


def discriminating_probe(
    transitions: Sequence[Transition],
    *,
    known_forms: Sequence[tuple[str, str, Callable[[int, int], int]]] = (),
) -> Probe | None:
    """A state whose answer would settle which surviving rule is right.

    "More observations would settle it" was the honest verdict and a useless
    one: it named a shortage without naming what would end it, so the only move
    left was to wait. The rules that survive are known, and two rules that
    disagree somewhere disagree on a state that can be constructed.

    Distinct cells, so the probe cannot be ambiguous about where anything came
    from, and the shortest length that separates them, because a person has to
    answer it.
    """

    observed = [
        Transition(tuple(item.before), tuple(item.after))
        for item in transitions
        if item is not None
    ]
    if not observed:
        return None
    possibilities = [_possible_sources(item.before, item.after) for item in observed]
    if not all(item is not None for item in possibilities):
        return None

    surviving: dict[str, Callable[[int, int], int]] = {}
    fitted = [
        _forms_that_fit(item, known_forms or (), compose=True)
        for item in possibilities
        if item
    ]
    if not fitted or not all(fitted):
        return None
    common = set.intersection(
        *({description for _f, description, _r in each} for each in fitted)
    )
    for _family, description, rule in fitted[0]:
        if description in common and description not in surviving:
            surviving[description] = rule
    if len(surviving) < 2:
        return None

    seen_lengths = {len(item.before) for item in observed}
    for length in sorted(set(range(2, 10)) | seen_lengths):
        state = tuple(range(1, length + 1))
        answers: dict[tuple[Any, ...], str] = {}
        for description, rule in surviving.items():
            try:
                result = tuple(state[rule(place, length)] for place in range(length))
            except (IndexError, ValueError, ZeroDivisionError):
                continue
            answers.setdefault(result, description)
        if len(answers) >= 2:
            return Probe(
                state=state,
                rivals=tuple(
                    (description, result) for result, description in answers.items()
                ),
            )
    return None


def _note_a_step() -> None:
    """One candidate walked, where the record is available to hear it."""
    try:
        from core.cognition.the_record_of_her_own_work import note_a_step

        note_a_step()
    except ImportError:  # no-op: counting is not what this module is for
        pass


#: Observations below which nothing is settled, whatever fits. One worked
#: example is consistent with too much; it takes a second, at a different
#: length, before agreement between two shapes means they are the same shape.
ENOUGH_TO_SETTLE = 2


def invent_relation(
    transitions: Sequence[Transition],
    *,
    held_out: Sequence[Transition] = (),
    prefer: dict[str, int] | None = None,
    known_forms: Sequence[tuple[str, str, Callable[[int, int], int]]] = (),
    without: frozenset[str] = frozenset(),
    about: Sequence[int] = (),
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

    ``about`` is the length or lengths the answer will be applied to, when the
    caller knows. Whether the observations settle a question depends on the
    question, and without it settledness is checked over a neighbourhood of
    the lengths shown rather than over the case in hand.
    """

    observed = [
        Transition(tuple(item.before), tuple(item.after)) for item in transitions if item is not None
    ]
    if not observed:
        return None

    # Did anything move, or did the values themselves change?
    possibilities = [_possible_sources(item.before, item.after) for item in observed]
    if all(item is not None for item in possibilities):
        def read(
            *, force_compose: bool, family: bool = False, three_deep: bool = False
        ) -> list:
            return [
                _forms_that_fit(
                    item,
                    () if "known_forms" in without else (known_forms or ()),
                    compose="composition" not in without,
                    without=without,
                    force_compose=force_compose,
                    reach_for_the_family=family,
                    three_deep=three_deep,
                )
                for item in possibilities
                if item
            ]

        def agreed(options: list) -> dict[str, tuple[str, Callable[[int, int], int]]]:
            """The shapes every observation admits, in generation order."""

            out: dict[str, tuple[str, Callable[[int, int], int]]] = {}
            if not options or not all(options):
                return out
            common = set.intersection(
                *({description for _f, description, _r in each} for each in options)
            )
            for family, description, rule in options[0]:
                # One candidate considered, reported to the one counter every
                # search reports to. Without it this path spent thousands and
                # said it spent nothing, so every answer it gave priced at
                # nothing and no change was ever worth making on the path that
                # answers most questions.
                _note_a_step()
                if description in common and description not in out:
                    out[description] = (family, rule)
            return out

        # A shape has to fit EVERY observation. With one observation several
        # will; with two of different lengths, usually one.
        fitted = read(force_compose=False)
        shared = agreed(fitted)
        if not shared and "composition" not in without:
            # Whether a world needs two shapes is not a fact about one
            # observation. It was being decided inside each one, before
            # anything could see whether they AGREE: a single form fitting
            # length six on its own stopped the composition that was the only
            # thing fitting six and eight together, and the world scored zero
            # with the answer never generated.
            fitted = read(force_compose=True)
            shared = agreed(fitted)
        if not shared and "affine" not in without:
            # Only now. The affine family is a wider net than the written-down
            # forms and it catches things they would have caught, so offering
            # it alongside them changes answers that were already right — three
            # groupings went from found to lost that way, and the family had
            # not gained a single problem to pay for them.
            #
            # A language is extended where it fails, not where it works.
            for compose_too in (False, True):
                fitted = read(force_compose=compose_too, family=True)
                shared = agreed(fitted)
                if shared:
                    break
        if not shared and "composition" not in without:
            # The last rung: three shapes one after another, searched across
            # every observation rather than inside one. It is the dearest and
            # it is last for that reason; everything above it is cheaper and
            # answers most worlds.
            fitted = read(force_compose=True, three_deep=True)
            shared = agreed(fitted)
        # The prior chooses among shapes the observations do not separate.
        # With no prior this is the order the shapes are generated in, which is
        # what the measurement compares against.
        #
        # In preference ORDER, not one pick. One shape was chosen and then
        # checked, and a shape that fit the observations but failed the
        # held-out case ended the world — with a shape that would have passed
        # sitting unexamined in the same set. Choosing before checking is only
        # safe when the check cannot fail.
        order = sorted(
            shared,
            key=lambda text: (
                -int((prefer or {}).get(shared[text][0], 0)),
                list(shared).index(text),
            ),
        )
        for description in order:
            family, rule = shared[description]
            operator = _permutation_operator(rule)
            if not explains(operator, observed):
                continue
            if held_out and not explains(operator, held_out):
                continue
            disagreeing = _which_others_disagree(
                description, shared, observed, about=about
            )
            return InventedRelation(
                kind="rearrangement",
                form=description,
                generalises=True,
                apply=operator,
                family=family,
                learned_from=len(observed),
                held_out_checked=len(held_out),
                index_rule=rule,
                components=_components_of(description, known_forms or ()),
                settled=not disagreeing and len(observed) >= ENOUGH_TO_SETTLE,
                also_fits=disagreeing,
                detail={"fitting_shapes": sorted(shared)},
            )
        if shared:
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
                def _only_at_its_own_length(
                    i: int, n: int, _f: tuple[int, ...] = fixed
                ) -> int:
                    """A fixed correspondence has no answer at another length.

                    Read out of range it raised IndexError from inside the
                    lambda, which reaches a caller as a crash rather than as
                    a refusal. ``generalises`` already says this only fits one
                    length; saying it again where it is used is what makes it
                    safe for anything that did not check.
                    """

                    if n != len(_f):
                        raise ValueError(
                            f"this correspondence was learned at length "
                            f"{len(_f)} and says nothing at length {n}"
                        )
                    return _f[i]

                operator = _permutation_operator(_only_at_its_own_length)
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
                        settled=len(observed) >= ENOUGH_TO_SETTLE,
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
                settled=len(observed) >= ENOUGH_TO_SETTLE,
            )

    # A move AND a map, where neither alone accounts for it.
    #
    # The two sides were solved separately and could not meet. "Mirror, then
    # add one to every cell" is not a rearrangement, because no cell that came
    # out ever went in; and it is not a value map, because the cells also
    # moved. Between them they say it exactly.
    #
    # Undoing the map is what makes this cheap. A map that can be inverted
    # turns the observed states back into the ones the move produced, and the
    # move is then an ordinary positional world, solved by everything above.
    together = (
        None
        if "value_map_composition" in without
        else _map_then_move(observed, held_out, prefer, known_forms or (), without)
    )
    if together is not None:
        return together
    return None


def _map_then_move(
    observed: list[Transition],
    held_out: Sequence[Transition],
    prefer: dict[str, int] | None,
    known_forms: Sequence[tuple[str, str, Callable[[int, int], int]]],
    without: frozenset[str],
) -> InventedRelation | None:
    """A positional rule and a value map, when neither alone explains it."""

    if any(len(item.before) != len(item.after) for item in observed):
        return None
    # Pair the SORTED states, not the positions.
    #
    # The map is applied to what the move produced, so before[i] and after[i]
    # are not a pair of the map's — they are a pair of the composition's, and
    # reading them as the map's gave four different offsets for one offset.
    # A move is a permutation, so the two multisets are related by the map
    # alone, and sorting both recovers the pairs it was actually applied to.
    changed: list[tuple[Any, Any]] = []
    for item in observed:
        try:
            went_in = sorted(item.before)
            came_out = sorted(item.after)
        except TypeError:
            return None
        if len(went_in) != len(came_out):
            return None
        changed.extend(zip(went_in, came_out, strict=True))
    for said, undo in _invertible_value_maps(changed):
        try:
            undone = [
                Transition(item.before, tuple(undo(cell) for cell in item.after))
                for item in observed
            ]
        except (ArithmeticError, TypeError, ValueError):
            continue
        moved = invent_relation(
            undone,
            prefer=prefer,
            known_forms=known_forms,
            without=without | {"value_map_composition"},
        )
        if moved is None or moved.kind != "rearrangement":
            continue
        rule = moved.index_rule
        if rule is None:
            continue

        def apply(state: Sequence[Any], _r=rule, _said=said) -> tuple[Any, ...]:
            size = len(state)
            moved_state = tuple(state[_r(place, size)] for place in range(size))
            return tuple(_said[1](cell) for cell in moved_state)

        if not explains(apply, observed):
            continue
        if held_out and not explains(apply, held_out):
            continue
        return InventedRelation(
            kind="rearrangement and substitution",
            family="move and map",
            form=f"{moved.form}, and then {said[0]}",
            generalises=True,
            apply=apply,
            learned_from=len(observed),
            held_out_checked=len(held_out),
            index_rule=rule,
            # Two rules at once, fitted together. If one observation cannot
            # settle a move, it certainly cannot settle a move and a map.
            settled=len(observed) >= ENOUGH_TO_SETTLE and moved.settled,
        )
    return None


def _invertible_value_maps(
    changed: list[tuple[Any, Any]],
) -> list[tuple[tuple[str, Callable[[Any], Any]], Callable[[Any], Any]]]:
    """Value maps that can be undone, so the move underneath can be seen.

    A constant is not here: it destroys what it was applied to, and a state it
    produced cannot be turned back into the one the move made. Only maps that
    keep the information are usable this way, which is a property rather than a
    list somebody chose.
    """

    numbers = [(a, b) for a, b in changed if isinstance(a, int) and isinstance(b, int)]
    if len(numbers) < 2:
        return []
    found: list[tuple[tuple[str, Callable[[Any], Any]], Callable[[Any], Any]]] = []
    offsets = {b - a for a, b in numbers}
    if len(offsets) == 1:
        delta = next(iter(offsets))
        if delta:
            found.append(
                (
                    (f"every value gains {delta}", lambda v, _d=delta: v + _d),
                    lambda v, _d=delta: v - _d,
                )
            )
    apart = next(
        (
            (one, other)
            for index, one in enumerate(numbers)
            for other in numbers[index + 1 :]
            if one[0] != other[0]
        ),
        None,
    )
    if apart is not None:
        (x1, y1), (x2, y2) = apart
        rise, run = y2 - y1, x2 - x1
        if run and rise % run == 0:
            slope = rise // run
            shift = y1 - slope * x1
            if slope not in (0, 1) and all(
                slope * a + shift == b for a, b in numbers
            ):
                said = f"every value becomes {slope} times itself"
                if shift:
                    said += f", plus {shift}" if shift > 0 else f", minus {abs(shift)}"
                found.append(
                    (
                        (said, lambda v, _s=slope, _b=shift: _s * v + _b),
                        lambda v, _s=slope, _b=shift: (v - _b) // _s,
                    )
                )
    return found
