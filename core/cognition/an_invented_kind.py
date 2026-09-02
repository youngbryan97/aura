"""Kinds of rule whose meaning she worked out, not kinds somebody wrote a branch for.

A learned rule is a node with a kind, and the interpreter knows three kinds:
apply these in turn, read the positions, read the cells. Anything else returns
None. So she can compose programs out of the meanings she was given, and the
set of meanings never grows — a node of an unknown kind has no semantics, and
acquiring one has always meant a person editing the interpreter.

This is the registry that makes the set grow. A kind admitted here carries its
own executable meaning, and the interpreter consults it exactly as it consults
the three it was born with. Adding one is not an edit to the interpreter.

Where the meaning comes from
----------------------------
Not from a name. A meaning here is a point in a small algebra over finite
states: which two places the value at each position is read from, and what is
done with the pair. Reversal, rotation, shifting, taking the larger of a pair and
combining a pair all fall out of the same two choices, and none of them is in
the space by name — the same discipline the rule space and the measure space
already follow.

What will not be admitted
-------------------------
Anything that only explains the examples it was induced from. A meaning is
admitted on transitions it was NOT built from, and refused otherwise, because
a rule fitted to everything has been tested against nothing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Iterator, Sequence

__all__ = [
    "ENOUGH_HELD_BACK",
    "WAYS_TO_BUILD",
    "addressings",
    "UNSETTLED",
    "hold_unsettled",
    "settle_with",
    "what_they_agree_on",
    "everything_that_fits",
    "what_would_tell_them_apart",
    "Induced",
    "KINDS",
    "admit",
    "every_meaning",
    "forget",
    "how_many_were_walked",
    "induce_from",
    "start_counting_again",
    "interpretation_of",
]

logger = logging.getLogger("Aura.AnInventedKind")

#: How many transitions a meaning must get right that it was not induced from.
#: One is a coincidence when states are short.
ENOUGH_HELD_BACK = 2


# ── where the value at a position comes from ─────────────────────────────

WHERE_FROM: dict[str, Callable[[int, int], int]] = {
    "here": lambda index, size: index,
    "the far end": lambda index, size: size - 1 - index,
    "one along": lambda index, size: (index + 1) % size,
    "one back": lambda index, size: (index - 1) % size,
    "its partner": lambda index, size: index + 1 if index % 2 == 0 else index - 1,
}

#: A second place, chosen the same way. What makes the pair itself something
#: she works out rather than something fixed: pairing each place with the one
#: after it, with its partner, or with its mirror are different meanings, and
#: which one holds is a question for the examples.


# ── what is done with what is found there ────────────────────────────────


def _as_it_is(one: Any, _other: Any) -> Any:
    return one


def _the_larger(one: Any, other: Any) -> Any:
    try:
        return one if float(one) >= float(other) else other
    except (TypeError, ValueError):
        return one


def _the_smaller(one: Any, other: Any) -> Any:
    try:
        return one if float(one) <= float(other) else other
    except (TypeError, ValueError):
        return one


def _both_together(one: Any, other: Any) -> Any:
    try:
        return type(one)(float(one) + float(other)) if isinstance(one, (int, float)) else one
    except (TypeError, ValueError):
        return one


WHAT_OF_IT: dict[str, Callable[[Any, Any], Any]] = {
    "as it is": _as_it_is,
    "the larger of it and its neighbour": _the_larger,
    "the smaller of it and its neighbour": _the_smaller,
    "both together": _both_together,
}


@dataclass(frozen=True)
class Induced:
    """One meaning: where each value comes from, and what is done with it.

    Executable, and expressible in the same breath, so what she admitted can
    be said out loud as well as run.
    """

    where_from: str
    and_from: str
    what_of_it: str
    #: How it did on transitions it was not induced from. Nought means it was
    #: never held to any, which is not the same as failing them.
    held_back: float = 0.0
    from_examples: int = 0

    @property
    def name(self) -> str:
        if self.what_of_it == "as it is":
            return f"take {self.where_from}"
        return f"take {self.where_from} and {self.and_from}, {self.what_of_it}"

    def read(self, cells: Sequence[Any]) -> tuple[Any, ...] | None:
        """The state this meaning turns ``cells`` into, or None where it cannot."""
        found = tuple(cells)
        size = len(found)
        if size == 0:
            return ()
        try:
            where_all = addressings()
            where = where_all[self.where_from]
            other = where_all[self.and_from]
            what = WHAT_OF_IT[self.what_of_it]
        except KeyError:
            return None
        out: list[Any] = []
        for index in range(size):
            try:
                one = found[where(index, size) % size]
                two = found[other(index, size) % size]
                out.append(what(one, two))
            except (IndexError, KeyError, TypeError, ValueError, ZeroDivisionError):
                # A word she derived refuses what it has never seen: an
                # addressing read off length four says nothing about length
                # six, and an operation read off the pairs in front of her
                # says nothing about a pair that was not. Refusing is the
                # honest answer and it is what makes a derived word safe to
                # put in the language everything else is built from.
                return None
        return tuple(out)

    def describe(self) -> str:
        held = (
            f", right about {self.held_back:.0%} of what it was not built from"
            if self.from_examples
            else ""
        )
        return f"{self.name}{held}"


#: Ways of BUILDING an addressing out of addressings.
#:
#: One at first, and it is not written here because it is the absence of one:
#: use a word as it is. A second entry is not a new word, it is a new kind of
#: word-making — and admitting one enlarges the language everywhere at once,
#: because every addressing she has and every one she ever derives is put
#: through it.
#:
#: This is the difference between inventing a thought, inventing a kind of
#: thought, and inventing a way of inventing kinds of thought. WHERE_FROM and
#: WHAT_OF_IT are the first. A word derived and admitted to them is the second.
#: An entry here is the third.
WAYS_TO_BUILD: dict[str, Callable[[dict[str, Any]], dict[str, Any]]] = {}


#: The last answer ``addressings`` gave, and what the language looked like when
#: it gave it. Rebuilding is cheap once and ruinous a million times: reading a
#: single state consults it, so a search over the space consults it once per
#: expression, and after one level of growth that is 525 words rebuilt a
#: million times over.
_LAST_BUILT: tuple[Any, dict[str, Any]] | None = None


def _what_the_language_is() -> Any:
    """A value that changes exactly when the language does.

    Which sizes words are told apart at is part of what the language IS: widen
    them and two words that were one become two again, so a cache that ignored
    them would hand back a vocabulary built under the narrower reading.
    """
    try:
        from core.cognition.one_thing_many_spellings import (
            sizes_words_are_told_apart_at,
        )

        apart = sizes_words_are_told_apart_at()
    except (ImportError, AttributeError):
        apart = ()
    return (tuple(sorted(WHERE_FROM)), tuple(sorted(WAYS_TO_BUILD)), apart)


def addressings() -> dict[str, Any]:
    """Every way of saying where a value comes from, however it was arrived at.

    The words she was given, the words she derived, and everything the ways of
    building make out of them. One dictionary, because nothing downstream
    should care which of the three a word came from.
    """
    global _LAST_BUILT

    now = _what_the_language_is()
    if _LAST_BUILT is not None and _LAST_BUILT[0] == now:
        return _LAST_BUILT[1]
    made = dict(WHERE_FROM)
    # Each maker builds on what the makers before it made.
    #
    # Every one of them was handed `dict(WHERE_FROM)` — the words she was
    # GIVEN, and only those. So the language was W0 together with M1(W0),
    # M2(W0), M3(W0) and so on, never M2(M1(W0)): a maker could be found
    # while an earlier maker's word sat in its hole, and then be rebuilt
    # against a vocabulary that word was not in. Nothing she made could be
    # made ON anything she had made, which is the whole of what accumulating
    # was supposed to mean.
    #
    # Order is the order she arrived at them, so a maker sees its
    # predecessors and not its successors, and the vocabulary stays a
    # foundation rather than a knot.
    for build in list(WAYS_TO_BUILD.values()):
        try:
            made.update(build(dict(made)))
        except (TypeError, ValueError, KeyError):
            continue
    # One word per behaviour. A maker produces the same thing several ways —
    # "if this then here else here" is "here" — and every duplicate is another
    # branch at every step of every search, bought for nothing.
    try:
        from core.cognition.one_thing_many_spellings import one_of_each

        made = one_of_each(made)
    except (ImportError, AttributeError, RuntimeError, TypeError, ValueError):
        pass
    _LAST_BUILT = (now, made)
    return made


def every_meaning() -> Iterator[Induced]:
    """The whole space, shortest first, so nothing had to be thought of in advance.

    The order matters as much as the contents. Enumerated however the loops
    happen to run, a nine-symbol coincidence is checked before a two-symbol
    explanation, and whichever is checked first wins a tie — so a language that
    grew a composed word would start preferring the composed reading of things
    the simple word already explained. Shortest first is the same preference as
    favouring the simpler hypothesis.

    Ordered by bucketing the words by what they cost and walking the buckets,
    rather than by building the space and sorting it. Once she has grown a
    level the space runs past a million expressions, and the answer is usually
    in the first few hundred — so materialising all of it to find out costs
    more than the search does.
    """
    from core.cognition.what_it_costs_to_say import _symbols

    where_all = addressings()
    by_cost: dict[int, list[str]] = {}
    for name in where_all:
        by_cost.setdefault(_symbols(name), []).append(name)
    for names in by_cost.values():
        names.sort()
    costs = sorted(by_cost)
    combining = sorted(name for name in WHAT_OF_IT if name != "as it is")

    longest = max(costs) if costs else 0
    for total in range(1, 2 * longest + 2):
        # Taking a value as it is never reads the second place, so it is not
        # charged for one.
        for name in by_cost.get(total - 1, ()):
            yield Induced(where_from=name, and_from=name, what_of_it="as it is")
        for first in costs:
            second = total - first - 1
            if second not in by_cost:
                continue
            for where_from in by_cost[first]:
                for and_from in by_cost[second]:
                    for what_of_it in combining:
                        yield Induced(
                            where_from=where_from,
                            and_from=and_from,
                            what_of_it=what_of_it,
                        )


#: Kinds she has worked out the meaning of, by name. Empty at import: nothing
#: is here that a person put here.
KINDS: dict[str, Induced] = {}


#: Lengths a pair of meanings is compared over. Short, because behaviour here
#: is decided by which places a value is read from — a difference that shows up
#: at one length shows up at every length that has those places.
_LENGTHS_TO_SETTLE_IT = (2, 3, 4, 5)


def _how_it_behaves(meaning: Induced, of_length: int = 4) -> tuple[Any, ...]:
    """What a meaning does across every telling state, as one comparable value.

    Computed once per meaning rather than once per pair. Comparing meanings by
    searching for a disagreement between each of them is quadratic in a set
    that is often twenty strong, and the search is exhaustive — which turned a
    correct answer into a suite that took half an hour.
    """
    return tuple(
        meaning.read(state)
        for size in sorted({int(of_length), *_LENGTHS_TO_SETTLE_IT})
        if size >= 2
        for state in _every_telling_state(size)
    )


def what_would_tell_them_apart(
    one: Induced, other: Induced, *, of_length: int = 4
) -> tuple[Any, ...] | None:
    """A state these two answer differently, searched EXHAUSTIVELY.

    Not sampled. Two hundred random probes over the six and a half thousand
    states of length four is "no counterexample found", and using that as "the
    same meaning" is exactly the kind of quiet approximation this subsystem
    exists not to make.

    The domain is finite and small enough to walk. Behaviour here depends on
    which two places each position reads from and what is done with the pair,
    so a state whose values are all different exposes every difference in the
    reading, and states with repeats expose differences in the doing. Both are
    enumerated, over several lengths, and None means checked rather than
    unlucky.
    """
    for size in sorted({int(of_length), *_LENGTHS_TO_SETTLE_IT}):
        if size < 2:
            continue
        for state in _every_telling_state(size):
            mine, theirs = one.read(state), other.read(state)
            if mine is not None and theirs is not None and mine != theirs:
                return state
    return None


def _every_telling_state(size: int) -> Iterator[tuple[int, ...]]:
    """States of this length that between them expose any difference.

    All-distinct values first, which separate any two readings that take from
    different places. Then every state over two values, which separates any two
    doings that treat a pair differently — including the ties a set of distinct
    values can never produce.
    """
    from itertools import permutations, product

    yield from permutations(range(1, size + 1))
    yield from product((1, 2), repeat=size)
def everything_that_fits(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> list[Induced]:
    """Every meaning in the space that accounts for all of these.

    More than one usually does, and that is a fact about the examples rather
    than about her. Four examples where the first value happens to be the
    smallest are explained by "repeat the first" and by "take the smaller of
    each pair" alike, and picking one of those quietly is how a confident
    answer gets made out of evidence that does not support it.
    """
    pairs = [(tuple(before), tuple(after)) for before, after in transitions]
    if not pairs:
        return []
    fitting = [
        meaning
        for meaning in every_meaning()
        if all(meaning.read(before) == after for before, after in pairs)
    ]
    # Told apart by what they DO, not by how they are written. Taking the
    # smaller of a pair reads two ways round and is one meaning either way, and
    # reporting that as two readings of the evidence would invent a doubt that
    # is not there.
    size = len(pairs[0][0])
    distinct: list[Induced] = []
    seen: set[tuple[Any, ...]] = set()
    for meaning in fitting:
        behaves = _how_it_behaves(meaning, size)
        if behaves in seen:
            continue
        seen.add(behaves)
        distinct.append(meaning)
    return distinct or fitting[:1]


#: How many meanings the last search walked. What a failed search costs, in
#: the one unit everything about developing is priced in, and a measurement
#: rather than an estimate: a search that finds nothing walks all of them.
_WALKED: list[int] = [0]


def how_many_were_walked() -> int:
    """Meanings walked since this was last reset."""
    return _WALKED[0]


def start_counting_again() -> int:
    """Reset the walk counter and give back what it held."""
    was = _WALKED[0]
    _WALKED[0] = 0
    return was


def induce_from(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> Induced | None:
    """Work out a meaning that accounts for these before-and-after pairs.

    Half to solve on, half to be judged on, because a meaning fitted to every
    example it has seen has been tested against nothing. Returns nothing when
    no meaning in the space survives the half it never saw — which is the
    honest answer and how "the language cannot say this" stays sayable.
    """
    pairs = [(tuple(before), tuple(after)) for before, after in transitions]
    if len(pairs) < ENOUGH_HELD_BACK + 1:
        return None
    # Words are told apart at the sizes something has asked about, and this is
    # the asking. Without it, identity was decided on three fixed sizes and the
    # only word answering a family of size six was dropped as a duplicate.
    try:
        from core.cognition.one_thing_many_spellings import also_compare_at

        also_compare_at({len(before) for before, _after in pairs})
    except (ImportError, AttributeError, TypeError, ValueError):
        logger.debug("could not widen where words are told apart", exc_info=True)
    solving = pairs[0::2]
    judging = pairs[1::2]
    if len(judging) < ENOUGH_HELD_BACK:
        return None
    for meaning in every_meaning():
        _WALKED[0] += 1
        if not all(meaning.read(before) == after for before, after in solving):
            continue
        right = sum(1 for before, after in judging if meaning.read(before) == after)
        if right < len(judging):
            continue
        found = Induced(
            where_from=meaning.where_from,
            and_from=meaning.and_from,
            what_of_it=meaning.what_of_it,
            held_back=right / len(judging),
            from_examples=len(solving),
        )
        logger.info(
            "a meaning nobody wrote accounts for this: %s", found.describe()
        )
        return found
    return None


#: Kinds whose meaning is NOT settled: several readings account for everything
#: seen and they disagree about what comes next. Held as a set on purpose.
UNSETTLED: dict[str, tuple[Induced, ...]] = {}


def hold_unsettled(kind: str, meanings: Sequence[Induced]) -> str:
    """Keep several readings of a kind, because the evidence chose none of them.

    Reporting an ambiguity and then admitting one of the candidates anyway is
    saying two different things: out loud, "your examples do not settle it";
    in memory, "I learned this one". The second is what steers the next
    answer, so the first was decoration.

    A kind whose evidence is thin has a SET of meanings, and it keeps that set
    until something tells them apart.
    """
    name = str(kind or "").strip()
    kept = tuple(one for one in meanings if isinstance(one, Induced))
    if not name or len(kept) < 2:
        return ""
    UNSETTLED[name] = kept
    logger.info("%r is not settled: %d readings account for everything seen", name, len(kept))
    return name


def what_they_agree_on(kind: str, cells: Sequence[Any]) -> tuple[Any, ...] | None:
    """What every unsettled reading of this kind says, when they all say the same.

    None when they disagree, which is the honest answer: the evidence does not
    determine this case, and picking one of them would be inventing a
    certainty. Cases they happen to agree on are answerable without settling
    anything, and that is worth having — most cases are.
    """
    meanings = UNSETTLED.get(str(kind or ""))
    if not meanings:
        return None
    said = [one.read(cells) for one in meanings]
    if any(answer is None for answer in said):
        return None
    return said[0] if all(answer == said[0] for answer in said) else None


def settle_with(kind: str, transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]]) -> str:
    """Rule out the readings these observations contradict.

    When one survives it is admitted and the set is gone: the discriminating
    observation promoted it, which is what an unsettled set is FOR.
    """
    name = str(kind or "").strip()
    meanings = UNSETTLED.get(name)
    if not meanings:
        return ""
    pairs = [(tuple(before), tuple(after)) for before, after in transitions]
    surviving = tuple(
        one for one in meanings
        if all(one.read(before) == after for before, after in pairs)
    )
    if not surviving:
        UNSETTLED.pop(name, None)
        logger.info("%r: nothing that fitted before fits now — the set is gone", name)
        return "none"
    if len(surviving) == len(meanings):
        return ""
    if len(surviving) == 1:
        UNSETTLED.pop(name, None)
        # The survivor carries what settled it. A meaning admitted with no
        # evidence behind it is refused, rightly — and the evidence here is
        # every observation it has survived, including the one that ruled the
        # others out.
        settled = Induced(
            where_from=surviving[0].where_from,
            and_from=surviving[0].and_from,
            what_of_it=surviving[0].what_of_it,
            held_back=1.0,
            from_examples=len(pairs),
        )
        admit(name, settled)
        logger.info("%r is settled: %s", name, settled.name)
        return "settled"
    UNSETTLED[name] = surviving
    return "narrowed"


def admit(kind: str, meaning: Induced) -> str:
    """Give a kind of node a meaning, so the interpreter can run it.

    No edit to the interpreter. That is the whole point: the set of things a
    node can mean grows because she worked one out, not because somebody added
    a branch for it.
    """
    name = str(kind or "").strip()
    if not name or not isinstance(meaning, Induced):
        return ""
    if not meaning.from_examples:
        # A meaning that was never held to anything it did not come from is a
        # guess with an executable body, and running it would be worse than
        # saying she cannot.
        return ""
    KINDS[name] = meaning
    logger.info("she gave %r a meaning: %s", name, meaning.describe())
    return name


def forget(kind: str) -> bool:
    """Take a meaning back out. What was admitted on evidence can lose it."""
    return KINDS.pop(str(kind), None) is not None


def interpretation_of(kind: str) -> Callable[[Sequence[Any]], tuple[Any, ...] | None] | None:
    """How to run a node of that kind, or nothing if she has no meaning for it."""
    meaning = KINDS.get(str(kind or ""))
    if meaning is not None:
        return meaning.read
    # A kind whose meaning is not settled still answers the cases its readings
    # agree about, and refuses the ones they do not.
    if str(kind or "") in UNSETTLED:
        return lambda cells: what_they_agree_on(str(kind), cells)
    return None
