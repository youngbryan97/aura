"""One algebra, applied to itself, so no level of it is a list somebody wrote.

She could add a word, then a way of building words, then a way of building
those. Each level was a set of named things, and every one of those sets was
written down by a person:

    A(t)   the words                    grew, by derivation
    M(t)   the ways of building words   grew, by recipe
    B(t)   the ways of writing recipes  never grew

B was composition, inversion and iteration. She could reach any constructor in
their closure and nothing outside it, so a family needing a way of building
that is not one of those three — a maker that BRANCHES, say — was unreachable
however long she searched. Adding a fourth to the list moves the ceiling; it
does not remove it.

There is no ceiling here because there is no list. A way of building is a TERM
in the same algebra a word is a term in, with a hole where a word goes. Filling
the hole gives a word; leaving it open gives a way of making words. So:

    composition   through(hole 1, through(hole 0, where))
    iteration     through(hole 0, through(hole 0, where))
    inversion     undo(hole 0, where)
    branching     if (many is even) then hole 0 else hole 1

None of those four is a primitive. They are four things that can be written,
and so is everything else the grammar admits — which is what "no authored
ceiling" has to mean: not a longer list, but no list.

What IS written down here is the grammar: an index, a size, a number read off
the data, arithmetic, a comparison, a branch, applying a word, and undoing one.
That is the floor of computing rather than a menu of ideas, and past it the
honest measure stops being "what can she express" and becomes "what can she
reach" — which core/cognition/what_it_costs_to_say.py already counts.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Sequence

__all__ = [
    "HEADS",
    "HOW_MANY_PARTS",
    "Term",
    "what_it_rests_on",
    "a_maker_she_wrote",
    "as_a_maker",
    "every_term",
    "holes_in",
    "read_back",
    "run",
    "written_down",
]

logger = logging.getLogger("Aura.OneAlgebra")


@dataclass(frozen=True)
class Term:
    """A piece of the algebra. The same shape at every level, which is the point."""

    head: str
    parts: tuple["Term", ...] = ()
    value: Any = None

    @property
    def name(self) -> str:
        if self.head == "where":
            return "where it is"
        if self.head == "many":
            return "how many there are"
        if self.head == "fixed":
            return str(self.value)
        if self.head == "hole":
            return f"a word (#{int(self.value or 0)})"
        if self.head == "through":
            return f"{self.parts[0].name} of {self.parts[1].name}"
        if self.head == "over again":
            return f"{self.parts[1].name}, {self.parts[0].name} times over"
        if self.head == "undo":
            return f"{self.parts[0].name} undone at {self.parts[1].name}"
        if self.head == "if":
            return (
                f"if {self.parts[0].name} then {self.parts[1].name} "
                f"else {self.parts[2].name}"
            )
        return f"{self.parts[0].name} {self.head} {self.parts[1].name}"

    def how_long(self) -> int:
        return 1 + sum(part.how_long() for part in self.parts)


def _plus(a: int, b: int) -> int:
    return a + b


def _minus(a: int, b: int) -> int:
    return a - b


def _times(a: int, b: int) -> int:
    return a * b


def _over(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("nothing goes into nothing")
    return a // b


def _left_over(a: int, b: int) -> int:
    if b == 0:
        raise ZeroDivisionError("nothing is left over from nothing")
    return a % b


def _below(a: int, b: int) -> int:
    return 1 if a < b else 0


def _same(a: int, b: int) -> int:
    return 1 if a == b else 0


#: The grammar. Not a menu of ways to build — the floor of computing, in which
#: any way of building can be written. Two of them take words rather than
#: numbers, and those two are what make a term a way of MAKING words.
HEADS: dict[str, Callable[[int, int], int]] = {
    "plus": _plus,
    "minus": _minus,
    "times": _times,
    "over": _over,
    "left over": _left_over,
    "below": _below,
    "same as": _same,
}

#: How many parts each head takes, for every head ``run`` can evaluate.
#:
#: One table, because there were two and they disagreed. ``run`` has always
#: evaluated "over again"; ``read_back`` listed the heads it would accept and
#: that head was not among them, so a maker she wrote using it was written
#: down correctly, refused on the way back in, and logged as a term that does
#: not read. The head the docstring above calls the one with a shape no
#: fixed-length composition has was the one that could not survive a restart.
#:
#: Deriving the check from this table rather than from a second literal is
#: what stops it happening again: a head added to ``run`` and not to here has
#: no arity, and :func:`read_back` refuses it loudly instead of silently.
HOW_MANY_PARTS: dict[str, int] = {
    **dict.fromkeys(HEADS, 2),
    "where": 0,
    "many": 0,
    "fixed": 0,
    "hole": 0,
    "through": 2,
    "undo": 2,
    "over again": 2,
    "if": 3,
}


def run(
    term: Term, index: int, size: int, words: Sequence[Any], depth: int = 0
) -> int:
    """What this term says, at this place, in a thing of this size.

    Not remembered, and that was measured rather than assumed. A term is a pure
    function of where it is asked and how long the thing is, so the same
    question always has the same answer and caching is sound; a search walks
    millions of terms sharing most of their parts, so the hit rate is high. It
    is still a loss: keying on the term means hashing it, hashing a term walks
    it, and walking it costs what running it costs — these are a few integer
    operations, not a model call. Measured at two depths, remembering answers
    did half as much work per second as not remembering them.
    """
    if depth > 32:
        raise ValueError("a term that will not settle")
    head = term.head
    if head == "where":
        return int(index)
    if head == "many":
        return int(size)
    if head == "fixed":
        return int(term.value or 0)
    if head == "hole":
        which = int(term.value or 0)
        if not (0 <= which < len(words)):
            raise IndexError("no word for that hole")
        return int(words[which](index, size)) % max(1, size)
    if head == "through":
        inner = run(term.parts[1], index, size, words, depth + 1) % max(1, size)
        return _apply(term.parts[0], inner, size, words, depth)
    if head == "over again":
        # Doing something as many times as the thing itself says to.
        #
        # Every other head composes a FIXED number of times, so what a term of
        # length L can reach is a fixed number of steps. This one takes its
        # count from the term, and the term can read the size — so "go half way
        # along" is one short term here and needs a composition as long as the
        # thing is anywhere else. That is why it adds something the others
        # cannot: not a shorter way of saying what was sayable, but a shape no
        # fixed-length composition has.
        #
        # The count is capped at the size, and that is not a budget. Walking a
        # set of n places n times has already reached whatever cycle it is
        # going to reach; past that, going round again says nothing new. So the
        # cap is where the answers stop changing, which is the world's number
        # rather than mine, and it makes this decidable at every size.
        times = run(term.parts[0], index, size, words, depth + 1)
        at = int(index)
        for _ in range(max(0, min(int(times), max(1, size)))):
            at = _apply(term.parts[1], at, size, words, depth)
        return at % max(1, size)
    if head == "undo":
        wanted = run(term.parts[1], index, size, words, depth + 1) % max(1, size)
        found = [
            at
            for at in range(size)
            if _apply(term.parts[0], at, size, words, depth) == wanted
        ]
        if len(found) != 1:
            raise ValueError("nothing to undo: it does not move things one for one")
        return found[0]
    if head == "if":
        chosen = 1 if run(term.parts[0], index, size, words, depth + 1) else 2
        return run(term.parts[chosen], index, size, words, depth + 1)
    work = HEADS.get(head)
    if work is None:
        raise ValueError(f"nothing in the grammar called {head!r}")
    return work(
        run(term.parts[0], index, size, words, depth + 1),
        run(term.parts[1], index, size, words, depth + 1),
    )


def _apply(what: Term, at: int, size: int, words: Sequence[Any], depth: int) -> int:
    """Use a word — or a hole standing for one — at a place."""
    if what.head == "hole":
        which = int(what.value or 0)
        if not (0 <= which < len(words)):
            raise IndexError("no word for that hole")
        return int(words[which](at, size)) % max(1, size)
    return run(what, at, size, words, depth + 1) % max(1, size)


def holes_in(term: Term) -> int:
    """How many words this term takes. None means it IS a word."""
    if term.head == "hole":
        return int(term.value or 0) + 1
    return max((holes_in(part) for part in term.parts), default=0)


@dataclass(frozen=True)
class Made:
    """A word a maker made, kept with the term and the words that made it."""

    term: Term
    words: tuple[Any, ...]
    #: The NAMES of the words in its holes. Provenance, not decoration.
    #:
    #: Without it, whether a maker built on a word an earlier maker made was
    #: answered by looking for one name inside another as text — which is a
    #: guess about spelling, not a fact about construction. Two words with the
    #: same name from different makers would confuse it, and a maker whose
    #: product happened to contain a substring would satisfy it.
    built_from: tuple[str, ...] = ()

    def __call__(self, index: int, size: int) -> int:
        return run(self.term, index, size, self.words)


def as_a_maker(term: Term) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Turn a term with holes into a way of building words.

    The whole collapse in one function: a maker is not a kind of thing, it is a
    term that has not been given all its words yet.
    """
    takes = max(1, holes_in(term))

    def make(words: dict[str, Any]) -> dict[str, Any]:
        made: dict[str, Any] = {}
        names = list(words)
        for chosen in _choose(names, takes):
            said = ", ".join(chosen)
            made[f"{term.name} [{said}]"] = Made(
                term=term,
                words=tuple(words[name] for name in chosen),
                built_from=tuple(chosen),
            )
        return made

    make.term = term  # type: ignore[attr-defined]
    return make


def what_it_rests_on(name: str, words: Mapping[str, Any]) -> frozenset[str]:
    """Every word this one is built on, all the way down to what she was given.

    A fact read off the construction rather than off the spelling. A word she
    was given rests on nothing; a word a maker made rests on the words in its
    holes and on everything THOSE rest on.
    """
    seen: set[str] = set()
    edge = [name]
    while edge:
        here = edge.pop()
        word = words.get(here)
        for parent in getattr(word, "built_from", ()) or ():
            if parent not in seen:
                seen.add(parent)
                edge.append(parent)
    return frozenset(seen)


def _choose(names: Sequence[str], takes: int) -> Iterator[tuple[str, ...]]:
    """Every way of picking that many words, in order."""
    if takes <= 0:
        yield ()
        return
    for name in names:
        for rest in _choose(names, takes - 1):
            yield (name, *rest)


#: How deep a term may go. Read off the problem rather than chosen: a family
#: whose states are five long cannot need a term deeper than five, and every
#: extra level multiplies what has to be walked.
def _how_deep(transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]]) -> int:
    return max(2, min(4, max((len(before) for before, _ in transitions), default=2)))


def _numbers_the_problem_shows(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> tuple[int, ...]:
    """Constants the states themselves put on the table.

    Solved for, never searched. The sizes she has seen and the distances
    between places in them are what a rule about position can be made of; a
    range somebody picked is not.
    """
    found: set[int] = {0, 1}
    for before, _after in transitions:
        size = len(before)
        found.add(size)
        found.add(size - 1)
        found.add(size // 2)
    return tuple(sorted(value for value in found if 0 <= value <= 64))


def every_term(
    constants: Sequence[int], *, holes: int, deepest: int
) -> Iterator[Term]:
    """Every term the grammar admits, shortest first, up to that depth.

    Shortest first for the reason it is everywhere else here: the shortest
    thing that accounts for the evidence is the one to believe, and the space
    grows fast enough that the order decides what is reachable at all.
    """
    leaves = [Term("where"), Term("many")]
    leaves += [Term("fixed", value=int(k)) for k in constants]
    slots = [Term("hole", value=n) for n in range(max(1, holes))]
    yield from leaves
    # A hole is a piece like any other.
    #
    # It was kept in its own list and only ever used as the FIRST part of
    # "through" and "undo", so the only composition she could write was "that
    # word, of this term". The other direction — this term, of what that word
    # gives you — was never generated at all, although `run` has always
    # evaluated it. Criterion 6 is exactly that direction: a maker built on a
    # word an earlier maker made. The term she needed was five symbols long,
    # computed the family correctly, and did not appear in sixty thousand
    # candidates because nothing could produce its shape.
    by_size: dict[int, list[Term]] = {1: [*leaves, *slots]}
    # Applying a word and undoing one take a term where the others take a
    # number, and they are what makes a term able to make words at all.
    heads = (*HEADS, "through", "undo", "over again")
    # Every size, not every other one.
    #
    # Stepping by two was right while every head took two parts: one plus two
    # odd sizes is odd, so nothing even was ever needed. Branching takes
    # THREE, and one plus three odd sizes is even — so the size an `if` needs
    # was never walked, `by_size` never held an even bucket, and the branch
    # loop below asked for children it could not be given.
    #
    # Branching is the head the docstring under it calls the one no amount of
    # composing, undoing or repeating can produce. Not one has ever been
    # produced either: zero in three hundred thousand terms, at any depth.
    for size in range(3, 2 * max(1, deepest) + 2):
        grown: list[Term] = []
        for left_size in range(1, size - 1):
            right_size = size - left_size - 1
            for left in by_size.get(left_size, ()):
                for right in by_size.get(right_size, ()):
                    for head in heads:
                        made = Term(head, parts=(left, right))
                        grown.append(made)
                        yield made
        # And branching, which is the one no amount of composing, undoing or
        # repeating can produce.
        for test_size in range(1, size - 2):
            for rest in range(1, size - test_size - 1):
                third = size - test_size - rest - 1
                for test in by_size.get(test_size, ()):
                    for then in by_size.get(rest, ()):
                        for otherwise in by_size.get(third, ()):
                            made = Term("if", parts=(test, then, otherwise))
                            grown.append(made)
                            yield made
        by_size[size] = grown


def the_closure_of_composing_undoing_and_repeating(
    words: dict[str, Any], *, deepest: int = 3
) -> set[tuple[int, ...]]:
    """Every word those three ways of building can reach, as behaviours.

    Here so that "outside that basis" is a thing she can check rather than a
    thing said about her. A term whose behaviour is not in this set is not
    something composition, inversion or iteration could have produced,
    whatever order they were applied in.
    """
    sizes = (3, 4, 5)

    def behaviour(word: Any) -> tuple[int, ...] | None:
        try:
            return tuple(
                int(word(at, size)) % size for size in sizes for at in range(size)
            )
        except (ArithmeticError, IndexError, TypeError, ValueError):
            return None

    reached: dict[tuple[int, ...], Any] = {}
    for word in words.values():
        shape = behaviour(word)
        if shape is not None:
            reached[shape] = word
    for _ in range(max(1, deepest)):
        grown: dict[tuple[int, ...], Any] = dict(reached)
        for one in list(reached.values()):
            for other in list(reached.values()):
                for made in (
                    _after(one, other),
                    _after(one, one),
                    _undone(one),
                ):
                    if made is None:
                        continue
                    shape = behaviour(made)
                    if shape is not None:
                        grown.setdefault(shape, made)
        if len(grown) == len(reached):
            break
        reached = grown
    return set(reached)


def _after(first: Any, then: Any) -> Any:
    def made(index: int, size: int) -> int:
        return int(then(int(first(index, size)) % max(1, size), size))

    return made


def _undone(word: Any) -> Any:
    def made(index: int, size: int) -> int:
        found = [at for at in range(size) if int(word(at, size)) % size == index % size]
        if len(found) != 1:
            raise ValueError("nothing to undo")
        return found[0]

    return made


def _where_each_came_from(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> dict[int, tuple[int, ...]]:
    """For each size, where each place took its value from.

    Readable straight off the examples wherever the values in a state are
    distinct, which is what makes this a synthesis problem rather than a
    search: she is not looking for a term that happens to make a family
    sayable, she is looking for the term that computes a correspondence she
    can already see.
    """
    at: dict[int, tuple[int, ...]] = {}
    for before, after in ((tuple(b), tuple(a)) for b, a in transitions):
        if len(before) != len(after) or len(set(before)) != len(before):
            continue
        seen = {value: index for index, value in enumerate(before)}
        try:
            found = tuple(seen[value] for value in after)
        except KeyError:
            continue
        if at.get(len(before), found) != found:
            return {}
        at[len(before)] = found
    return at


#: How long one synthesis may run. Measured, not chosen: over three families the
#: fixed slab of two dozen words this replaces took 58.6s, and the two it solved
#: took 12.2s and 12.9s. The widening gets the same allowance per family and
#: spends it on the likeliest words first.
_AS_LONG_AS_THE_OLD_WAY_TOOK = 20.0


def a_maker_she_wrote(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    *,
    now_sayable: Callable[[], bool],
    holes: int = 2,
    within: float = _AS_LONG_AS_THE_OLD_WAY_TOOK,
) -> Term | None:
    """Write a way of building words for the family in front of her.

    No list is consulted. What is tried is every term the grammar admits,
    shortest first, and a term with a hole in it IS a way of building words —
    so what she can arrive at is not three ways and their closure but anything
    the algebra can say.

    Written against the correspondence rather than against the whole space of
    meanings. Asking "does admitting this make the family sayable?" of every
    candidate runs the entire induction per candidate and does not finish;
    asking "does this term compute what I can see happening?" is a few dozen
    integer operations, and the one confirmation at the end is what proves it
    was the right question.
    """
    from core.cognition.an_invented_kind import WAYS_TO_BUILD, addressings

    if now_sayable():
        return None
    wanted = _where_each_came_from(transitions)
    if not wanted:
        return None
    deepest = _how_deep(transitions)
    constants = _numbers_the_problem_shows(transitions)
    # Every word she has, including the ones earlier makers produced.
    #
    # Filled from the words she was GIVEN, a maker could never be built on
    # what another maker made — so nothing composed with anything and each
    # invention was an island. What she can write next has to be able to use
    # what she wrote last, or the growth does not accumulate.
    #
    # Shortest first, and bounded: the search is quadratic in the number of
    # words, and a language that has grown holds hundreds. The bound is a
    # budget on this search rather than a claim about how much she may know.
    from core.cognition.an_invented_kind import addressings
    from core.cognition.how_she_learns_to_look import (
        in_the_order_worth_trying,
        remember_what_worked,
        widening_word_lists,
    )
    from core.cognition.what_it_costs_to_say import _symbols

    every = addressings()
    # Ordered by what this family shows AND by what every family before showed.
    #
    # A word that lands on none of the right places is unlikely to be the one
    # inside the term that does; one that lands on most of them is nearly the
    # answer already. That much is evidence from the case. How often a word
    # turned up in a term that survived its own gate is evidence from her
    # history, and the two multiply.
    #
    # Only the ORDER is learned. What is kept is decided by the same gate as
    # before, so a prior that learned to propose rubbish loses time and keeps
    # nothing.
    names = in_the_order_worth_trying(
        every, _tells_her_the_answer, wanted, shortest=_symbols
    )
    # And widening rather than a fixed slab of words: an easy family is one
    # whose answer is near the front, and paying a hard family's price to find
    # it buys nothing.
    began = time.monotonic()
    # The terms are walked once and kept, as they come.
    #
    # Widening the word list re-enters this loop, and generating the terms
    # again each round cost more than the shorter word lists saved — the
    # measurement showed no gain at all until they were kept. Materialising
    # them up front was worse still: it runs before any check of the clock, so
    # the time budget could not apply to it and a deep family hung. The cache
    # fills as the search goes, which is the only version where the budget
    # covers everything.
    stream = every_term(constants, holes=holes, deepest=deepest)
    seen_terms: list[Term] = []

    def terms_so_far() -> Any:
        yield from seen_terms
        for term in stream:
            if holes_in(term) < 1:
                continue
            seen_terms.append(term)
            yield term

    tried: set[tuple[str, tuple[int, ...]]] = set()
    for shortlist in widening_word_lists(
        names, holes=holes, within=within, started=began
    ):
        fillings = [
            tuple(every[name] for name in chosen)
            for chosen in _choose(shortlist, max(1, holes))
        ]
        for term in terms_so_far():
            if time.monotonic() - began >= within:
                logger.info("gave up writing a maker after %.1fs", within)
                return None
            for words in fillings:
                # A round re-offers what the round before it tried. Checking
                # those again is the cost of widening, and it is avoidable.
                already = (term.name, tuple(id(one) for one in words))
                if already in tried:
                    continue
                tried.add(already)
                if not _computes(term, words, wanted):
                    continue
                name = f"a way she wrote: {term.name}"
                if name in WAYS_TO_BUILD:
                    continue
                # The clock again, before the expensive half.
                #
                # It was checked once per term, which bounds the search but
                # not one candidate's admission: putting a maker into the
                # language rebuilds every word it makes and weighs the result,
                # and a maker that makes hundreds takes minutes on its own. A
                # thirty-second budget ran for over ten, all of it inside a
                # single iteration that the check at the top had already
                # passed.
                if time.monotonic() - began >= within:
                    logger.info("gave up writing a maker after %.1fs", within)
                    return None
                if not _holds_at_a_size_it_never_saw(term, words, transitions):
                    continue
                before = len(addressings())
                WAYS_TO_BUILD[name] = as_a_maker(term)
                if now_sayable() and _earns_its_place(term, transitions, before):
                    logger.info("she wrote a way of building words: %s", term.name)
                    remember_what_worked(
                        one for one in shortlist if every[one] in words
                    )
                    return term
                WAYS_TO_BUILD.pop(name, None)
    return None


def _holds_at_a_size_it_never_saw(
    term: Term, words: Sequence[Any], transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]]
) -> bool:
    """Whether it still says something at a length it was not fitted to.

    Held-out examples are already how a meaning is judged. The SIZE it was
    fitted at is a second dimension of the same idea, and nothing looked at
    it: a term that works at four and five and falls apart at nine has been
    tested at no length it did not see.

    What can be checked there is what the family does not say: the family
    gives no wanted answer at an unseen length, so the test is that the term
    still names a place inside the thing, rather than raising or pointing
    outside it. That catches a term that only holds where it was fitted.

    It does NOT prove the term holds at every length, and nothing here should
    be read as proving it. Passing at one unseen length is evidence of the
    same kind as passing one unseen example — worth having, and not a theorem.
    """
    lengths = {len(before) for before, _after in transitions}
    if not lengths:
        return True
    # Two lengths past the longest it saw, and one prime-ish length beside
    # them, because a term keyed to even or odd alone survives the first.
    beyond = max(lengths)
    for size in (beyond + 1, beyond + 2, beyond + 5):
        for at in range(size):
            try:
                said = run(term, at, size, words)
            except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
                return False
            if not isinstance(said, int) or not 0 <= said % size < size:
                return False
    return True


def _earns_its_place(
    term: Term,
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    vocabulary_before: int,
) -> bool:
    """Whether it is worth carrying, measured on what it was not built from.

    Making the family in front of her sayable proves it fits the evidence it
    was made from, which is the test a lookup table passes. What it has to earn
    is a place in the language she thinks in from now on, and every word it
    makes is another branch at every step of every search.

    Weighed on the half of the evidence the synthesis did not see. A maker is
    always worth everything on the family it was made for.
    """
    from core.cognition.an_invented_kind import addressings, induce_from
    from core.cognition.is_it_worth_keeping import what_it_is_worth

    held_out = list(transitions)[1::2]
    if len(held_out) < 2:
        # Nothing was held back, so there is nothing to weigh it on. Making
        # the family sayable is all the evidence there is.
        return True
    # What it saves is paid once and used many times.
    #
    # Counted on the ruler she cannot move: a word this maker produced has a
    # short NAME, and counting names a maker that labels long things briefly
    # collapses every length in the system and every promotion looks like a
    # triumph. Nothing improved — the ruler moved. Written out in the
    # substrate, the maker's term costs what it costs, and the saving is that
    # it is written out ONCE rather than inside each word it makes.
    made = sum(1 for name in addressings() if term.name in name)
    saved_each = max(0, term.how_long() - 1)
    worth = what_it_is_worth(
        now_sayable=lambda family: induce_from(list(family)) is not None,
        held_out=[held_out],
        was_sayable=(False,),
        vocabulary_before=max(1, vocabulary_before),
        vocabulary_after=max(1, len(addressings())),
        longest=max(2, max((len(before) for before, _ in transitions), default=2)),
        shorter_by=saved_each,
        used=max(0, made - 1),
    )
    if not worth.keep_it:
        logger.info("not keeping %s — %s", term.name, worth.describes())
    return worth.keep_it


def _tells_her_the_answer(word: Any, wanted: dict[int, tuple[int, ...]]) -> int:
    """Whether the answer is a function of what this word says.

    The condition every term over a word has to satisfy, and the reason a word
    is worth trying inside one. If the same thing said by the word ever has to
    become two different answers, no term over that word can exist, however
    long the search runs. If it never does, one might.

    Strictly more than agreement, which asks whether the word IS the answer —
    a special case, where the function is identity. Ordering by agreement put
    the word criterion 6 needs at seventeenth of twenty-five, so it was only
    reached in the last widening round after eighty thousand terms; the word
    does not resemble the answer at all, it determines it, and one shift away
    is as good as identity for anything that can be written over it.

    Counted in places, so it is on the same scale agreement was.
    """
    told = 0
    for size, found in wanted.items():
        if size <= 0:
            continue
        says: dict[int, int] = {}
        try:
            for at in range(size):
                said = int(word(at, size)) % size
                if says.setdefault(said, found[at]) == found[at]:
                    told += 1
        except (ArithmeticError, IndexError, TypeError, ValueError):
            continue
    return told


def _agrees_with(word: Any, wanted: dict[int, tuple[int, ...]]) -> int:
    """How many of the places this word already puts where they belong."""
    agreed = 0
    for size, found in wanted.items():
        if size <= 0:
            continue
        try:
            agreed += sum(
                1 for at in range(size) if int(word(at, size)) % size == found[at]
            )
        except (ArithmeticError, IndexError, TypeError, ValueError):
            continue
    return agreed


def _computes(term: Term, words: Sequence[Any], wanted: dict[int, tuple[int, ...]]) -> bool:
    """Whether this term, given these words, is the correspondence she saw.

    Stops at the first place it disagrees. Building the whole answer for a
    size before comparing it does the work for every position of a term that
    was wrong at the first, and the great majority of them are.
    """
    for size, found in wanted.items():
        if size <= 0:
            return False
        for at in range(size):
            try:
                if run(term, at, size, words) % size != found[at]:
                    return False
            except (ArithmeticError, IndexError, RecursionError, TypeError, ValueError):
                return False
    return True


def written_down(term: Term) -> dict[str, Any]:
    """The term as plain data, so what she wrote survives a restart."""
    return {
        "head": term.head,
        "value": term.value,
        "parts": [written_down(part) for part in term.parts],
    }


def read_back(row: Any) -> Term | None:
    """A term from what was written down, or nothing when it does not read."""
    if not isinstance(row, dict):
        return None
    head = str(row.get("head") or "")
    wanted = HOW_MANY_PARTS.get(head)
    if wanted is None:
        return None
    parts = tuple(
        part for part in (read_back(one) for one in row.get("parts") or ()) if part
    )
    if len(parts) != wanted:
        return None
    return Term(head=head, parts=parts, value=row.get("value"))
