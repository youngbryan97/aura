"""A rule whose shape is a term, so the shape is not a thing somebody wrote.

`an_invented_kind.Induced` is the last authored ceiling above the floor, and it
is a different kind of ceiling from the heads. A rule there is always

    after[i] = what(before[g(i, n)], before[h(i, n)])

Two sources, one operation, both of them value-blind. Every word she derives,
every maker she writes and every head she now writes fits inside that sentence,
and no amount of any of them changes it. A family wanting three sources, or an
operation that depends on where it is, or a value that was never in the state,
is unsayable however wide the vocabulary gets.

`language_limits.certify` already names the last of those exactly: where the
cells themselves changed, it says the question belongs to the other side and
refuses to rule. This is the other side.

A rule here is a floor term. It is handed the whole state before, the length,
where it is, and itself, and it gives the value that goes at that place. The
number of sources, whether it reads values or positions, what it does with
them, and whether it counts down are all inside the term, so none of them is a
field in a record and none of them has a fixed arity.

The vocabulary it is given
--------------------------
Four bindings, and one term: reading the state at a place needs a fixed point,
and a fixed point is past what a shortest-first search reaches, so it is
supplied the way the five positional words are supplied.

That is a vocabulary, and a vocabulary is not a ceiling. The difference matters
and is worth stating rather than assuming: an authored TERM is something she
could have written, sits in the same registry as the things she does write, and
can be added to. Authored CODE — a record with three fields, a schema in
Python — is something she cannot add to at all. The first is data and the
second is a ceiling, and this module exists to move one of them into the other.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from core.cognition.the_floor_she_stands_on import (
    FST,
    IF,
    MINUS,
    SAME,
    SND,
    A,
    Code,
    L,
    N,
    OutOfFuel,
    Stuck,
    V,
    Y,
    build,
    every_code,
    from_list,
    how_long,
    read_back,
    run,
    written_down,
)

__all__ = [
    "RULES_WITH_NO_SHAPE",
    "THE_CELL_AT",
    "WHAT_A_RULE_IS_GIVEN",
    "Rule",
    "a_rule_she_wrote",
    "as_a_rule",
    "read_a_rule_back",
    "the_rule_written_down",
]

logger = logging.getLogger("Aura.ARuleWithNoShape")

#: What a rule is given, innermost binder first — so a term written for it
#: reads index nought as the state and index three as itself.
WHAT_A_RULE_IS_GIVEN: tuple[str, ...] = (
    "the whole state before",
    "how long the state is",
    "where it is",
    "itself, given two numbers",
)

_THE_STATE, _HOW_LONG, _WHERE, _ITSELF = 0, 1, 2, 3

#: Reading the state at a place. A term, not a head and not a primitive, and
#: supplied for the same reason the fixed point is: finding it costs twenty
#: symbols and using it costs three.
THE_CELL_AT: Code = build(
    Y(
        "at",
        L(
            "cells",
            L(
                "k",
                IF(
                    SAME(V("k"), N(0)),
                    FST(V("cells")),
                    A(V("at"), SND(V("cells")), MINUS(V("k"), N(1))),
                ),
            ),
        ),
    )
)

#: The rules she wrote whose shape is their own. Empty at boot and filled from
#: what was kept, the same as the heads.
RULES_WITH_NO_SHAPE: dict[str, Rule] = {}

#: What one rule may spend on one place.
_A_RULE_MAY_SPEND = 20_000

#: How many distinct sets of places to try. Read off the enumeration rather
#: than chosen: over six lengths, expressions of six symbols or fewer name
#: about this many distinct sets between them, and past that the enumeration
#: is producing longer spellings of places already tried.
_HOW_MANY_PLACES_ARE_WORTH_TRYING = 240

#: How many first sources are given a second. Folding costs a solve per place
#: per source, so this and the next bound the expensive route.
_HOW_MANY_STARTS_ARE_WORTH_FOLDING = 12

#: How many places a second source may be read at.
_HOW_MANY_PLACES_ARE_WORTH_FOLDING = 48


def as_a_rule(body: Code) -> Code:
    """Close a body over everything a rule is given, including itself."""
    rebound: Any = body
    for _ in range(4):
        rebound = Code("given a thing", parts=(rebound,))
    step = L(
        "itself",
        L(
            "first",
            L(
                "second",
                A(rebound, V("itself"), V("first"), V("many"), V("cells")),
            ),
        ),
    )
    inner = L(
        "x",
        A(step, L("a", L("b", A(V("x"), V("x"), V("a"), V("b"))))),
    )
    return build(
        L(
            "where",
            L(
                "many",
                L("cells", A(A(inner, inner), V("where"), V("many"))),
            ),
        )
    )


@dataclass(frozen=True)
class Rule:
    """A rule with no shape but the one its term has."""

    body: Code
    #: Lengths it was fitted at, and lengths it was only judged at.
    fitted_at: tuple[int, ...] = ()
    judged_at: tuple[int, ...] = ()
    #: Whether it puts a value into the state that was never in it. The case
    #: language_limits refuses to rule on, because no rule about where a cell
    #: came from can produce it.
    makes_new_values: bool = False

    @property
    def how_long(self) -> int:
        return how_long(self.body)

    def describes(self) -> str:
        said = " and it makes values that were not there" if self.makes_new_values else ""
        return (
            f"{self.how_long} symbols, fitted at {self.fitted_at}, held to "
            f"{self.judged_at}{said}"
        )

    def read(self, cells: Sequence[Any]) -> tuple[Any, ...] | None:
        """The state this rule turns these cells into, or nothing where it cannot."""
        found = tuple(cells)
        size = len(found)
        if size == 0:
            return ()
        try:
            numbers = [int(one) for one in found]
        except (TypeError, ValueError):
            # A rule here computes over numbers. A state of something else is
            # a state this rule has nothing to say about, and saying nothing
            # is what keeps it safe to put in the language.
            return None
        table = from_list(numbers)
        closed = as_a_rule(self.body)
        out: list[Any] = []
        for at in range(size):
            said = _ask(closed, at, size, table)
            if said is None:
                return None
            out.append(said)
        return tuple(out)


def _ask(closed: Code, at: int, size: int, table: Any) -> int | None:
    """Run a closed rule at one place, or nothing where it refuses."""
    try:
        work = run(closed, fuel=_A_RULE_MAY_SPEND)
        for one in (at, size, table):
            work = run(work.body, (one, *work.env), fuel=_A_RULE_MAY_SPEND)
        return int(work)
    except (OutOfFuel, Stuck, RecursionError, TypeError, ValueError, AttributeError):
        return None


def _numbers_the_states_show(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> tuple[int, ...]:
    """Constants the examples put on the table. Solved for, never searched."""
    found: set[int] = {0, 1}
    for before, after in transitions:
        found.add(len(before))
        for one, other in zip(before, after, strict=False):
            try:
                found.add(int(other) - int(one))
            except (TypeError, ValueError):
                continue
    return tuple(sorted(one for one in found if -16 <= one <= 64))


def _makes_new_values(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> bool:
    """Whether a cell comes out that never went in."""
    for before, after in transitions:
        if set(after) - set(before):
            return True
    return False


def _where_it_could_read(
    states: Sequence[tuple[list[int], list[int]]], *, deepest: int = 6
) -> list[tuple[Code, list[list[int]]]]:
    """The distinct places a short expression could name, shortest first.

    Four thousand expressions over where it is and how long the state is name
    a few dozen distinct sets of places between them: `here`, `one along`, the
    far end, half way, and a handful more. Walking all four thousand and
    inverting an operation against each was the whole cost of this search —
    one case took longer than the seven together take now.

    So they are deduplicated by what they NAME rather than by how they are
    spelled, and the shortest spelling of each is kept. That is the same
    argument `one_thing_many_spellings` makes about words, one level down.
    """
    seen: set[tuple[tuple[int, ...], ...]] = set()
    found: list[tuple[Code, list[list[int]]]] = []
    for candidate in every_code(deepest=deepest, variables=2, constants=(0, 1, 2)):
        named = _the_places_it_names(candidate, states)
        if named is None:
            continue
        signature = tuple(tuple(one) for one in named)
        if signature in seen:
            continue
        seen.add(signature)
        found.append((candidate, named))
        if len(found) >= _HOW_MANY_PLACES_ARE_WORTH_TRYING:
            break
    return found


def _the_places_it_names(
    where: Code, states: Sequence[tuple[list[int], list[int]]]
) -> list[list[int]] | None:
    """Which cell this expression names at every place of every state."""
    named: list[list[int]] = []
    for before, _after in states:
        size = len(before)
        here: list[int] = []
        for at in range(size):
            try:
                said = run(where, (size, at), fuel=_A_RULE_MAY_SPEND)
            except (OutOfFuel, Stuck, RecursionError, TypeError, ValueError):
                return None
            if not isinstance(said, int):
                return None
            here.append(said % max(1, size))
        named.append(here)
    return named


def _the_places_the_answers_name(
    states: Sequence[tuple[list[int], list[int]]],
) -> list[list[int]] | None:
    """Where each answer already sits in the state before it, if it sits anywhere.

    Read straight off the examples rather than searched for, which is the same
    move `language_limits` makes and the same one `_where_each_came_from`
    makes. For a rule that only moves cells this pins the source exactly, and
    an enumeration that has to reach `the far end` by spelling it out walks
    thousands of expressions to arrive at what one pass over the data says.
    """
    named: list[list[int]] = []
    for before, after in states:
        here: list[int] = []
        for said in after:
            where = [at for at, one in enumerate(before) if one == said]
            if len(where) != 1:
                return None
            here.append(where[0])
        named.append(here)
    return named


def _what_is_done_with_it(
    states: Sequence[tuple[list[int], list[int]]],
    named: Sequence[list[int]],
    carried: Sequence[list[int]] | None,
    *,
    deepest: int = 2,
) -> tuple[Any, list[list[int]]] | None:
    """What turns this cell, and what was carried, into the answer.

    Inverted rather than searched. `an_operation_that_generalises` asks what
    the other side would have had to be and answers in one pass, which is why
    the number of sources can be left open — each one costs a solve rather
    than a walk.
    """
    from core.cognition.an_operation_that_generalises import (
        an_operation_that_generalises,
    )

    pairs: list[tuple[int, int, int]] = []
    for at_state, (before, after) in enumerate(states):
        for at in range(len(before)):
            value = before[named[at_state][at]]
            other = carried[at_state][at] if carried is not None else at
            pairs.append((value, other, after[at]))
    rule = an_operation_that_generalises(pairs, deepest=deepest)
    if rule is None:
        return None
    made: list[list[int]] = []
    for at_state, (before, _after) in enumerate(states):
        here = []
        for at in range(len(before)):
            value = before[named[at_state][at]]
            other = carried[at_state][at] if carried is not None else at
            try:
                here.append(int(rule(value, other)))
            except (ArithmeticError, TypeError, ValueError):
                return None
        made.append(here)
    return rule, made


def _the_cells_at(
    states: Sequence[tuple[list[int], list[int]]], named: Sequence[list[int]]
) -> list[list[int]]:
    return [
        [before[named[at_state][at]] for at in range(len(before))]
        for at_state, (before, _after) in enumerate(states)
    ]


def _as_a_body(
    stages: Sequence[tuple[Code, Code]],
) -> Code:
    """The stages as one floor term over what a rule is given.

    Each stage reads a cell and combines it with what the stages before it
    made. The first has nothing carried, so it is combined with where it is.
    """
    made: Code | None = None
    for where, operation in stages:
        # The place is brought back inside the state before it is read.
        #
        # A solved place expression can be negative — the mirror comes back as
        # `nought minus one more than where it is`, which names the right cell
        # only once it is taken modulo the length. Reading a list at a negative
        # place walks past the end of it and never stops, so the head that
        # reads the state gets a place that is already inside.
        cell = Code(
            "of",
            parts=(
                Code("of", parts=(THE_CELL_AT, Code("the one it was given", value=_THE_STATE))),
                Code(
                    "left over",
                    parts=(
                        _over_a_rules_bindings(where),
                        Code("the one it was given", value=_HOW_LONG),
                    ),
                ),
            ),
        )
        carried = made if made is not None else Code(
            "the one it was given", value=_WHERE
        )
        made = Code("of", parts=(Code("of", parts=(operation, cell)), carried))
    if made is None:  # pragma: no cover - a rule with no stage is refused above
        raise ValueError("a rule with no stages")
    return made


def _over_a_rules_bindings(where: Code) -> Code:
    """An index expression written over (how long, where) as a rule sees them."""
    if where.head == "the one it was given":
        which = int(where.value or 0)
        return Code(
            "the one it was given",
            value=_HOW_LONG if which == 0 else _WHERE,
        )
    return Code(
        where.head,
        parts=tuple(_over_a_rules_bindings(part) for part in where.parts),
        value=where.value,
    )


def a_rule_she_wrote(
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
    *,
    now_sayable: Any,
    already: Sequence[Code] = (),
    most_sources: int = 3,
    within: float = 20.0,
) -> Rule | None:
    """Write a rule for the family in front of her, with no shape assumed.

    Built one source at a time. Read a cell, work out what is done with it,
    and where that does not account for the family read another and fold it
    in. Nothing fixes how many sources there are — the loop stops when the
    family is accounted for, and ``most_sources`` is a budget on this call
    rather than a statement about rules.

    Neither half is walked blindly. Where the answers already sit in the state
    is read straight off the examples; where they do not, the places are short
    expressions over how long the state is and where it is, deduplicated by
    what they NAME rather than by how they are spelled. What is done with a
    cell is INVERTED by `an_operation_that_generalises` rather than searched
    for. Blind enumeration over four bindings runs past four million terms at
    depth six and finds none of these, which is measured rather than assumed.
    """
    if now_sayable():
        return None
    numbered: list[tuple[list[int], list[int]]] = []
    for before, after in transitions:
        try:
            numbered.append(([int(one) for one in before], [int(one) for one in after]))
        except (TypeError, ValueError):
            return None
        if len(numbered[-1][0]) != len(numbered[-1][1]):
            return None
    if len(numbered) < 4:
        # Two to fit on and two to be judged by. One of each is a coincidence.
        return None

    fitting, judging = numbered[0::2], numbered[1::2]
    began = time.monotonic()
    reading = _where_it_could_read(fitting)

    # Where the answers already sit goes first, and it is one pass over the
    # data rather than a walk.
    off_the_data = _the_places_the_answers_name(fitting)
    starts: list[tuple[Code | None, list[list[int]]]] = []
    if off_the_data is not None:
        starts.append((None, off_the_data))
    starts.extend((where, named) for where, named in reading)

    # One source first, over every start, before any start is given a second.
    #
    # Ordering matters more than it looks. Reading the answers off the data
    # gives a start that is often wrong on a family that makes values, and
    # giving that wrong start a second and third source walks the whole
    # reading list twice before the right start is ever tried — which is a
    # family's entire budget spent on the first candidate. Cheap routes over
    # every start, then the expensive one: the same discipline the ladder in
    # sequence_induction already follows.
    from core.cognition.an_operation_that_generalises import Expression

    spelled_starts: list[tuple[Code, list[list[int]]]] = []
    for where_first, named_first in starts:
        if time.monotonic() - began >= within:
            logger.info("gave up writing a rule after %.1fs", within)
            return None
        spelled = (
            where_first
            if where_first is not None
            else _spell(named_first, reading, fitting)
        )
        if spelled is None:
            continue
        spelled_starts.append((spelled, named_first))
        # One combination, over every start. Asking for a nest of them here
        # costs a quarter of a second per start and there are hundreds, so the
        # whole budget goes on the cheapest route before the fold is ever
        # reached — measured: a family needing two sources timed out with the
        # answer six places into a fold that never ran.
        done = _what_is_done_with_it(fitting, named_first, None, deepest=1)
        if done is None:
            continue
        operation, carried = done
        if all(carried[at] == after for at, (_b, after) in enumerate(fitting)):
            found = _finish([(spelled, operation)], fitting, judging, transitions)
            if found is not None:
                return found

    # Folding is the expensive route and it is bounded on both sides. A second
    # source is read at a place with a short name, and what joins it to what
    # was carried is one combination rather than a nest of them — so the fold
    # walks the shortest places and asks for a depth-one operation. Both are
    # budgets on this call, and a family that needs more than they allow comes
    # back unsolved rather than slowly.
    for spelled, named_first in spelled_starts[:_HOW_MANY_STARTS_ARE_WORTH_FOLDING]:
        carried = _the_cells_at(fitting, named_first)
        stages: list[tuple[Code, Any]] = [(spelled, Expression("the first"))]
        for _ in range(max(1, int(most_sources)) - 1):
            folded = None
            for where_next, named_next in reading[:_HOW_MANY_PLACES_ARE_WORTH_FOLDING]:
                if time.monotonic() - began >= within:
                    logger.info("gave up writing a rule after %.1fs", within)
                    return None
                done = _what_is_done_with_it(
                    fitting, named_next, carried, deepest=1
                )
                if done is None:
                    continue
                operation, made = done
                if all(made[at] == after for at, (_b, after) in enumerate(fitting)):
                    folded = (where_next, operation, made)
                    break
            if folded is None:
                break
            where_next, operation, carried = folded
            stages.append((where_next, operation))
            found = _finish(stages, fitting, judging, transitions)
            if found is not None:
                return found

    # And a deeper single source last, so bounding the first pass to one
    # combination loses nothing rather than trading a family for a budget.
    #
    # Two levels, not three. The clock is only read between starts, and one
    # solve at three levels can run for minutes on a leaf set that is large
    # because none of the differences recur — a family that should have come
    # back unsolved in twenty-five seconds took nine minutes.
    for spelled, named_first in spelled_starts:
        if time.monotonic() - began >= within:
            return None
        done = _what_is_done_with_it(fitting, named_first, None, deepest=2)
        if done is None:
            continue
        operation, carried = done
        if all(carried[at] == after for at, (_b, after) in enumerate(fitting)):
            found = _finish([(spelled, operation)], fitting, judging, transitions)
            if found is not None:
                return found
    return None


def _spell(
    named: Sequence[list[int]],
    reading: Sequence[tuple[Code, list[list[int]]]],
    states: Sequence[tuple[list[int], list[int]]] | None = None,
) -> Code | None:
    """An expression naming these places.

    Looked up among the ones already walked, and where none of them names
    these places, SOLVED for: the places are a function of how long the state
    is and where it is, so what turns those two into a place is the same
    question `an_operation_that_generalises` answers by inverting.

    That matters for the commonest rule there is. `the far end` is
    ``how long minus one minus where``, five symbols, and a shortest-first
    walk reaches two hundred and forty distinct sets of places before it gets
    there — so a mirror was unreachable by looking and is immediate by
    solving.
    """
    wanted = [list(one) for one in named]
    for where, said in reading:
        if said == wanted:
            return where
    if states is None:
        return None
    from core.cognition.an_operation_that_generalises import (
        an_operation_that_generalises,
    )
    from core.cognition.the_old_language_on_the_floor import compile_an_operation

    pairs: list[tuple[int, int, int]] = []
    for at_state, (before, _after) in enumerate(states):
        for at in range(len(before)):
            pairs.append((len(before), at, wanted[at_state][at]))
    # Two, not three. A place is a short function of two numbers, and the
    # third level costs most of a family's whole budget on a leaf set that is
    # large because none of the differences recur — measured: one family spent
    # twenty-five seconds here and found nothing.
    rule = an_operation_that_generalises(pairs, deepest=2)
    if rule is None:
        return None
    closed = compile_an_operation(rule)
    body = closed
    while body.head == "given a thing":
        body = body.parts[0]
    # Compiled, the pair sits with the second at nought and the first at one;
    # a place is read with how long at nought and where at one. One swap.
    return _the_other_way_round(body)


def _the_other_way_round(code: Code) -> Code:
    if code.head == "the one it was given":
        which = int(code.value or 0)
        if which in (0, 1):
            return Code("the one it was given", value=1 - which)
        return code
    return Code(
        code.head,
        parts=tuple(_the_other_way_round(part) for part in code.parts),
        value=code.value,
    )


def _finish(
    stages: Sequence[tuple[Code, Any]],
    fitting: Sequence[tuple[list[int], list[int]]],
    judging: Sequence[tuple[list[int], list[int]]],
    transitions: Sequence[tuple[Sequence[Any], Sequence[Any]]],
) -> Rule | None:
    """Build the rule and hold it to the half it never saw."""
    from core.cognition.the_old_language_on_the_floor import compile_an_operation

    try:
        body = _as_a_body(
            [(where, compile_an_operation(operation)) for where, operation in stages]
        )
    except (ValueError, RecursionError):
        return None
    found = Rule(
        body=body,
        fitted_at=tuple(len(before) for before, _ in fitting),
        judged_at=tuple(len(before) for before, _ in judging),
        makes_new_values=_makes_new_values(transitions),
    )
    if not all(found.read(before) == tuple(after) for before, after in judging):
        # It fitted the half it saw and not the half it did not, which is what
        # a table always does.
        return None
    logger.info("she wrote a rule with no shape: %s", found.describes())
    return found


def the_rule_written_down(rule: Rule) -> dict[str, Any]:
    """The rule as plain data, so what she wrote survives a restart."""
    return {
        "body": written_down(rule.body),
        "fitted_at": list(rule.fitted_at),
        "judged_at": list(rule.judged_at),
        "makes_new_values": bool(rule.makes_new_values),
    }


def read_a_rule_back(row: Any) -> Rule | None:
    """A rule from what was written down, or nothing where it does not read."""
    if not isinstance(row, dict):
        return None
    body = read_back(row.get("body"))
    if body is None:
        return None
    return Rule(
        body=body,
        fitted_at=tuple(int(one) for one in row.get("fitted_at") or ()),
        judged_at=tuple(int(one) for one in row.get("judged_at") or ()),
        makes_new_values=bool(row.get("makes_new_values")),
    )
