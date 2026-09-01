"""The floor, its evaluator written in itself, and the ceiling it replaces.

Four things have to hold, and the fourth is the one the whole argument rests
on.

1. The machine computes what the heads say it computes, and refuses rather
   than hangs.
2. A term is a value and comes back as itself, in memory and across a restart.
3. The interpreter written IN the floor agrees with the machine that runs the
   floor. Without that, "the mechanism is an object of the language" is a
   slogan.
4. The positional algebra she thinks in cannot say something the floor says,
   which is what makes moving to the floor a gain rather than a preference.
"""

from __future__ import annotations

import ast
import inspect
import pathlib
import re

import pytest

from core.cognition import the_floor_she_stands_on as floor
from core.cognition.the_floor_she_stands_on import (
    A,
    ARITHMETIC,
    BELOW,
    Code,
    HOW_MANY_PARTS,
    IF,
    ISPAIR,
    L,
    LET,
    MINUS,
    N,
    NIL,
    NOTHING,
    OutOfFuel,
    PAIR,
    PLUS,
    QUOTE,
    SAME,
    SIGNATURE,
    Stuck,
    TIMES,
    V,
    Y,
    FST,
    SND,
    Pair,
    build,
    decode,
    encode,
    every_code,
    how_long,
    read_back,
    run,
    steps_taken,
    written_down,
)

_FACTORIAL = build(
    Y(
        "fac",
        L(
            "n",
            IF(
                SAME(V("n"), N(0)),
                N(1),
                TIMES(V("n"), A(V("fac"), MINUS(V("n"), N(1)))),
            ),
        ),
    )
)


# ── 1. the machine ────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("named", "want"),
    [
        (PLUS(N(2), N(3)), 5),
        (MINUS(N(2), N(9)), -7),
        (TIMES(N(6), N(7)), 42),
        (BELOW(N(1), N(2)), 1),
        (BELOW(N(2), N(1)), 0),
        (SAME(N(4), N(4)), 1),
        (IF(N(0), N(10), N(20)), 20),
        (IF(N(5), N(10), N(20)), 10),
        (FST(PAIR(N(1), N(2))), 1),
        (SND(PAIR(N(1), N(2))), 2),
        (ISPAIR(PAIR(N(1), N(2))), 1),
        (ISPAIR(N(1)), 0),
        (A(L("x", TIMES(V("x"), V("x"))), N(7)), 49),
        (LET("x", N(3), PLUS(V("x"), V("x"))), 6),
    ],
)
def test_the_heads_say_what_they_say(named, want) -> None:
    assert run(build(named)) == want


def test_it_can_call_itself() -> None:
    assert run(build(A(_FACTORIAL, N(10)))) == 3628800


def test_a_term_that_will_not_stop_is_refused_rather_than_waited_on() -> None:
    forever = build(A(Y("go", L("n", A(V("go"), V("n")))), N(1)))
    with pytest.raises(OutOfFuel):
        run(forever, fuel=5000)


def test_the_meter_is_what_a_budget_is_spent_in() -> None:
    cheap = steps_taken(build(A(_FACTORIAL, N(3))))
    dear = steps_taken(build(A(_FACTORIAL, N(30))))
    assert 0 < cheap < dear


def test_it_refuses_rather_than_guessing() -> None:
    with pytest.raises(Stuck):
        run(build(FST(N(1))))
    with pytest.raises(Stuck):
        run(build(A(N(1), N(2))))
    with pytest.raises(Stuck):
        run(build(PLUS(NIL, N(1))))
    with pytest.raises(Stuck):
        run(Code("a head nobody wrote"))
    with pytest.raises(Stuck):
        run(build(_divide_by_nothing()))


def _divide_by_nothing():
    from core.cognition.the_floor_she_stands_on import OVER

    return OVER(N(1), N(0))


def test_deep_terms_do_not_hit_a_python_limit() -> None:
    """The machine is a loop over a stack, and this is why.

    Built as terms rather than through the assembler, which does recurse. The
    assembler is a convenience for writing terms by hand and a hand-written
    term is never four thousand deep; the machine is what runs whatever a
    search produces, and it is the thing that must not fall over.
    """
    piled = Code("a number", value=0)
    for _ in range(4000):
        piled = Code("plus", parts=(piled, Code("a number", value=1)))
    assert run(piled, fuel=200_000) == 4000
    assert how_long(piled) == 8001


# ── 2. a term is a value, and comes back ──────────────────────────────────


@pytest.mark.parametrize("head", sorted(HOW_MANY_PARTS))
def test_every_head_encodes_and_decodes(head: str) -> None:
    leaf = Code("nothing")
    parts = tuple(leaf for _ in range(HOW_MANY_PARTS[head]))
    value = 2 if head in {"a number", "the one it was given"} else None
    term = Code(head, parts=parts, value=value)
    assert decode(encode(term)) == term


@pytest.mark.parametrize("head", sorted(HOW_MANY_PARTS))
def test_every_head_survives_a_restart(head: str) -> None:
    leaf = Code("nothing")
    parts = tuple(leaf for _ in range(HOW_MANY_PARTS[head]))
    value = 2 if head in {"a number", "the one it was given"} else None
    term = Code(head, parts=parts, value=value)
    assert read_back(written_down(term)) == term


def test_a_lot_of_terms_round_trip_both_ways() -> None:
    seen = 0
    for term in every_code(deepest=3):
        seen += 1
        assert decode(encode(term)) == term
        assert read_back(written_down(term)) == term
    assert seen > 500


def test_quotation_gives_the_encoding() -> None:
    term = build(PLUS(N(2), N(3)))
    assert run(build(QUOTE(term))) == encode(term)


def test_a_corrupt_row_is_refused() -> None:
    assert read_back({"head": "nothing anybody wrote", "parts": []}) is None
    assert read_back({"head": "if", "parts": []}) is None
    assert read_back(7) is None


def test_the_signature_order_is_a_contract() -> None:
    """Quotation writes a head as its place here, so reordering breaks meaning."""
    assert len(set(SIGNATURE)) == len(SIGNATURE)
    assert set(SIGNATURE) == set(HOW_MANY_PARTS)


# ── 3. the interpreter written in the floor ───────────────────────────────


def test_the_interpreter_agrees_with_the_machine() -> None:
    from core.cognition.the_floor_reading_itself import interpret

    for named in (
        PLUS(N(2), N(3)),
        A(L("x", TIMES(V("x"), V("x"))), N(7)),
        IF(BELOW(N(1), N(2)), N(10), N(20)),
        FST(PAIR(N(1), PAIR(N(2), NIL))),
        ISPAIR(NIL),
        LET("x", N(4), MINUS(V("x"), N(9))),
    ):
        term = build(named)
        assert interpret(term) == run(term), repr(term)


def test_the_interpreter_runs_a_program_that_calls_itself() -> None:
    from core.cognition.the_floor_reading_itself import interpret

    term = build(A(_FACTORIAL, N(6)))
    assert interpret(term, fuel=4_000_000) == run(term) == 720


def test_the_interpreter_is_a_term_of_the_floor_and_nothing_else() -> None:
    from core.cognition.the_floor_reading_itself import THE_INTERPRETER

    def heads(code: Code):
        yield code.head
        for part in code.parts:
            yield from heads(part)

    assert set(heads(THE_INTERPRETER)) <= set(HOW_MANY_PARTS)
    assert how_long(THE_INTERPRETER) < 1000


def test_the_interpreter_can_be_written_down_and_read_back() -> None:
    from core.cognition.the_floor_reading_itself import THE_INTERPRETER

    assert read_back(written_down(THE_INTERPRETER)) == THE_INTERPRETER
    assert decode(encode(THE_INTERPRETER)) == THE_INTERPRETER


def test_a_term_can_be_handed_a_term_and_take_it_apart() -> None:
    """Homoiconicity, as behaviour rather than as a property of the design."""
    term = build(TIMES(N(3), N(4)))
    head_of = build(FST(QUOTE(term)))
    assert run(head_of) == SIGNATURE.index("times")


# ── 4. what the floor can say, and what the old language cannot ───────────


def test_the_three_starting_points_and_the_three_ways_of_building() -> None:
    from core.cognition.what_the_floor_can_say import (
        SUCCESS,
        ZERO,
        by_recursion,
        take_the_one_at,
        the_least_where,
    )

    def apply(work: Code, *given: int):
        made = work
        for one in given:
            made = Code("of", parts=(made, Code("a number", value=one)))
        return run(made)

    assert apply(ZERO, 9) == 0
    assert apply(SUCCESS, 9) == 10
    assert apply(take_the_one_at(5, 3), 10, 11, 12, 13, 14) == 13

    add = by_recursion(
        build(L("x", V("x"))),
        build(L("x", L("n", L("r", PLUS(V("r"), N(1)))))),
    )
    assert apply(add, 7, 5) == 12
    times = by_recursion(
        build(L("x", N(0))),
        build(L("x", L("n", L("r", PLUS(V("r"), V("x")))))),
    )
    assert apply(times, 7, 5) == 35

    root = the_least_where(build(L("k", MINUS(TIMES(V("k"), V("k")), N(49)))))
    assert run(root) == 7


def test_unbounded_search_is_genuinely_unbounded() -> None:
    """The construct that can fail to stop, failing to stop.

    A search that always terminated would mean the floor computed a strict
    subset of the total functions, and something computable would be outside
    it forever. This test is the difference.
    """
    from core.cognition.what_the_floor_can_say import the_least_where

    never = the_least_where(build(L("k", N(1))))
    with pytest.raises(OutOfFuel):
        run(never, fuel=50_000)


def test_four_of_the_seven_arithmetic_heads_are_not_load_bearing() -> None:
    from core.cognition.what_the_floor_can_say import what_the_arithmetic_rests_on

    found = what_the_arithmetic_rests_on()
    assert found["all_agree"], found["agrees"]
    assert set(found["derived"]) == {"times", "over", "left over", "same as"}
    assert set(found["irreducible"]) | set(found["derived"]) == set(ARITHMETIC)


def test_the_bound_the_ceiling_argument_rests_on_actually_holds() -> None:
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.what_the_old_language_cannot_say import (
        a_sample_of_terms,
        the_bound_holds_on,
    )

    terms = list(a_sample_of_terms(2500))
    words = list(WHERE_FROM.values())[:2]
    found = the_bound_holds_on(terms, words)
    assert found["checked"] > 50_000
    assert found["holds"], found["broken"][:5]


def test_the_induction_has_a_case_for_every_head_the_interpreter_runs() -> None:
    """A proof by cases is worth what its case list is worth."""
    from core.cognition.one_algebra import HEADS as POSITIONAL_HEADS
    from core.cognition.one_algebra import run as positional_run
    from core.cognition.what_the_old_language_cannot_say import (
        the_heads_the_argument_covers,
    )

    source = inspect.getsource(positional_run)
    dispatched = set(re.findall(r'head == "([^"]+)"', source)) | set(POSITIONAL_HEADS)
    assert dispatched == the_heads_the_argument_covers()


def test_doubling_is_outside_the_positional_language_and_inside_the_floor() -> None:
    from core.cognition.what_the_old_language_cannot_say import why_it_cannot_be_said

    found = why_it_cannot_be_said()
    assert found.the_floor_says_it
    assert found.doubling_says > found.the_most_it_could_say
    assert found.strictly_wider
    assert found.on_the_floor < 60


def test_the_escape_point_moves_with_the_length_so_no_length_escapes() -> None:
    """The conclusion is not budget-relative, and this is why."""
    from core.cognition.what_the_old_language_cannot_say import where_doubling_escapes

    points = [where_doubling_escapes(length) for length in (4, 8, 16, 32, 64)]
    assert points == sorted(points)
    for length, size in zip((4, 8, 16, 32, 64), points):
        assert 2**size > max(size, 64, 2) ** length


# ── the floor cannot reach the world ──────────────────────────────────────


def test_nothing_on_the_floor_can_touch_anything() -> None:
    """A term computes over numbers and pairs. Admitting one is not a privilege.

    Checked against the imports rather than argued: the module may not reach
    for anything that opens a file, makes a call, or starts a process.
    """
    source = pathlib.Path(inspect.getfile(floor)).read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module.split(".")[0])
    assert imported <= {"__future__", "dataclasses", "logging", "typing"}, imported
