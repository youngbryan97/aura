"""Every positional term computes on the floor what it computes in its own interpreter.

`one_algebra` is called one algebra because a word and a way of building words
are the same kind of thing. Two algebras remained: positional terms run by
`one_algebra.run`, and value expressions run by their own caller. This closes
half of it — the floor becomes the semantics and the positional algebra becomes
a specialised way of writing some of it, which is what lets a head, a word and
a proposal rule all be objects of one kind.

Structural equality is not the test. Behaviour is, everywhere the pair can be
asked, and a term that refuses in one language must refuse in the other. That
last clause is not decoration: it is what found the one real disagreement.
Substituting the inner expression into the body of `through` made the floor
lazy where the interpreter is strict, so a term dividing by nothing refused in
one language and answered nought in the other. Binding it restored the
strictness.
"""

from __future__ import annotations

import itertools

import pytest

from core.cognition.one_algebra import Term, every_term
from core.cognition.the_old_language_on_the_floor import (
    THE_WORDS_SHE_WAS_GIVEN,
    as_a_floor_term,
    compile_positional,
    they_agree_everywhere,
)


def _a_sample(how_many: int = 2400) -> list[Term]:
    return list(
        itertools.islice(every_term((0, 1, 2, 3), holes=2, deepest=3), how_many)
    )


@pytest.mark.parametrize(
    "pair",
    [("here", "one along"), ("the far end", "its partner"), ("one back", "here")],
)
def test_the_two_languages_agree_everywhere(pair) -> None:
    found = they_agree_everywhere(_a_sample(), pair, sizes=(2, 3, 4, 5, 6, 7))
    assert found["checked"] > 50_000
    assert found["agree"], found["apart"][:5]


def test_every_head_the_interpreter_runs_is_covered_by_the_sample() -> None:
    """A compiler checked on a sample that misses a head proves nothing there."""
    import inspect
    import re

    from core.cognition.one_algebra import HEADS
    from core.cognition.one_algebra import run as positional_run

    source = inspect.getsource(positional_run)
    dispatched = set(re.findall(r'head == "([^"]+)"', source)) | set(HEADS)
    dispatched.discard("hole")  # a hole is a word, and words fill the sample
    seen = {term.head for term in _a_sample()}
    assert dispatched <= seen, sorted(dispatched - seen)


def test_a_refusal_in_one_language_is_a_refusal_in_the_other() -> None:
    """The clause that found the only real disagreement."""
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.one_algebra import run as positional_run
    from core.cognition.the_floor_she_stands_on import Code, Stuck
    from core.cognition.the_floor_she_stands_on import run as run_on_the_floor

    # `through(many, over(where, where))` divides by nothing at position nought
    # and the outer part ignores its position, so a lazy compilation answers
    # where the interpreter refuses.
    term = Term(
        "through",
        parts=(Term("many"), Term("over", parts=(Term("where"), Term("where")))),
    )
    with pytest.raises(ZeroDivisionError):
        positional_run(term, 0, 5, (WHERE_FROM["here"], WHERE_FROM["one along"]))
    compiled = as_a_floor_term(term, ("here", "one along"))
    with pytest.raises(Stuck):
        run_on_the_floor(
            Code(
                "of",
                parts=(
                    Code("of", parts=(compiled, Code("a number", value=0))),
                    Code("a number", value=5),
                ),
            )
        )


def test_the_words_she_was_given_compile_to_what_they_compute() -> None:
    from core.cognition.an_invented_kind import WHERE_FROM
    from core.cognition.the_floor_she_stands_on import Code
    from core.cognition.the_floor_she_stands_on import run as run_on_the_floor

    assert set(THE_WORDS_SHE_WAS_GIVEN) == set(WHERE_FROM)
    for name, compiled in THE_WORDS_SHE_WAS_GIVEN.items():
        for size in (2, 3, 4, 5, 8):
            for at in range(size):
                got = int(
                    run_on_the_floor(
                        Code(
                            "of",
                            parts=(
                                Code("of", parts=(compiled, Code("a number", value=at))),
                                Code("a number", value=size),
                            ),
                        )
                    )
                ) % size
                assert got == int(WHERE_FROM[name](at, size)) % size, (name, at, size)


def test_a_compiled_term_is_a_term_of_the_floor_and_nothing_else() -> None:
    from core.cognition.the_floor_she_stands_on import HOW_MANY_PARTS

    def heads(code):
        yield code.head
        for part in code.parts:
            yield from heads(part)

    for term in itertools.islice(every_term((0, 1, 2), holes=2, deepest=2), 400):
        compiled = compile_positional(
            term,
            [THE_WORDS_SHE_WAS_GIVEN["here"], THE_WORDS_SHE_WAS_GIVEN["one along"]],
        )
        assert set(heads(compiled)) <= set(HOW_MANY_PARTS)


def test_a_head_the_compiler_does_not_know_is_refused_rather_than_guessed() -> None:
    with pytest.raises(ValueError):
        compile_positional(
            Term("something nobody wrote"),
            [THE_WORDS_SHE_WAS_GIVEN["here"]],
        )
