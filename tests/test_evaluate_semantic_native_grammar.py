"""Complete graph selection sees programs and scores, never their targets."""

from types import SimpleNamespace

import pytest

from tools.evaluate_semantic_native_grammar import select_search_proposal


def search(programs):
    return SimpleNamespace(candidates=tuple(SimpleNamespace(
        result=SimpleNamespace(program=program)) for program in programs))


def test_every_program_is_scored_before_choosing_a_later_alternative():
    calls = []
    def score(program):
        calls.append(program)
        return {"a": -3., "b": -1., "c": -2.}[program]
    assert select_search_proposal(search(("a", "b", "c")), score) == (1, (-3., -1., -2.))
    assert calls == ["a", "b", "c"]


def test_empty_search_does_not_invent_a_candidate():
    assert select_search_proposal(search(()), lambda _: pytest.fail("no graph exists")) == (None, ())


@pytest.mark.parametrize("invalid", [True, None, float("nan"), float("inf")])
def test_unmeasured_graphs_cannot_be_skipped_to_claim_a_winner(invalid):
    with pytest.raises(ValueError, match="finite"):
        select_search_proposal(search(("a", "b")), lambda program: 1. if program == "a" else invalid)
