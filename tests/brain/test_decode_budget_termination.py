"""Running out of layer-app budget mid-decode is a bounded stop, not a failure.

LIVE, 2026-08-10, on a plain conversational question ("would you rather be shut
down cleanly and restored exactly, or keep running with a slow drift you can't
detect?"):

    CognitiveEngine retained single ownership of the failed turn
    (failure_class=decode_incomplete:budget_exhausted)

and the person got "I couldn't get to an answer I'd stand behind on that one."

Three budget dimensions bound a decode — tokens, wall clock, and layer
applications. The accepted-termination set listed the first two and omitted the
third, so the same event killed the turn depending only on which meter ran out.
The set's own comment states the principle it was breaking:

    "A time-bounded stop has the same epistemic status as a token-bounded one:
     the product-quality gate — terminal completeness, facet and subject
     coverage — judges whether the text stands as an answer, not the budget
     dimension that ended sampling."

The one real distinction is whether any tokens were produced. Exhausting the
budget BEFORE the first token yields nothing to judge, so it keeps its own
termination and stays a failure.
"""

from __future__ import annotations

from core.brain.llm.latent_cortex import engine
from tests.source_contract import family_text


def _accepted_terminations_source() -> str:
    source = family_text(engine)
    marker = "if not failure_reason and receipt.decode_termination not in {"
    start = source.find(marker)
    assert start != -1, "the accepted-termination set moved"
    return source[start : source.find("}:", start)]


def test_mid_decode_budget_exhaustion_is_an_accepted_stop() -> None:
    accepted = _accepted_terminations_source()

    assert '"budget_exhausted",' in accepted


def test_the_other_two_budget_dimensions_are_still_accepted() -> None:
    """The fix is to complete the set, not to replace its members."""
    accepted = _accepted_terminations_source()

    assert '"token_limit",' in accepted
    assert '"wall_reserve_sentence_grace",' in accepted


def test_a_wall_stop_with_no_sentence_boundary_is_not_accepted() -> None:
    """Raw wall_reserve now means the reserve was crossed mid-clause. The
    text is a fragment, so it cannot also be a time-bounded answer."""
    accepted = _accepted_terminations_source()

    assert '"wall_reserve",' not in accepted


def test_the_wind_down_decides_which_kind_the_reserve_emits() -> None:
    import ast

    source = family_text(engine)
    tree = ast.parse(source)

    for node in ast.walk(tree):
        if not isinstance(node, ast.IfExp):
            continue
        rendered = ast.get_source_segment(source, node) or ""
        if '"wall_reserve_sentence_grace"' not in rendered:
            continue
        assert "sentence_done" in rendered
        assert '"wall_reserve"' in rendered
        return
    raise AssertionError("the wall-reserve termination choice was not found")


def test_exhaustion_before_the_first_token_is_not_accepted() -> None:
    """Nothing was sampled, so there is no text for the product gate to judge."""
    accepted = _accepted_terminations_source()

    assert "budget_exhausted_before_decode" not in accepted


def test_the_pre_decode_path_uses_its_own_termination() -> None:
    source = family_text(engine)

    assert 'return out, "budget_exhausted_before_decode"' in source
    assert 'return out, "budget_exhausted"' not in source
