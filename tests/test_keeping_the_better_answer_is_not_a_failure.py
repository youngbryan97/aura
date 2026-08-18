"""Doing the right thing got the turn refused and the person got an apology.

LIVE 2026-08-18. Two turns in a row served "I couldn't get to an answer I'd
stand behind on that one" while the header read REPLY PATH BLOCKED — and the
conversation lane reported state=ready, conversation_ready=true, blockers=[].
Nothing was broken.

The diagnostic added the day before named the cause outright:

    missing: response_path:cognitive_engine_completion_incumbent,
             duplicate_foreground_model_generation
    path=cognitive_engine_completion_incumbent generations=3
    completion_retries=2 repair_attempts=0 consumed=True

chat.py sets that path when completion_incumbent_preserved is true: a
completion retry ran, was compared against the original, and the ORIGINAL won.
That is the retry machinery at its best. The path appeared in neither
accepted_full_mind_response_paths nor the single-owner clause, so the answer
already judged the better of two could not be served.

Three generations for one incumbent plus two retries is exactly
1 + completion_retry_count, so the generation arithmetic already agreed. Only
the name was unrecognised — a response path with tests asserting it is SET and
nothing authorising it, which is this codebase's most repeated shape.
"""
from __future__ import annotations

import inspect

import pytest

from interface.routes import chat_turn_contract as contract


def _source() -> str:
    return inspect.getsource(contract)


@pytest.mark.parametrize(
    "path",
    [
        "cognitive_engine_completion_retry",
        "cognitive_engine_completion_incumbent",
    ],
)
def test_both_completion_outcomes_are_accepted_full_mind_paths(path):
    """Adopting the retry and keeping the incumbent are both real answers."""
    source = _source()
    start = source.index("accepted_full_mind_response_paths = {")
    block = source[start : source.index("}", start)]

    assert f'"{path}"' in block, path


@pytest.mark.parametrize(
    "path",
    [
        "cognitive_engine_completion_retry",
        "cognitive_engine_completion_incumbent",
    ],
)
def test_both_completion_outcomes_prove_single_ownership(path):
    """Which answer won says nothing about who authored them.

    One owner generated once and then continued; the comparison's verdict does
    not change the ownership story, and naming only one outcome meant the
    better one could not be served.
    """
    source = _source()
    start = source.index("single_owner_model_generation_proven")
    block = source[start : start + 1800]

    assert f'"{path}"' in block, path


def test_the_incumbent_path_is_the_one_chat_actually_sets():
    """Guards the literal. A rename here silently re-breaks the refusal."""
    from interface.routes import chat

    assert "cognitive_engine_completion_incumbent" in inspect.getsource(chat)


def test_a_genuinely_unknown_path_is_still_refused():
    """Widening the accepted set must not become accepting anything."""
    source = _source()
    start = source.index("accepted_full_mind_response_paths = {")
    block = source[start : source.index("}", start)]

    for never in ("bounded_contract", "legacy_fallback", "repair_text"):
        assert f'"{never}"' not in block, never


def test_the_generation_arithmetic_is_unchanged():
    """The count rule is what makes ownership provable; it must not loosen."""
    source = _source()
    start = source.index("single_owner_model_generation_proven")
    block = source[start : start + 1800]

    assert "foreground_model_generation_count == 1 + completion_retry_count" in block
    assert "completion_retry_count <= _MAX_USER_SURFACE_CONTINUATIONS" in block
