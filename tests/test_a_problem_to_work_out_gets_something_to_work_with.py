"""A problem with an exact answer is offered the means to compute it.

LIVE, 2026-08-20. Six people at a round table, four constraints, one
arrangement. She answered by narration: the opposite-of-Chen half was right,
the neighbours half was wrong, and the layout she stated contradicted the
conclusion she drew from it.

She holds a Python sandbox. It was offered nothing, because the tool set is
gated on the turn asking for a CAPABILITY — "read this", "search that" — and
a problem to work out asks for none.
"""

from __future__ import annotations

import pytest

from core.intent.declared_capability import settles_by_computation

PUZZLE = (
    "six people sit around a round table with six seats. Boris sits directly "
    "opposite Ada. Chen and Dara sit next to each other. Emil sits immediately "
    "clockwise of Ada. Chen is exactly two seats from Ada. Who sits opposite "
    "Chen, and who are Dara's two neighbours?"
)


def test_the_puzzle_that_was_answered_by_narration() -> None:
    assert settles_by_computation(PUZZLE) is True


@pytest.mark.parametrize(
    "message",
    [
        "how are you feeling today?",
        "what is 2+2",
        "tell me about Solaris",
        "",
        "read /etc/hosts and tell me the first line",
    ],
)
def test_conversation_is_left_alone(message: str) -> None:
    assert settles_by_computation(message) is False


def _catalogue():
    from core.skills.discovery import build_skill_catalog

    class _Meta:
        def __init__(self, declaration):
            self.description = declaration.description
            self.effect_scope = declaration.effect_scope
            self.enabled = True
            self.name = declaration.name
            self.module_path = declaration.module_path
            self.class_name = declaration.class_name
            self.skill_class = None
            self.instance = None

    return {d.name: _Meta(d) for d in build_skill_catalog().accepted}


def _selected(objective: str) -> list[str]:
    from core.intent.capability_selection import select_capabilities

    return select_capabilities(
        objective,
        _catalogue(),
        ceiling="sandboxed_compute",
        admissible_scopes=frozenset({"status", "pure_compute", "read_only", "sandboxed_compute"}),
    )


def test_the_puzzle_is_offered_something_that_can_compute() -> None:
    offered = set(_selected(PUZZLE))
    assert offered & {"code_repl", "run_code", "internal_sandbox"}


def test_it_is_not_offered_the_web() -> None:
    """Searching for a seating puzzle is noise, and context is not free."""
    offered = set(_selected(PUZZLE))
    assert not offered & {"web_search", "free_search", "grounded_search", "http_request"}


def test_an_ordinary_conversational_turn_is_still_offered_nothing() -> None:
    assert _selected("how are you feeling today?") == []


def test_a_read_request_keeps_its_own_set() -> None:
    offered = set(_selected("read /etc/hosts and tell me the first line"))
    assert "http_request" in offered or "file_operation" in offered
