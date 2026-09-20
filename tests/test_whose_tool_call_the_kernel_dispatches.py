"""A tool call raised inside a person's turn is the person's tool call.

The kernel's dispatch had its own list of origins that count as a person —
the fourteenth in the tree — and it lacked `desktop`, `chat` and every
sub-route. A chat turn that reached web_search from one of them presented
as `godmode_phase`; the Will, correctly, refused an unattributed tool call
for lack of authority; and the person's question went unanswered. The
shared predicate answers now, and the turn bound to the context knows what
it started as.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit


def _kernel_with_origin(origin: str):
    from core.kernel.upgrades_10x import GodModeToolPhase

    executor = GodModeToolPhase.__new__(GodModeToolPhase)
    state = SimpleNamespace(cognition=SimpleNamespace(current_origin=origin))
    return executor, state


@pytest.mark.parametrize("origin", ["user", "chat", "chat_api", "desktop", "desktop-ui", "voice", "api"])
def test_a_persons_origin_is_the_source(origin: str) -> None:
    executor, state = _kernel_with_origin(origin)
    assert executor._resolve_tool_source(state) == executor._normalize_origin(origin)


@pytest.mark.parametrize("origin", ["curiosity_engine", "stream_narrative", "system", ""])
def test_her_own_loops_stay_her_own(origin: str) -> None:
    executor, state = _kernel_with_origin(origin)
    assert executor._resolve_tool_source(state) == "godmode_phase"


def test_a_phase_name_defers_to_the_bound_turn() -> None:
    """The state carries a phase name nobody can parse; the turn knows."""
    from core.runtime.turn_outcome import TurnOutcome, bind_turn

    executor, state = _kernel_with_origin("response_generation")
    with bind_turn(TurnOutcome(origin="user")):
        assert executor._resolve_tool_source(state) == "user"
    assert executor._resolve_tool_source(state) == "godmode_phase"
