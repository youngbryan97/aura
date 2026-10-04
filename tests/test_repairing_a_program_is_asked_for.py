"""A request to mend somebody's program reaches the repair, and names the file it means."""
from __future__ import annotations

import re

import pytest

from core.skills.default_trigger_patterns import default_trigger_patterns
from core.skills.repairing_a_program import the_file_named_in

pytestmark = pytest.mark.unit


def _routes(message: str) -> list[str]:
    return [skill for skill, patterns in default_trigger_patterns().items() if any(re.search(p, message, re.IGNORECASE) for p in patterns)]


@pytest.mark.parametrize(
    "message",
    [
        "The Pong game at /Users/bryan/Desktop/pong/pong.html is broken. Fix it, then play it against the computer until you win.",
        "Can you fix this game? It is at ~/Desktop/pong/pong.html",
        "repair the broken page ~/work/index.html",
    ],
)
def test_a_broken_program_reaches_the_repair(message):
    assert "repair_a_program" in _routes(message)


def test_her_own_bug_does_not():
    assert "repair_a_program" not in _routes("fix the bug in your memory")


@pytest.mark.parametrize(
    ("message", "path"),
    [
        ("The Pong game at /Users/bryan/Desktop/pong/pong.html is broken.", "/Users/bryan/Desktop/pong/pong.html"),
        ("fix ~/games/pong.html, please", "~/games/pong.html"),
        ("open file:///tmp/x/game.html and fix it", "/tmp/x/game.html"),
        ("fix my game", ""),
    ],
)
def test_the_file_a_request_names(message, path):
    assert the_file_named_in(message) == path


def test_an_order_to_fix_a_named_file_may_write_files(tmp_path):
    """Asking for a file to be changed is asking for that effect, as asking for one to exist is."""
    from core.phases.response_contract import requested_effect_ceiling

    game = tmp_path / "pong.html"
    game.write_text("<canvas></canvas>")
    assert requested_effect_ceiling(f"The Pong game at {game} is broken. Fix it, then play it.")[0] == "read_write_artifacts"
    assert requested_effect_ceiling(f"What is in {game}?")[0] != "read_write_artifacts"
    assert requested_effect_ceiling(f"I don't want you to change {game}; describe it.")[0] != "read_write_artifacts"


def test_a_turn_that_asks_for_a_repair_can_be_offered_it():
    """Rated by scope alone, a file-writing skill needs a confirmation the turn cannot ask for."""
    from core.brain.inference_gate import _needs_a_confirmation_nobody_can_give

    assert not _needs_a_confirmation_nobody_can_give("repair_a_program", "read_write_artifacts")
