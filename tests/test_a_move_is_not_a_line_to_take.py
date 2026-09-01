"""An approach a single act exhausts cannot be held across the next one.

LIVE 2026-09-01, asked how to go about playing 2048:

  she answered how to go about it: 'I am choosing **right** because the
  "Did you finish applying?" prompt is actively blocking the main feed and
  needs to be dismissed before I can see what's behind it'
  saying out loud: 'Going left — I am about to press left to move away from
  this job feed ...'

Held as the line, announced as the line, and sixteen seconds later she pressed
left. The test for this existed and ran only where she had named nothing
watchable, so a move with a reason attached went past it: the reason supplied
something concrete to watch, and the thing being watched was never asked
whether it was an approach at all.
"""

from __future__ import annotations

import pytest

from core.agency.deliberate_action import ActionOption
from core.agency.standing_strategy import (
    _says_enough_to_be_an_approach,
    read_strategy,
)

pytestmark = pytest.mark.unit

MOVES = [ActionOption(name=name) for name in ("up", "down", "left", "right")]


@pytest.mark.parametrize(
    "said",
    [
        "I am choosing right",
        "I'll go with left",
        "let's press up next",
        "I will take the down move now",
        "right",
    ],
)
def test_naming_which_act_is_a_move(said: str) -> None:
    assert _says_enough_to_be_an_approach(said, MOVES) is False


@pytest.mark.parametrize(
    "said",
    [
        "Push left to keep the top row in order",
        "stack the largest tile into the bottom-left corner",
        "keep the corridor behind me clear so I can retreat",
        "work the bottom row and never lift the corner",
    ],
)
def test_saying_something_about_the_situation_is_a_line(said: str) -> None:
    assert _says_enough_to_be_an_approach(said, MOVES) is True


def test_the_live_answer_is_refused_as_a_line() -> None:
    """A reason attached to a move does not make the move a line."""

    said = (
        'I am choosing **right** because the "Did you finish applying?" prompt '
        "is actively blocking the main feed and needs to be dismissed before I "
        "can see what's behind it"
    )
    assert read_strategy(said, MOVES, situation="a 16 and an 8 in the top row") is None


def test_a_real_line_still_reads_with_what_would_end_it() -> None:
    held = read_strategy(
        "My plan is to keep the largest tile in the bottom-left corner and feed "
        "the bottom row, while the 64 is still there",
        MOVES,
        situation="a 64 in the bottom-left",
    )
    assert held is not None
    assert "bottom-left corner" in held.approach
    assert held.holds_while.describes


def test_the_test_runs_whatever_else_the_answer_mentions() -> None:
    """It used to run only when nothing watchable had been named."""

    import ast
    from pathlib import Path

    source = Path("core/agency/standing_strategy.py").read_text(encoding="utf-8")
    lines = source.splitlines()
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if not isinstance(node, ast.FunctionDef) or node.name != "read_strategy":
            continue
        # Raw source, not unparsed: unparse normalises the boolean grouping
        # this is looking for.
        body = "\n".join(lines[node.lineno - 1 : node.end_lineno])
        # It runs before the branch that used to be its only caller, and the
        # answer is refused there rather than dressed up as a line.
        first = body.index("_says_enough_to_be_an_approach")
        guarded = body.index("if not keep and not avoid")
        assert first < guarded, "the line-versus-move test is still conditional"
        assert "not a line to take" in body
        return
    raise AssertionError("read_strategy is gone")
