"""General bug patterns are found where they are, and not in code that is right.

The broken Pong (tests/fixtures/pong_repair/pong.html) has five flaws put in on
purpose; tools/grade_pong_repair.py names them. Each is a general pattern, so
each must be found, and the same program mended by hand must raise nothing.
"""
from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("tree_sitter_javascript")

from core.self_modification.code_that_looks_wrong import applied, what_looks_wrong  # noqa: E402

pytestmark = pytest.mark.unit

BROKEN = Path(__file__).parent / "fixtures" / "pong_repair" / "pong.html"


def _mended(source: str) -> str:
    return (
        source.replace('if (keys["ArrowUp"]) player.y += PLAYER_SPEED * dt;', 'if (keys["ArrowUp"]) player.y -= PLAYER_SPEED * dt;')
        .replace('if (keys["ArrowDown"]) player.y -= PLAYER_SPEED * dt;', 'if (keys["ArrowDown"]) player.y += PLAYER_SPEED * dt;')
        .replace("target - computer.y)) * 0;", "target - computer.y));")
        .replace("ball.y < paddle.y + PADDLE_W &&", "ball.y < paddle.y + PADDLE_H &&")
        .replace("  if (ball.y + BALL > H) {", "  if (ball.y < 0) {\n    ball.y = 0;\n    ball.vy = Math.abs(ball.vy);\n  }\n  if (ball.y + BALL > H) {")
        .replace("  if (ball.x + BALL < 0) {\n    playerScore += 1;", "  if (ball.x + BALL < 0) {\n    computerScore += 1;")
    )


def test_every_flaw_put_in_is_found_by_a_general_pattern():
    found = what_looks_wrong(BROKEN.read_text(), ".html")
    by_function = {(s.pattern, s.function) for s in found}
    assert ("direction against its name", "movePlayer") in by_function
    assert ("always zero", "moveComputer") in by_function
    assert ("crossed dimension", "hitsPaddle") in by_function
    assert ("one-sided boundary", "moveBall") in by_function
    assert ("same in both branches", "moveBall") in by_function


def test_the_mended_program_raises_nothing():
    assert what_looks_wrong(_mended(BROKEN.read_text()), ".html") == []


def test_an_edit_lands_where_it_says():
    source = BROKEN.read_text()
    zero = next(s for s in what_looks_wrong(source, ".html") if s.pattern == "always zero")
    assert "* 0;" not in applied(source, zero.edits).split("function moveComputer")[1].split("}")[0]
