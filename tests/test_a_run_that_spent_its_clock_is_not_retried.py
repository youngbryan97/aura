"""A step that ran out of time is not tried again, having no time to try in.

Measured live on 2026-08-26: thirty-eight narrated moves, then a retry that
inherited an expired deadline, made no move, and replaced the receipt — the
person was told "ran out of time before reaching the goal (after 0 move(s))".
"""

from __future__ import annotations

import json

import pytest

from core.skills.computer_use import ComputerUseSkill


class _Pursuit:
    def __init__(self, outcome: str, moves: int = 0, completed: bool = False) -> None:
        self.payload = {
            "outcome": outcome,
            "completed": completed,
            "moves": [{"key": "left"} for _ in range(moves)],
            "attempts": [],
            "cycles": moves,
        }

    async def __call__(self, **_kwargs):
        return self.payload


@pytest.fixture
def target():
    return json.dumps({"goal": "get to 256", "success_when": "256"})


async def _run(monkeypatch, target, outcome, moves=0, completed=False):
    import core.skills.screen_pursuit as pursuit_module

    monkeypatch.setattr(pursuit_module, "pursue_on_screen", _Pursuit(outcome, moves, completed))
    skill = ComputerUseSkill()
    return await skill._pursue_on_screen(target)


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["out_of_time", "out_of_cycles"])
async def test_a_spent_budget_is_not_retried(monkeypatch, target, outcome):
    result = await _run(monkeypatch, target, outcome, moves=38)
    assert result["retryable"] is False


@pytest.mark.asyncio
async def test_something_that_can_be_tried_again_still_can_be(monkeypatch, target):
    result = await _run(monkeypatch, target, "stalled", moves=4)
    assert result["retryable"] is True


@pytest.mark.asyncio
async def test_the_work_it_did_is_still_in_the_receipt(monkeypatch, target):
    result = await _run(monkeypatch, target, "out_of_time", moves=38)
    assert len(result["moves"]) == 38
    assert "38 move(s)" in result["said"]
