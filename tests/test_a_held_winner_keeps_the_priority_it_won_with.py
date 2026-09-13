"""The workspace reported its last winner's priority as of the moment it was read.

A bid's priority falls with its age, so the same winner, re-scored against a
later clock, read lower each frame. Under a lesion the workspace was restored
after every phase and `G.winner_priority` still moved by 0.267 in one run and
0.247 in the next, which refused the lesion as a cut that had not severed. The
snapshot now reports the priority the winner won with, at the competition's
own instant.
"""

from __future__ import annotations

import pytest

import core.consciousness.global_workspace as workspace_module
from core.consciousness.global_workspace import CognitiveCandidate, ContentType, GlobalWorkspace


@pytest.mark.asyncio
async def test_the_reported_priority_does_not_age_after_the_win(monkeypatch) -> None:
    workspace = GlobalWorkspace()
    await workspace.submit(
        CognitiveCandidate(
            content="a bid that wins alone",
            source="winner_under_test",
            priority=0.7,
            content_type=ContentType.INTENTIONAL,
        )
    )
    winner = await workspace.run_competition()
    assert winner is not None
    reported = workspace.get_snapshot()["last_priority"]

    now = workspace_module.time.time()
    monkeypatch.setattr(workspace_module.time, "time", lambda: now + 2.0)

    # The winner itself has aged, so a snapshot that re-scored it would move.
    assert round(winner.effective_priority, 3) != reported
    assert workspace.get_snapshot()["last_priority"] == reported


def test_the_winning_priority_is_state_the_workspace_carries() -> None:
    """Carried on the workspace, so a clamp that restores the organ holds it."""
    assert "last_winner_priority" in vars(GlobalWorkspace())
