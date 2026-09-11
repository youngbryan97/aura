"""Five open goals, and which five they are.

Deliberation reaches attention through one number: what an intention states it
is asking to be thought about. Three things stood between a pressing intention
and the workspace, and all three were rankings that ranked nothing.

The goal list was capped at five in the order things had been written to it, so
a goal formed this turn because a need had just become urgent was dropped for
five older ones that were not pressing at all. The workspace then bid the last
two of whatever survived, again by position. And an intention reproposed under
the same text had its urgency raised by a max, so a need that had once been at
its worst went on asking at that level after it was met.
"""

from __future__ import annotations

import pytest

from core.consciousness.executive_closure import _by_pressure, _stated_urgency
from core.consciousness.workspace_feed import _goal_priority


def _goal(text: str, urgency: float | str | None) -> dict[str, object]:
    record: dict[str, object] = {"description": text, "goal": text, "status": "pending"}
    if urgency is not None:
        record["urgency"] = urgency
    return record


def test_the_cap_keeps_what_presses_not_what_is_oldest() -> None:
    stale = [_goal(f"old {index}", 0.1) for index in range(5)]
    pressing = _goal("the need that just became urgent", 0.9)
    kept = _by_pressure([*stale, pressing])[:5]
    assert pressing in kept, "a goal that just became urgent was dropped for five quiet ones"
    assert kept[0] is pressing


def test_a_tie_keeps_the_order_it_was_written_in() -> None:
    """Ranking must not reorder what it cannot distinguish.

    Every goal states 0.5 on an ordinary turn, and a sort that shuffled them
    would make which intention reaches the workspace a property of the sort
    rather than of the organism.
    """
    goals = [_goal(f"goal {index}", 0.5) for index in range(6)]
    assert _by_pressure(goals) == goals


def test_the_closure_and_the_workspace_price_a_goal_the_same_way() -> None:
    """One rule, read in both places, or the ranking here selects goals the
    competition then prices differently."""
    for urgency in (0.0, 0.25, 0.5, 0.9, 1.0, "critical", "low", None, "nonsense"):
        goal = _goal("something", urgency)
        assert _stated_urgency(goal) == pytest.approx(_goal_priority(goal))


def test_a_goal_that_states_nothing_makes_no_claim() -> None:
    assert _stated_urgency({"description": "x"}) == pytest.approx(0.5)
    assert _stated_urgency("a bare string") == pytest.approx(0.5)


@pytest.mark.asyncio
async def test_a_reproposed_intention_asks_for_what_it_asks_for_now(service_container) -> None:
    """The urgency ratchet, in the case that showed it.

    The motivation phase reproposes the same goal for the same drive on every
    turn it runs. With a max, the first time a need was at its most depleted
    set the level it pressed at for the life of the entry.
    """
    from types import SimpleNamespace

    from core.consciousness.executive_authority import ExecutiveAuthority
    from core.state.aura_state import AuraState

    state = AuraState()
    repo = SimpleNamespace()
    repo.current = state

    async def _get_current():
        return repo.current

    async def _commit(new_state, _cause):
        repo.current = new_state
        return new_state

    repo.get_current = _get_current
    repo.commit = _commit
    service_container.register_instance("state_repository", repo)

    authority = ExecutiveAuthority()
    await authority.queue_initiative("Attend to the growth drive", source="motivation_update", urgency=0.9)
    await authority.queue_initiative("Attend to the growth drive", source="motivation_update", urgency=0.2)

    pending = repo.current.cognition.pending_initiatives
    assert len(pending) == 1
    assert pending[0]["urgency"] == pytest.approx(0.2, abs=1e-6), (
        "a met need went on pressing at the level it reached when it was worst"
    )
