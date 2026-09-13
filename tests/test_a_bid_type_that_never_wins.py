"""A source that never wins is a channel into attention that cannot fire.

The winner alone cannot show it: a source that lost every competition looks
exactly like one that never spoke, and both look like one the gate refused. So
the workspace counts what was offered as well as what won, before any gate, and
a run reports which sources bid and never got in.

Ten sources feed this competition. One of them permanently silent is a domain
whose route to attention is closed, and the battery would read that as the
domain having nothing to say.
"""

from __future__ import annotations

import asyncio

from core.consciousness.global_workspace import (
    CognitiveCandidate,
    ContentType,
    GlobalWorkspace,
)


def _run(priorities: dict[str, float], ticks: int = 5) -> dict:
    async def go() -> dict:
        workspace = GlobalWorkspace()
        for tick in range(ticks):
            for source, priority in priorities.items():
                await workspace.submit(
                    CognitiveCandidate(
                        content=f"{source} {tick}",
                        source=source,
                        priority=priority,
                        content_type=ContentType.PERCEPTUAL,
                    )
                )
            await workspace.run_competition()
        return workspace.get_snapshot()

    return asyncio.run(go())


def test_the_workspace_says_who_bid_and_who_won() -> None:
    snapshot = _run({"perception": 0.7, "memory": 0.4, "ontogeny": 0.2})
    assert snapshot["bids_by_source"]["ontogeny"] == 5
    assert "perception" in snapshot["wins_by_source"]


def test_a_source_that_never_wins_is_named() -> None:
    snapshot = _run({"perception": 0.9, "ontogeny": 0.05})
    assert "ontogeny" in snapshot["sources_that_never_won"]


def test_a_source_that_wins_is_not_named() -> None:
    snapshot = _run({"perception": 0.9, "memory": 0.88})
    never = set(snapshot["sources_that_never_won"])
    assert "perception" not in never


def test_bids_are_counted_before_any_gate() -> None:
    """A source refused every time is as dead a channel as one that never wins,
    and the two are told apart by whether it ever got in."""
    import inspect

    source = inspect.getsource(GlobalWorkspace.submit)
    assert "Counted before any gate" in source
    offered = source.index("self._bids_by_source[offered]")
    inhibited = source.index("Check internal inhibition")
    assert offered < inhibited
