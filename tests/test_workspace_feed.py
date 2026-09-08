"""A workspace with no candidates is not a workspace.

The competition ran on every tick of the cognitive cycle with an empty
candidate list, so the winner was None and nothing was ever broadcast. These
tests hold the feed to the two properties that make it a competition rather
than a rota: a domain with nothing to say does not bid, and what a domain bids
is priced by its own state rather than by a number chosen here.
"""

from __future__ import annotations

import asyncio
import time

import pytest

from core.consciousness.workspace_feed import build_candidates, feed_workspace
from core.state.aura_state import AuraState


def test_a_default_state_bids_only_what_it_has():
    """No percepts, no memory, no goals, nothing wrong: almost nothing to say."""
    bids = build_candidates(AuraState.default())
    assert {bid.source for bid in bids} <= {"affect_anticipation", "affect_happiness"}


def test_a_salient_percept_bids_with_its_own_salience():
    state = AuraState.default()
    state.world.recent_percepts.append(
        {"source": "chat", "content": "someone spoke", "salience": 0.62, "timestamp": time.time()}
    )
    percept = next(bid for bid in build_candidates(state) if bid.source == "perception")
    assert percept.priority == pytest.approx(0.62)


def test_a_faint_percept_does_not_bid_at_all():
    state = AuraState.default()
    state.world.recent_percepts.append(
        {"source": "chat", "content": "background", "salience": 0.01, "timestamp": time.time()}
    )
    assert not any(bid.source == "perception" for bid in build_candidates(state))


def test_the_feeling_carries_arousal_as_its_affect_weight():
    state = AuraState.default()
    state.affect.arousal = 0.9
    state.affect.emotions["fear"] = 0.8
    feeling = next(bid for bid in build_candidates(state) if bid.source.startswith("affect_"))
    assert feeling.source == "affect_fear"
    assert feeling.affect_weight == pytest.approx(0.9)


def test_a_stale_memory_is_dated_when_it_was_formed():
    """Otherwise recall wins every tick on a flat priority of one."""
    state = AuraState.default()
    old = time.time() - 3600
    state.cognition.working_memory.append({"role": "user", "content": "long ago", "timestamp": old})
    memory = next(bid for bid in build_candidates(state) if bid.source == "memory")
    assert memory.submitted_at == pytest.approx(old)
    assert memory.effective_priority < memory.priority


def test_incoherence_bids_in_proportion_to_how_bad_it_is():
    state = AuraState.default()
    state.cognition.coherence_score = 0.4
    meta = next(bid for bid in build_candidates(state) if bid.source == "metacognition")
    assert meta.priority == pytest.approx(0.6)


def test_a_body_under_load_bids_and_a_quiet_one_does_not():
    state = AuraState.default()
    state.soma.hardware["cpu_usage"] = 95.0
    assert any(bid.source == "interoception" for bid in build_candidates(state))
    state.soma.hardware["cpu_usage"] = 5.0
    state.soma.hardware["temperature"] = 30.0
    assert not any(bid.source == "interoception" for bid in build_candidates(state))


def test_feeding_a_real_workspace_produces_a_winner_and_a_broadcast():
    from core.consciousness.global_workspace import GlobalWorkspace

    workspace = GlobalWorkspace()
    seen: list[str] = []

    async def watcher(event):
        seen.append(event.winners[0].source)

    workspace.register_processor(watcher)
    state = AuraState.default()
    state.affect.emotions["fear"] = 0.9
    state.affect.arousal = 0.9

    winner = asyncio.run(feed_workspace(state, workspace))
    assert winner is not None
    assert seen == [winner.source]


def test_no_workspace_is_not_an_error():
    assert asyncio.run(feed_workspace(AuraState.default(), None)) is None
