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
from types import SimpleNamespace

import pytest

from core.consciousness.workspace_feed import build_candidates, feed_workspace
from core.state.aura_state import AuraState


def test_a_default_state_bids_only_what_it_has():
    """No percepts, no memory, no goals, nothing wrong: almost nothing to say.

    The lifetime reading is reset first. It is process-wide and any earlier
    test that advanced it leaves a novelty bid behind, which would make this
    assertion depend on the order the file was run in rather than on the state
    it was handed.
    """
    from core.ontogeny.lifetime import reset_for_test

    reset_for_test()
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


def test_an_ignited_broadcast_becomes_context_for_the_cycle():
    """What wins should reach the thing that speaks, which is the claim GWT makes."""
    from core.consciousness.global_workspace import GlobalWorkspace

    workspace = GlobalWorkspace()
    state = AuraState.default()
    state.affect.emotions["fear"] = 0.9
    state.affect.arousal = 0.9
    asyncio.run(feed_workspace(state, workspace))
    if workspace.ignited:
        assert any(str(line).startswith("[broadcast: ") for line in state.cognition.long_term_memory)


def test_a_broadcast_that_did_not_ignite_is_not_asserted_as_context():
    from core.consciousness.workspace_feed import _remember_broadcast

    state = AuraState.default()
    winner = SimpleNamespace(source="memory", content="quiet")
    _remember_broadcast(state, winner, ignited=False)
    assert state.cognition.long_term_memory == []


def test_the_context_line_does_not_accumulate():
    from core.consciousness.workspace_feed import CONTEXT_LIMIT, _remember_broadcast

    state = AuraState.default()
    for index in range(20):
        _remember_broadcast(
            state, SimpleNamespace(source=f"s{index}", content="x"), ignited=True
        )
    lines = [i for i in state.cognition.long_term_memory if str(i).startswith("[broadcast: ")]
    assert len(lines) == 1
    assert len(state.cognition.long_term_memory) <= CONTEXT_LIMIT


def test_the_winner_becomes_where_she_is_looking_whether_or_not_it_ignited():
    """`cognition.attention_focus` has two readers and had no writer at all."""
    from core.consciousness.workspace_feed import _remember_broadcast

    state = AuraState.default()
    winner = SimpleNamespace(source="perception", content="a window moved")
    _remember_broadcast(state, winner, ignited=False)
    assert state.cognition.attention_focus.startswith("perception: ")
    assert state.cognition.long_term_memory == []


def test_an_ordinary_moment_still_bids_on_its_novelty():
    """The reservoir puts an ordinary moment near 0.2, so a gate at 0.5
    excluded every real reading and admitted only the placeholder it returns
    before it has a distribution to compare against."""
    from types import SimpleNamespace as _NS

    import core.consciousness.workspace_feed as feed

    state = AuraState.default()
    import core.ontogeny.lifetime as lifetime

    original = lifetime.last_reading
    try:
        lifetime.last_reading = lambda: _NS(novelty=0.22, displacement=0.1)
        sources = {bid.source for bid in feed.build_candidates(state)}
        assert "ontogeny" in sources
    finally:
        lifetime.last_reading = original


def test_a_novelty_of_nothing_does_not_bid():
    from types import SimpleNamespace as _NS

    import core.consciousness.workspace_feed as feed
    import core.ontogeny.lifetime as lifetime

    original = lifetime.last_reading
    try:
        lifetime.last_reading = lambda: _NS(novelty=0.0, displacement=0.0)
        sources = {bid.source for bid in feed.build_candidates(AuraState.default())}
        assert "ontogeny" not in sources
    finally:
        lifetime.last_reading = original


def test_the_feed_stops_competing_once_something_else_is():
    """Submission is the cycle's job and arbitration is the heartbeat's. The
    first version competed here too, which emptied the candidate list before
    the heartbeat reached it — so the heartbeat's winner was None and the focus
    it hands the self-prediction loop was the string "none" on every beat."""
    from core.consciousness.global_workspace import GlobalWorkspace

    workspace = GlobalWorkspace()
    state = AuraState.default()
    state.affect.emotions["fear"] = 0.9

    assert asyncio.run(feed_workspace(state, workspace)) is not None
    before = workspace._tick

    async def heartbeat_then_feed():
        state.affect.emotions["fear"] = 0.8
        await feed_workspace(state, workspace)
        await workspace.run_competition()
        state.affect.emotions["fear"] = 0.7
        await feed_workspace(state, workspace)
        return workspace._tick

    after = asyncio.run(heartbeat_then_feed())
    # Two beats since: the one the feed ran on its first pass through the
    # helper, and the explicit one standing in for the heartbeat. The third
    # call must not have added a fourth.
    assert after - before <= 2
