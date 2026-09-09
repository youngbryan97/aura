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


def test_the_memory_bid_is_a_recollection_not_the_turn_just_finished():
    """Bidding the last working-memory item bids the turn that has only this
    moment finished, which is fresh on every cycle — so it entered at full
    priority every time and the other domains' bids never decided anything."""
    state = AuraState.default()
    state.cognition.working_memory.append(
        {"role": "user", "content": "just said", "timestamp": time.time()}
    )
    assert not any(bid.source == "memory" for bid in build_candidates(state))

    state.cognition.long_term_memory = ["something recalled"]
    memory = next(bid for bid in build_candidates(state) if bid.source == "memory")
    assert memory.content == "something recalled"
    # Neutral, because the retrieval score is not carried into the context and
    # there is no reading here to price it by. At the maximum it won almost
    # every competition.
    assert memory.priority == pytest.approx(0.5)


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


def test_a_surprising_world_bids_at_its_own_prediction_error():
    """Its only route into the workspace was a heartbeat branch gated at a free
    energy above 0.35, which is a different and much rarer event."""
    from types import SimpleNamespace as _NS

    from core.container import ServiceContainer

    class _Model:
        def surprise(self):
            return 0.63

    ServiceContainer.register_instance("unified_world_model", _Model())
    bid = next(b for b in build_candidates(AuraState.default()) if b.source == "world_model")
    assert bid.priority == pytest.approx(0.63)
    del _NS


def test_a_world_that_behaved_as_predicted_does_not_bid():
    from core.container import ServiceContainer

    class _Calm:
        def surprise(self):
            return 0.0

    ServiceContainer.register_instance("unified_world_model", _Calm())
    assert not any(b.source == "world_model" for b in build_candidates(AuraState.default()))


def test_the_exchange_bids_at_the_conversation_s_energy():
    """It belongs in the competition — at a reading, not at a flat maximum."""
    state = AuraState.default()
    state.cognition.working_memory.append({"role": "user", "content": "just said"})
    state.cognition.conversation_energy = 0.31
    bid = next(b for b in build_candidates(state) if b.source == "exchange")
    assert bid.priority == pytest.approx(0.31)

    state.cognition.conversation_energy = 0.0
    assert not any(b.source == "exchange" for b in build_candidates(state))


def test_a_real_goal_can_bid_at_all() -> None:
    """The bid read `urgency`; the goal engine writes `priority`.

    No producer in the tree has ever written `urgency`, so every real goal bid
    zero and deliberation never once reached the workspace.
    """
    from core.consciousness.workspace_feed import build_candidates
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [
        {"id": "g1", "goal": "finish the proof", "priority": 0.8, "status": "pending"}
    ]
    sources = {getattr(bid, "source", "") for bid in build_candidates(state)}
    assert "deliberation" in sources


def test_a_goal_with_a_named_priority_is_understood() -> None:
    from core.consciousness.workspace_feed import build_candidates
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [{"goal": "urgent thing", "priority": "critical"}]
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "deliberation"]
    assert bids and bids[0].priority == 1.0


def test_a_recollection_bids_at_how_well_it_matched() -> None:
    """Retrieval ranks by a score and used to throw it away."""
    from core.consciousness.workspace_feed import build_candidates
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.long_term_memory = ["a weak match", "the answer"]
    state.cognition.memory_scores = [0.2, 0.9]
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "memory"]
    assert bids and bids[0].priority == 0.9

    state.cognition.memory_scores = []
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "memory"]
    assert bids and bids[0].priority == 0.5


def test_retrieval_keeps_the_score_it_ranked_by() -> None:
    from pathlib import Path

    source = (Path(__file__).resolve().parents[1] / "core" / "phases" / "memory_retrieval.py").read_text()
    assert "cognition.memory_scores = scores" in source, "the ranking is discarded again"
