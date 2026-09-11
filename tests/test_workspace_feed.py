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


def test_the_feeling_carries_its_own_charge_not_the_moment_s_arousal():
    """`priority_at` adds three tenths of the affect weight to a bid's claim.

    The weight used to be the moment's global arousal — a quantity belonging to
    the whole moment, handed to one competitor as a private advantage that no
    other domain could earn. Affect won ninety-six competitions in a hundred
    because of it, so attention was `affect_*` whatever else was happening and
    the action chosen by what she was attending to was the same every turn.

    A bid's affect weight is a property of that bid: how far this feeling is
    above where it usually sits.
    """
    state = AuraState.default()
    state.affect.arousal = 0.9
    state.affect.emotions["fear"] = 0.8
    state.affect.mood_baselines["fear"] = 0.3
    feeling = next(bid for bid in build_candidates(state) if bid.source.startswith("affect_"))
    assert feeling.source == "affect_fear"
    assert feeling.priority == pytest.approx(0.8)
    assert feeling.affect_weight == pytest.approx(0.5)


def test_a_feeling_at_its_own_baseline_carries_no_charge():
    state = AuraState.default()
    state.affect.arousal = 0.9
    state.affect.emotions["fear"] = 0.4
    state.affect.mood_baselines["fear"] = 0.4
    feeling = next(bid for bid in build_candidates(state) if bid.source.startswith("affect_"))
    assert feeling.affect_weight == pytest.approx(0.0)


def test_no_other_bid_claims_an_affect_weight():
    """The bonus must be earned by the content, not by being the affect bid."""
    state = AuraState.default()
    state.affect.emotions["fear"] = 0.8
    state.world.recent_percepts.append(
        {"source": "chat", "content": "someone spoke", "salience": 0.62, "timestamp": time.time()}
    )
    for bid in build_candidates(state):
        if not bid.source.startswith("affect_"):
            assert bid.affect_weight == pytest.approx(0.0), bid.source


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


def test_a_body_under_load_bids_harder_than_a_quiet_one():
    """At the same floor as every other bid.

    This one alone was gated at a half, so the body could only speak once the
    machine was already at fifty percent. Deciding in advance which bids are
    worth hearing is the competition's job.
    """
    state = AuraState.default()
    state.soma.hardware["cpu_usage"] = 95.0
    loud = next(bid for bid in build_candidates(state) if bid.source == "interoception")
    state.soma.hardware["cpu_usage"] = 20.0
    state.soma.hardware["temperature"] = 30.0
    quiet = next(bid for bid in build_candidates(state) if bid.source == "interoception")
    assert loud.priority > quiet.priority
    state.soma.hardware["cpu_usage"] = 0.0
    state.soma.hardware["temperature"] = 0.0
    assert not any(bid.source == "interoception" for bid in build_candidates(state))


def test_her_own_exertion_can_bid_when_the_machine_is_idle():
    """The one somatic channel an experiment can leave free."""
    state = AuraState.default()
    state.soma.hardware["cpu_usage"] = 0.0
    state.soma.hardware["temperature"] = 0.0
    state.soma.exertion = 0.4
    bid = next(bid for bid in build_candidates(state) if bid.source == "interoception")
    assert bid.priority == pytest.approx(0.4)


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
    energy above 0.35, which is a different and much rarer event.

    Squashed rather than clipped. Prediction error is unbounded above, and a
    clip turns every surprise past one into the same maximum — so this bid sat
    at its ceiling on every turn and stopped being a reading of anything.
    """
    import math

    from core.container import ServiceContainer

    class _Model:
        def __init__(self, value: float) -> None:
            self.value = value

        def surprise(self):
            return self.value

    # With no running mean to compare against, any surprise is all of the
    # surprise there is.
    ServiceContainer.register_instance("unified_world_model", _Model(0.63))
    bid = next(b for b in build_candidates(AuraState.default()) if b.source == "world_model")
    assert bid.priority == pytest.approx(1.0)

    # Against a model whose ordinary error is three, two surprises past one are
    # still told apart — which a clip at one, or a squash flat past two and a
    # half, could not do.
    class _Settled(_Model):
        def status(self):
            return {"facets": {"learned": {"detail": {"mean_surprise": 3.0}}}}

    ServiceContainer.register_instance("unified_world_model", _Settled(1.5))
    milder = next(b for b in build_candidates(AuraState.default()) if b.source == "world_model")
    ServiceContainer.register_instance("unified_world_model", _Settled(6.0))
    worse = next(b for b in build_candidates(AuraState.default()) if b.source == "world_model")
    assert worse.priority > milder.priority
    assert milder.priority < 0.5 < worse.priority
    del math


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


def test_a_goal_with_a_named_urgency_is_understood() -> None:
    from core.consciousness.workspace_feed import build_candidates
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [{"goal": "urgent thing", "urgency": "critical"}]
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "deliberation"]
    assert bids and bids[0].priority == 1.0


def test_importance_is_not_a_claim_on_attention() -> None:
    """Priority is how important the work is; urgency is what it is asking for.

    Read as a claim, the goal engine's own projection arrived at a flat one and
    won every competition — deliberation beating perception, memory, affect and
    the body on every turn, not because anything was pressing but because a
    default had been read as a demand.
    """
    from core.consciousness.workspace_feed import build_candidates
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [
        {"goal": "important but not pressing", "priority": 1.0}
    ]
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "deliberation"]
    assert bids and bids[0].priority == pytest.approx(0.5)


def test_one_intention_bids_once_however_many_lists_it_is_in() -> None:
    from core.consciousness.workspace_feed import build_candidates
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [{"goal": "the same intention", "urgency": 0.7}]
    state.cognition.pending_initiatives = [{"goal": "the same intention", "urgency": 0.9}]
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "deliberation"]
    assert len(bids) == 1


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


def test_a_finished_goal_is_a_record_not_an_intention():
    """It bid at the priority of a completed thing and won everything.

    The action arm appends a goal marked done on every turn it acts. Bid at
    that priority it was a flat maximum, and a flat maximum in the competition
    crowds every other domain out — the same defect the memory and exchange
    bids had, and the reason displacing affect stopped changing what won.
    """
    state = AuraState.default()
    state.cognition.active_goals = [
        {"goal": "wrote the file", "status": "done", "urgency": 1.0},
        {"goal": "still working on this", "status": "pending", "urgency": 0.4},
    ]
    bids = [b for b in build_candidates(state) if getattr(b, "source", "") == "deliberation"]
    assert len(bids) == 1
    assert bids[0].priority == pytest.approx(0.4)
    assert "still working" in bids[0].content


def test_what_won_is_in_mind_at_the_strength_it_won_with():
    """The two lists are read side by side and only one of them was written.

    The workspace prices its memory bid from the score at the same index as the
    recollection, so anything that writes one has to write the other or the
    pairing comes apart without saying so.
    """
    from types import SimpleNamespace

    from core.consciousness.workspace_feed import _remember_broadcast
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.long_term_memory = ["something recalled earlier"]
    state.cognition.memory_scores = [0.4]
    winner = SimpleNamespace(source="perception", content="a window moved", effective_priority=0.8)
    _remember_broadcast(state, winner, ignited=True)

    assert len(state.cognition.memory_scores) == len(state.cognition.long_term_memory)
    assert state.cognition.memory_scores[-1] == pytest.approx(0.8)

    # And it does not bid itself back. What is in mind because it just won the
    # competition enters this list at the strength it won with, so bidding it
    # back made a loop: the winner became the strongest recollection, won
    # again, and the competition settled on whatever had won once. The
    # recollection that was there before still competes, at its own score.
    bid = next(b for b in build_candidates(state) if b.source == "memory")
    assert bid.priority == pytest.approx(0.4)
    assert "recalled earlier" in bid.content


def test_what_just_won_does_not_bid_itself_back():
    from types import SimpleNamespace

    from core.consciousness.workspace_feed import _remember_broadcast
    from core.state.aura_state import AuraState

    state = AuraState.default()
    winner = SimpleNamespace(source="perception", content="a window moved", effective_priority=0.95)
    _remember_broadcast(state, winner, ignited=True)
    assert not any(b.source == "memory" for b in build_candidates(state))
    # It is still in mind for the reply.
    assert any("a window moved" in str(line) for line in state.cognition.long_term_memory)


def test_the_selected_objective_states_what_it_is_asking_for():
    """Two records of one decision disagreed about how much it mattered.

    `ExecutiveClosure` writes the objective it has selected twice: as an
    initiative carrying the need pressure that selected it, and as a goal
    record two lines above carrying a flat priority of one and no urgency at
    all. The workspace prices deliberation's bid on urgency, so the thing she
    had just chosen to work on entered attention at the neutral default on
    every turn, and nothing about what she was trying to do could change what
    she attended to.
    """
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [
        {"description": "finish the migration", "priority": 1.0, "urgency": 0.82},
        {"description": "tidy the notes", "priority": 1.0, "urgency": 0.21},
    ]
    bids = {
        bid.content: bid.priority
        for bid in build_candidates(state)
        if bid.source == "deliberation"
    }
    assert bids["finish the migration"] == pytest.approx(0.82)
    assert bids["tidy the notes"] == pytest.approx(0.21)


def test_a_goal_that_states_no_urgency_still_enters_at_neutral():
    """Silence is not a claim, and it is not a refusal either."""
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.active_goals = [{"description": "something unpriced", "priority": 1.0}]
    bid = next(b for b in build_candidates(state) if b.source == "deliberation")
    assert bid.priority == pytest.approx(0.5)


def test_a_surprise_about_herself_competes_the_way_one_about_the_world_does(monkeypatch):
    """The two halves of the same bid.

    The world model's prediction error enters the competition. The self
    model's did not, though it computes one every tick and names the channel
    it missed on — so the only thing self-state could say to attention was how
    stable identity is, a number that barely moves. Everything else self-state
    contributed to what wins attention arrived through affect, which is what
    affect was already saying.

    That is measurable as redundancy: the preregistered synergy triple asks
    whether affect and self-state carry something about attention that neither
    carries alone, and two sources saying the same thing carry nothing jointly.
    """
    from types import SimpleNamespace

    import core.consciousness.workspace_feed as feed
    from core.state.aura_state import AuraState

    state = AuraState.default()

    def at(error: float) -> float:
        snapshot = {
            "smoothed_error": error,
            "valence_error_ema": 0.2,
            "drive_error_ema": 0.2,
            "focus_error_ema": 0.2,
            "most_unpredictable": "focus",
        }
        predictor = SimpleNamespace(get_snapshot=lambda: snapshot)
        monkeypatch.setattr(
            feed,
            "build_candidates",
            feed.build_candidates,
        )
        import core.runtime.service_registry as registry

        monkeypatch.setattr(
            registry,
            "get_runtime_service",
            lambda name, default=None: predictor if name == "self_prediction" else default,
        )
        bids = [
            bid
            for bid in feed.build_candidates(state)
            if getattr(bid, "source", "") == "self"
            and "predicted" in str(getattr(bid, "content", ""))
        ]
        return max((float(bid.priority) for bid in bids), default=0.0)

    predictable, surprising = at(0.05), at(0.9)
    assert surprising > predictable, (
        "a moment she failed to predict about herself bids no higher than one she did"
    )
    assert predictable >= 0.0
