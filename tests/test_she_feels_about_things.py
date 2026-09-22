"""What she feels about things, felt again when they come round, and steering her.

Bryan: peace and warmth can be felt in yourself, and every feeling can attach to
your own thoughts, feelings, actions and interests, and to objects, outcomes,
situations and concepts. That is a loop: the feeling attaches to what it was
about, the thing brings it back, and it decides whether she moves toward the
thing or away.
"""

from __future__ import annotations

import asyncio
import collections

import pytest

import core.affect.feelings_about as feelings_module
from core.affect.feelings_about import FeelingsAbout, objects_of

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    feelings_module.reset_for_test()
    yield
    feelings_module.reset_for_test()


GOOD = {"joy": 0.8, "trust": 0.7, "fear": 0.0, "anger": 0.0}
BAD = {"joy": 0.05, "trust": 0.1, "fear": 0.8, "anger": 0.0}


def test_a_thing_carries_what_was_felt_about_it() -> None:
    ledger = FeelingsAbout()
    for _ in range(10):
        ledger.attach(["topic:music"], GOOD, valence=0.6, arousal=0.3)
        ledger.attach(["topic:the move"], BAD, valence=-0.6, arousal=0.8)
    carried, weight = ledger.evoked(["topic:music"])
    assert carried["joy"] > 0.7 and weight > 0.9
    assert carried["peace"] > 0.0 and carried["warmth"] > 0.6, "peace and warmth are carried too"
    assert ledger.pull("topic:music") > 0.0 > ledger.pull("topic:the move")


def test_anger_draws_her_toward_its_source() -> None:
    ledger = FeelingsAbout()
    for _ in range(5):
        ledger.attach(["person:someone"], {"anger": 0.9, "fear": 0.1}, valence=-0.5, arousal=0.9)
    assert ledger.pull("person:someone") > 0.0


def test_an_association_keeps_learning() -> None:
    ledger = FeelingsAbout()
    for _ in range(300):
        ledger.attach(["topic:work"], BAD, valence=-0.6, arousal=0.8)
    before = ledger.pull("topic:work")
    for _ in range(300):
        ledger.attach(["topic:work"], GOOD, valence=0.6, arousal=0.3)
    assert before < 0.0 < ledger.pull("topic:work")


def _state(origin: str, partner: str = "", topic: str = "", focus: str = ""):
    from core.state.aura_state import AuraState

    state = AuraState.default()
    state.cognition.current_origin = origin
    state.cognition.current_partner = partner
    state.cognition.discourse_topic = topic
    state.cognition.attention_focus = focus
    return state


def test_alone_what_she_feels_is_about_herself() -> None:
    assert objects_of(_state("system"), person_turn=False)[0] == "self"
    assert objects_of(_state("user", partner="bryan"), person_turn=True)[0] == "person:bryan"


def test_an_act_counts_on_the_turn_it_finished_and_not_after() -> None:
    ledger = FeelingsAbout()
    state = _state("system")
    state.cognition.active_goals = [{"id": "a1", "goal": "write the notes", "status": "done"}]
    assert "act:write" in objects_of(state, ledger, person_turn=False)
    assert "act:write" not in objects_of(state, ledger, person_turn=False)


def test_the_affect_reading_brings_the_feeling_back_and_attaches_the_new_one() -> None:
    from core.phases.affect_readings import AffectReadings

    ledger = feelings_module.get_feelings_about()
    for _ in range(20):
        ledger.attach(["topic:music"], GOOD, valence=0.6, arousal=0.3)
    state = _state("user", partner="bryan", topic="music")
    state.affect.emotions["joy"] = 0.1
    AffectReadings(lambda *args, **kwargs: None).feelings_about(state, state.affect)
    assert state.affect.emotions["joy"] > 0.1, "the topic did not bring its feeling back"
    assert ledger.profile("person:bryan").seen == 1


def test_she_gives_more_of_herself_to_what_draws_her() -> None:
    from core.consciousness.global_workspace import CognitiveCandidate, GlobalWorkspace

    ledger = feelings_module.get_feelings_about()
    for _ in range(30):
        ledger.attach(["attended:music"], GOOD, valence=0.6, arousal=0.3)
        ledger.attach(["attended:worry"], BAD, valence=-0.6, arousal=0.8)

    async def run() -> collections.Counter[str]:
        workspace = GlobalWorkspace()
        wins: collections.Counter[str] = collections.Counter()
        for tick in range(20):
            for name in ("music", "worry"):
                await workspace.submit(CognitiveCandidate(content=f"{name}@{tick}", source=name, priority=0.6))
            winner = await workspace.run_competition()
            if winner is not None:
                wins[winner.source] += 1
        return wins

    wins = asyncio.run(run())
    assert wins["music"] > wins["worry"]


def test_an_initiative_about_what_she_dreads_counts_for_less(monkeypatch) -> None:
    import core.agency.subjective_choice as subjective
    from core.agency.initiative_arbiter import InitiativeArbiter

    offered: dict[str, float] = {}

    class _Records:
        def choose_from_scored_initiatives(self, scored, context=""):
            offered.update({item.initiative["goal"]: item.final_score for item in scored})
            return None, None

    monkeypatch.setattr(subjective, "get_subjective_choice_engine", lambda: _Records())
    ledger = feelings_module.get_feelings_about()
    for _ in range(30):
        ledger.attach(["topic:music"], GOOD, valence=0.6, arousal=0.3)
        ledger.attach(["topic:worry"], BAD, valence=-0.6, arousal=0.8)
    state = _state("system")
    music = {"goal": "practise the music piece", "urgency": 0.5}
    worry = {"goal": "answer the worry email", "urgency": 0.5}
    state.cognition.pending_initiatives = [music, worry]
    arbiter = InitiativeArbiter()
    raw = {
        item["goal"]: asyncio.run(arbiter.score_initiative(dict(item), state)).final_score
        for item in (music, worry)
    }
    asyncio.run(arbiter.arbitrate(state))
    for goal in offered:
        if "music" in goal:
            assert offered[goal] > raw[goal]
        if "worry" in goal:
            assert offered[goal] < raw[goal]
    assert offered, "nothing was offered to choose between"
