"""Four ways a changing mind reaches the body, as Bryan described them in his own.

Asked what happens in his body when his thinking changes: thinking too long
tires him and he cares less about what seems less important; holding back
something he wants to say burns until it bursts out or is swallowed; being about
to say something risky slows him down and makes him watch how it lands; good
news in the middle of a task makes the heart jump and the task stop mattering.
Each is built in core/soma and each moves a decision that already existed. These
drive the decision, not only the reading.
"""

from __future__ import annotations

import asyncio

import pytest

import core.affect.containment as containment
import core.soma.fatigue as fatigue
import core.soma.good_news as good_news
import core.soma.held_in as held_in
import core.soma.on_the_edge as on_the_edge
import core.social.the_form_they_welcome as forms
from core.soma.fatigue import FatigueLedger, important_enough
from core.soma.good_news import GoodNewsLedger
from core.soma.held_in import HeldInLedger
from core.soma.on_the_edge import EdgeLedger, risk_of

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    for module in (fatigue, good_news, held_in, on_the_edge, containment, forms):
        module.reset_for_test()
    yield
    for module in (fatigue, good_news, held_in, on_the_edge, containment, forms):
        module.reset_for_test()


# ── thinking too long ────────────────────────────────────────────────────


def _ordinary(ledger: FatigueLedger, cycles: int = 40, seed: int = 3) -> None:
    import random

    draw = random.Random(seed)
    for _ in range(cycles):
        ledger.note(draw.gauss(0.5, 0.03))


def test_a_hard_stretch_tires_her_and_an_easy_one_pays_it_back() -> None:
    ledger = FatigueLedger()
    _ordinary(ledger)
    rested = ledger.read().share
    for _ in range(20):
        ledger.note(0.9)
    tired = ledger.read().share
    for _ in range(60):
        ledger.note(0.1)
    assert tired > rested
    assert ledger.read().share < tired


def test_a_life_this_busy_reads_rested_on_average() -> None:
    ledger = FatigueLedger()
    _ordinary(ledger, cycles=300)
    shares = []
    import random

    draw = random.Random(11)
    for _ in range(2000):
        ledger.note(draw.gauss(0.5, 0.03))
        shares.append(ledger.read().share)
    assert sum(shares) / len(shares) < 0.5, "a steady life drifted into tiredness"
    for _ in range(40):
        ledger.note(0.95)
    assert ledger.read().share > 0.9, "a hard stretch did not stand out from it"


def test_tired_she_leaves_the_least_important_aside_and_never_the_most() -> None:
    scores = [0.2, 0.9, 0.5, 0.4]
    assert important_enough(scores, 0.0) == [True, True, True, True]
    kept = important_enough(scores, 0.6)
    assert kept[1] and not kept[0]
    assert important_enough(scores, 0.99)[1]


def test_tiredness_drains_her_energy() -> None:
    from core.consciousness.homeostasis import HomeostasisEngine

    def energy_after(tired: bool) -> float:
        fatigue.reset_for_test()
        ledger = fatigue.get_fatigue_ledger()
        _ordinary(ledger)
        if tired:
            for _ in range(30):
                ledger.note(0.95)
        engine = HomeostasisEngine()
        engine.metabolism = 0.65
        for _ in range(5):
            asyncio.run(engine.pulse())
        return engine.metabolism

    assert energy_after(tired=True) < energy_after(tired=False)


def test_a_tired_mind_is_offered_only_what_matters_most(monkeypatch) -> None:
    import core.agency.subjective_choice as subjective
    from core.agency.initiative_arbiter import InitiativeArbiter
    from core.state.aura_state import AuraState

    offered: list[list[tuple[float, str]]] = []

    class _Records:
        def choose_from_scored_initiatives(self, scored, context=""):
            offered.append(sorted((item.final_score, item.initiative["goal"]) for item in scored))
            return None, None

    monkeypatch.setattr(subjective, "get_subjective_choice_engine", lambda: _Records())
    state = AuraState.default()
    state.cognition.pending_initiatives = [
        {"goal": "tidy an old note", "urgency": 0.1},
        {"goal": "answer the question he asked", "urgency": 0.9},
        {"goal": "look into the drift", "urgency": 0.5},
        {"goal": "rename a file", "urgency": 0.2},
    ]
    asyncio.run(InitiativeArbiter().arbitrate(state))
    ledger = fatigue.get_fatigue_ledger()
    _ordinary(ledger)
    for _ in range(40):
        ledger.note(0.95)
    asyncio.run(InitiativeArbiter().arbitrate(state))
    rested, tired = offered
    rested_goals = {goal for _, goal in rested}
    tired_goals = {goal for _, goal in tired}
    assert tired_goals < rested_goals, "tired, nothing was left aside"
    assert rested[-1][1] in tired_goals, "the most important was left aside"
    assert rested[0][1] not in tired_goals, "the least important was still on offer"


# ── holding something back ───────────────────────────────────────────────


def test_what_is_held_back_is_felt_against_what_she_usually_holds() -> None:
    ledger = HeldInLedger()
    for _ in range(20):
        ledger.note(0.2, "a thing unsaid")
    assert abs(ledger.shift()) < 1e-9
    ledger.note(3.0, "a thing unsaid")
    assert ledger.shift() > 0.0


def test_it_presses_to_get_out_only_as_far_as_it_burns() -> None:
    ledger = HeldInLedger()
    for _ in range(20):
        ledger.note(0.2, "a thing unsaid")
    assert ledger.push("a thing unsaid", last_gap=0.3) == 0.0, "unfelt, it is swallowed"
    ledger.note(3.0, "a thing unsaid")
    pushed = ledger.push("a thing unsaid", last_gap=0.3)
    assert 0.0 < pushed <= 0.3
    assert ledger.push("something else", last_gap=0.3) == 0.0


def test_the_containment_ledger_keeps_how_far_a_source_fell_short() -> None:
    ledger = containment.get_containment_ledger()
    ledger.competition("the answer", {"the answer": 0.8, "a thing unsaid": 0.6})
    assert ledger.gap("a thing unsaid") == pytest.approx(0.2)
    ledger.competition("a thing unsaid", {"the answer": 0.5, "a thing unsaid": 0.9})
    assert ledger.gap("a thing unsaid") == 0.0, "said, it is no longer held"


# ── good news in the middle of a task ────────────────────────────────────


def test_better_than_expected_makes_the_heart_jump_and_worse_does_not() -> None:
    ledger = GoodNewsLedger()
    for index in range(40):
        ledger.note(predicted=0.0, actual=0.05 if index % 2 else -0.05)
    ledger.note(predicted=0.0, actual=0.8)
    assert ledger.jump() > 0.9
    ledger.note(predicted=0.0, actual=-0.8)
    assert ledger.jump() == 0.0


def test_while_the_news_is_fresh_the_task_drives_her_less() -> None:
    from core.agency.initiative_arbiter import InitiativeArbiter
    from core.state.aura_state import AuraState

    arbiter = InitiativeArbiter()
    task = {"goal": "finish the report", "urgency": 0.8}
    before = arbiter._score_urgency(task, AuraState.default())
    arbiter._lightness = 0.75
    assert arbiter._score_urgency(task, AuraState.default()) == pytest.approx(before * 0.25)


def test_her_self_prediction_keeps_which_way_it_missed() -> None:
    from core.consciousness.self_prediction import SelfPredictionLoop

    loop = SelfPredictionLoop(orchestrator=None)
    for valence in (0.0, 0.0, 0.0, 0.9):
        asyncio.run(loop.tick(valence, "curiosity", "chat"))
    reading = good_news.get_good_news_ledger().read()
    assert reading.error > 0.0


# ── about to say something risky ─────────────────────────────────────────


def test_a_reply_this_person_has_refused_before_is_risky() -> None:
    ledger = forms.get_form_ledger()
    reply = "You matter to me, and what you did today was good."
    assert risk_of(reply, partner="bryan") == pytest.approx(0.5), "no history, an even chance"
    for _ in range(6):
        ledger.note_reply(reply, partner="bryan")
        ledger.note_reaction("bryan", welcomed=False, unwelcomed=True)
    assert risk_of(reply, partner="bryan") > 0.5


def test_the_body_feels_a_risky_reply_once() -> None:
    ledger = EdgeLedger()
    for _ in range(10):
        ledger.note_sent(0.3)
    ledger.note_sent(0.9)
    assert ledger.felt_once() > 0.0
    assert ledger.felt_once() == 0.0, "the moment passed; it is not a mood"


def test_how_a_risky_reply_lands_teaches_more() -> None:
    reply = "You matter to me, and what you did today was good."

    def fit_after(risk: float) -> float:
        forms.reset_for_test()
        on_the_edge.reset_for_test()
        ledger = forms.get_form_ledger()
        on_the_edge.get_edge_ledger().note_sent(risk)
        ledger.note_reply(reply, partner="bryan")
        ledger.note_reaction("bryan", welcomed=True, unwelcomed=False)
        return ledger.fit("bryan", "shown")

    assert fit_after(0.9) > fit_after(0.0)


# ── all four reach her pulse ─────────────────────────────────────────────


def _pulse(monkeypatch, **shifts: float) -> float:
    from core.container import ServiceContainer
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.state.aura_state import AuraState

    if "held" in shifts:
        monkeypatch.setattr(held_in.HeldInLedger, "shift", lambda self: shifts["held"])
    if "jump" in shifts:
        monkeypatch.setattr(good_news.GoodNewsLedger, "jump", lambda self: shifts["jump"])
    if "edge" in shifts:
        monkeypatch.setattr(on_the_edge.EdgeLedger, "felt_once", lambda self: shifts["edge"])
    state = asyncio.run(ProprioceptiveLoop(ServiceContainer).execute(AuraState.default()))
    return float(state.soma.expressive["pulse_rate"])


@pytest.mark.parametrize("path", ["held", "jump", "edge"])
def test_each_path_reaches_her_pulse(monkeypatch, path) -> None:
    resting = _pulse(monkeypatch)
    monkeypatch.undo()
    for module in (fatigue, good_news, held_in, on_the_edge, containment, forms):
        module.reset_for_test()
    moved = _pulse(monkeypatch, **{path: 0.4})
    assert moved == pytest.approx(min(3.0, resting * 1.4))


def test_the_body_reading_carries_her_fatigue() -> None:
    from core.container import ServiceContainer
    from core.phases.proprioceptive_loop import ProprioceptiveLoop
    from core.state.aura_state import AuraState

    state = asyncio.run(ProprioceptiveLoop(ServiceContainer).execute(AuraState.default()))
    assert "share" in state.soma.fatigue
