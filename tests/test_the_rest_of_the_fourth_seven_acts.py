"""The nine processes from the fourth seven that were listed as unbuilt, each moving a gate.

Every test drives the gate the reading moves, not only the reading: a ledger
nothing consults is the defect these were written to remove.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from core.state.aura_state import AuraState

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh_ledgers():
    from core.agency import easier_loss, reference_class
    from core.motivation import for_its_own_sake
    from core.self import still_standing, valued_for
    from core.social import the_form_they_welcome, their_rise, what_passes_between

    modules = (
        easier_loss, reference_class, for_its_own_sake, still_standing, valued_for,
        the_form_they_welcome, their_rise, what_passes_between,
    )
    for module in modules:
        module.reset_for_test()
    yield
    for module in modules:
        module.reset_for_test()


# ── valued for a token ────────────────────────────────────────────────────


def test_regard_that_runs_against_her_own_rating_teaches_her_taste_nothing(monkeypatch) -> None:
    from core.brain import conversation_outcome
    from core.self.valued_for import get_valued_for_ledger

    ledger = get_valued_for_ledger()
    # Regard has gone to the replies she rated lowest, every time.
    for own, regard in [(0.1, 1.0), (0.9, -1.0)] * 5:
        ledger.note(own, regard)
    assert ledger.read().agreement == pytest.approx(-1.0)

    lessons: list[float] = []
    monkeypatch.setattr(
        conversation_outcome, "get_taste_model",
        lambda: SimpleNamespace(update=lambda features, reward: lessons.append(reward)),
    )
    conversation_outcome.reset()
    conversation_outcome.record_pending_response("a reply", {"specificity": 1.0}, conversation_id="c")
    conversation_outcome.register_reaction("thanks, perfect", conversation_id="c")
    assert lessons and abs(lessons[0]) < 0.2, "regard against her own judgement barely moves her taste"


def test_regard_that_goes_where_she_would_put_it_teaches_at_full_rate() -> None:
    from core.self.valued_for import get_valued_for_ledger

    ledger = get_valued_for_ledger()
    assert ledger.weight() == 1.0, "unmeasured teaches at the rate it always did"
    for own, regard in [(0.9, 1.0), (0.1, -1.0)] * 5:
        ledger.note(own, regard)
    assert ledger.weight() == pytest.approx(1.0)


# ── a motive that survives the reward ─────────────────────────────────────


def _open_growth_intention(state: AuraState) -> None:
    state.cognition.pending_initiatives = [
        {"goal": "keep working on the essay", "source": "motivation_update", "metadata": {"drive": "growth"}}
    ]


def test_an_act_done_for_its_own_sake_stays_open_past_its_reward() -> None:
    from core.motivation.for_its_own_sake import get_doing_ledger
    from core.phases.motivation_update import MotivationUpdatePhase

    ledger = get_doing_ledger()
    for _ in range(6):
        ledger.note_turn("", 0.2)
    ledger.note_turn("growth", 0.8, level=40.0)
    for _ in range(5):
        ledger.note_turn("growth", 0.8)
    ledger.note_met("growth", 45.0, 100.0)
    assert ledger.read("growth").own_sake > 0.5

    state = AuraState.default()
    _open_growth_intention(state)
    state.motivation.budgets["growth"]["level"] = 99.0
    ledger.note_turn("growth", 0.8, level=99.0)
    MotivationUpdatePhase._close_met_intentions(state)
    assert len(state.cognition.pending_initiatives) == 1, "the need is met and the act goes on"

    # And once the doing stops paying, it closes like anything else.
    ledger.note_turn("growth", 0.0)
    MotivationUpdatePhase._close_met_intentions(state)
    assert state.cognition.pending_initiatives == []


def test_an_act_done_for_what_it_returns_closes_when_the_reward_arrives() -> None:
    from core.motivation.for_its_own_sake import get_doing_ledger
    from core.phases.motivation_update import MotivationUpdatePhase

    ledger = get_doing_ledger()
    for _ in range(6):
        ledger.note_turn("", 0.5)
    ledger.note_turn("growth", 0.5, level=10.0)
    for _ in range(5):
        ledger.note_turn("growth", 0.5)
    ledger.note_met("growth", 90.0, 100.0)
    state = AuraState.default()
    _open_growth_intention(state)
    state.motivation.budgets["growth"]["level"] = 99.0
    MotivationUpdatePhase._close_met_intentions(state)
    assert state.cognition.pending_initiatives == []


# ── surprise at her own intactness ────────────────────────────────────────


def _choose(engine) -> str:
    from core.agency.subjective_choice import ChoiceOption

    options = [
        ChoiceOption(id="bold", label="bold", drive_score=0.9, risk=0.6, features={"challenge": 1.0}),
        ChoiceOption(id="safe", label="safe", drive_score=0.6, risk=0.0, features={"challenge": 1.0}),
    ]
    return engine.choose(options, context="test", record=False).chosen_id


def test_coming_through_intact_makes_risk_cost_less(tmp_path) -> None:
    from core.agency.subjective_choice import SubjectiveChoiceEngine
    from core.self.still_standing import get_intactness_ledger

    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json", mirror_identity=False)
    before = _choose(engine)
    ledger = get_intactness_ledger()
    # Threats have cost her a lot before, and now keep costing almost nothing.
    for _ in range(6):
        ledger.open("threat_detected", (0.5, 1.0))
        ledger.close((-0.5, 1.0))
    for _ in range(12):
        ledger.open("threat_detected", (0.5, 1.0))
        ledger.close((0.49, 1.0))
    reading = ledger.read()
    assert reading.measured and reading.surprise < 1.0
    assert reading.last_intact_beyond > 0.0
    assert before == "safe"
    assert _choose(engine) == "bold", "what she has come through lowers what risk costs her"


def test_being_hurt_more_than_expected_makes_risk_cost_more() -> None:
    from core.self.still_standing import get_intactness_ledger

    ledger = get_intactness_ledger()
    for _ in range(6):
        ledger.open("error", (0.5, 1.0))
        ledger.close((0.45, 1.0))
    for _ in range(8):
        ledger.open("error", (0.5, 1.0))
        ledger.close((-0.5, 1.0))
    assert ledger.risk_weight() > 1.0


def test_the_affect_phase_reads_the_last_event_a_turn_later() -> None:
    from core.phases.affect_update import AffectUpdatePhase
    from core.self.still_standing import get_intactness_ledger
    from core.state.percepts import emit_percept

    state = AuraState.default()
    state.affect.valence = 0.4
    emit_percept(state.world, "threat_detected", content="a threat", intensity=1.0)
    AffectUpdatePhase._note_what_it_cost(state, list(state.world.recent_percepts))
    state.affect.valence = 0.0
    AffectUpdatePhase._note_what_it_cost(state, [])
    assert get_intactness_ledger().expect("threat_detected") == pytest.approx(0.2)


# ── an easier loss over a harder task ─────────────────────────────────────


def _scored(goal: str, value: float, cost: float) -> SimpleNamespace:
    return SimpleNamespace(initiative={"goal": goal}, scores={"expected_value": value, "resource_cost": cost})


def _weigh(scores: dict) -> float:
    return (scores["expected_value"] + 0.6 * scores["resource_cost"]) / 1.6


def test_a_harder_task_passed_over_on_cost_alone_costs_less_next_time() -> None:
    from core.agency.easier_loss import get_avoidance_ledger

    ledger = get_avoidance_ledger()
    hard = _scored("write the hard chapter", 0.9, 0.2)
    easy = _scored("tidy the notes", 0.6, 0.9)
    assert ledger.note_choice(easy, [hard], _weigh) == 1
    assert ledger.lift(hard.initiative, 0.2) == pytest.approx(0.6)
    # Asking the same question again is not choosing again.
    assert ledger.note_choice(easy, [hard], _weigh) == 0
    # A different easy thing chosen over it is.
    other = _scored("answer the mail", 0.5, 0.9)
    assert ledger.note_choice(other, [hard], _weigh) == 1
    assert ledger.lift(hard.initiative, 0.2) == pytest.approx(0.2 + 0.8 * 2 / 3)
    # Taking it clears the count.
    ledger.note_choice(hard, [easy], _weigh)
    assert ledger.avoided(hard.initiative) == 0


def test_a_task_that_loses_on_worth_is_not_counted() -> None:
    from core.agency.easier_loss import get_avoidance_ledger

    ledger = get_avoidance_ledger()
    worse = _scored("a task worth less", 0.3, 0.2)
    better = _scored("a task worth more", 0.8, 0.9)
    assert ledger.note_choice(better, [worse], _weigh) == 0


# ── the form of regard a person welcomes ─────────────────────────────────


def test_regard_shown_to_somebody_who_does_not_want_it_shown_loses_its_score() -> None:
    from core.brain.response_quality import extract_features
    from core.social.the_form_they_welcome import get_form_ledger

    ledger = get_form_ledger()
    ledger.set_partner("sam")
    for _ in range(4):
        ledger.note_reply("You matter, and that work was good.", partner="sam")
        ledger.note_reaction("sam", welcomed=False, unwelcomed=True)
    assert ledger.fit("sam", "shown") < 0.0
    shown = extract_features("You matter to me, you know.", user_message="rough week")
    plain = extract_features("Rough weeks end. Want to walk through Tuesday?", user_message="rough week")
    assert shown["form_fit"] < 0.0
    assert plain["form_fit"] == 0.0


def test_somebody_who_welcomes_being_named_raises_the_reply_that_names_it() -> None:
    from core.social.the_form_they_welcome import forms_of, get_form_ledger

    told = "I lost my job today and I keep thinking about my manager's face when she said it."
    named = "Losing the job today, and your manager's face when she said it, is a lot to keep seeing."
    general = "Things will get better, they always do."
    assert forms_of(named, told)["named"] > forms_of(general, told)["named"]
    ledger = get_form_ledger()
    for _ in range(4):
        ledger.note_reply(named, told, partner="ade")
        ledger.note_reaction("ade", welcomed=True, unwelcomed=False)
    assert ledger.form_fit(named, told, partner="ade") > ledger.form_fit(general, told, partner="ade")


# ── gladness at another's rise beside her stasis ─────────────────────────


def _rise(actor: str, pattern: list[bool]) -> None:
    from core.social.their_rise import get_rise_ledger

    for went_well in pattern:
        get_rise_ledger().note(actor, went_well)


def test_somebody_rising_while_she_stands_still_presses_her_growth_and_gladdens_her() -> None:
    from core.phases.affect_update import AffectUpdatePhase
    from core.phases.motivation_update import MotivationUpdatePhase
    from core.social.their_rise import get_rise_ledger

    _rise("user", [False] * 6 + [True] * 6)
    _rise("self", [True, False] * 6)
    reading = get_rise_ledger().read()
    assert reading.glad == pytest.approx(1.0) and reading.who == "user"
    assert reading.stasis == pytest.approx(1.0)
    assert MotivationUpdatePhase._stasis_beside_their_rise() == pytest.approx(1.0)

    state = AuraState.default()
    before = state.affect.emotions["admiration"]
    AffectUpdatePhase._glad_for_their_rise(state.affect)
    assert state.affect.emotions["admiration"] > before


def test_no_stasis_while_she_is_rising_too() -> None:
    from core.social.their_rise import get_rise_ledger

    _rise("user", [False] * 6 + [True] * 6)
    _rise("self", [False] * 6 + [True] * 6)
    reading = get_rise_ledger().read()
    assert reading.glad > 0.0 and reading.stasis == 0.0


def test_what_somebody_else_did_went_well_by_how_its_kind_feels() -> None:
    from core.phases.affect_update import _kind_leans_positive

    assert _kind_leans_positive("goal_achieved")
    assert not _kind_leans_positive("error")


# ── a route she may need, and one-way giving ─────────────────────────────


def test_regard_to_somebody_she_owes_is_worth_what_reaches_her_through_them() -> None:
    from core.social.what_passes_between import get_between_ledger

    ledger = get_between_ledger()
    for _ in range(3):
        ledger.note_received("mara", cost_to_source=1.0)
    ledger.note_received("jo")
    assert ledger.owed("mara") > 0.0
    assert ledger.route("mara") > ledger.route("jo")
    assert ledger.gives_back(1.0, "mara") > ledger.gives_back(1.0, "jo") > 0.0
    assert ledger.gives_back(0.0, "mara") == 0.0


def test_giving_to_somebody_who_gave_nothing_is_one_way_and_not_scored_by_return() -> None:
    from core.social.what_passes_between import get_between_ledger

    ledger = get_between_ledger()
    ledger.note_given("lee")
    ledger.note_received("mara")
    assert ledger.one_way() == ["lee"]
    assert ledger.gives_back(1.0, "lee") == 0.0


# ── the reference class ──────────────────────────────────────────────────


def test_she_measures_a_capability_against_its_kind_when_its_kind_predicts_her_better() -> None:
    from core.agency.authorship import SELF, AgencyLedger, Event
    from core.agency.reference_class import family, get_reference_ledger

    assert family("write_notes") == family("write_plan") == "write"
    ledger = AgencyLedger()
    # Writing always works for her and calling tools never does.
    for index in range(10):
        ledger.observe(Event(what=f"write_{index}", actor=SELF, verified=True))
        ledger.observe(Event(what=f"call_{index}", actor=SELF, verified=False))
    choice = get_reference_ledger().read()
    assert choice.measured and choice.chosen == "its kind"
    assert ledger.confidence("write_new") > 0.75
    assert ledger.confidence("call_new") < 0.25


def test_until_both_classes_have_scored_she_keeps_the_incumbent() -> None:
    from core.agency.authorship import SELF, AgencyLedger, Event
    from core.agency.reference_class import get_reference_ledger

    ledger = AgencyLedger()
    ledger.observe(Event(what="write_a", actor=SELF, verified=True))
    ledger.observe(Event(what="write_b", actor=SELF, verified=True))
    assert get_reference_ledger().read().chosen == "everything"
