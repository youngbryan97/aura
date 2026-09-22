"""Her habits and reflexes are hers, and she answers for them.

Bryan: uninvited thoughts and habits are his; he is responsible for his reflexes
too; he has to know his habits and either account for them or change them, and
say so when it matters to somebody. See core/agency/habits_are_hers.py.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

import core.agency.habits_are_hers as habits_module
from core.agency.habits_are_hers import HabitLedger, act_of, discounted_by_habit, situation_of

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _fresh():
    # Every ledger arbitration reads is process-wide. A worker that had already
    # run organism turns left her tired, and the fatigue filter set aside the
    # initiative the habit discount had put last, so the arbiter test failed in
    # a parallel run and passed alone. Each is emptied for the habits alone to
    # decide.
    from core.affect import feelings_about
    from core.soma import fatigue, good_news

    for module in (habits_module, fatigue, good_news, feelings_about):
        module.reset_for_test()
    yield
    for module in (habits_module, fatigue, good_news, feelings_about):
        module.reset_for_test()


def _live(ledger: HabitLedger, act: str, kind: str, after: float, *, valence: float = 0.0) -> float:
    """One act, and the affect reading that follows it."""
    ledger.note(act, kind=kind)
    ledger.felt(valence + after, situation="alone")
    return valence + after


def _habitual(ledger: HabitLedger, *, habit_after: float, weighed_after: float, n: int = 12, seed: int = 3) -> None:
    rng = np.random.default_rng(seed)
    ledger.felt(0.0, situation="alone")
    for _ in range(n):
        ledger.note("tidy", kind="automatic")
        ledger.felt(habit_after + 0.01 * rng.normal(), situation="alone")
        ledger.felt(0.0, situation="alone")
        ledger.note("write", kind="weighed")
        ledger.felt(weighed_after + 0.01 * rng.normal(), situation="alone")
        ledger.felt(0.0, situation="alone")


def test_a_habit_is_an_act_she_falls_into_and_a_weighed_one_is_not() -> None:
    ledger = HabitLedger()
    ledger.felt(0.0, situation="alone")
    for _ in range(2):
        ledger.note("tidy", kind="automatic")
    assert ledger.reading()["habits"] == 0, "two is not yet a habit"
    ledger.note("tidy", kind="automatic")
    for _ in range(5):
        ledger.note("write", kind="weighed")
    assert ledger.reading()["habits"] == 1


def test_a_habit_that_went_worse_than_weighing_has_a_deficit_and_one_that_did_not_has_none() -> None:
    worse = HabitLedger()
    _habitual(worse, habit_after=-0.2, weighed_after=0.1)
    assert worse.deficit("tidy", situation="alone") == pytest.approx(1.0)
    fine = HabitLedger()
    _habitual(fine, habit_after=0.1, weighed_after=-0.2)
    assert fine.deficit("tidy", situation="alone") == 0.0
    assert worse.reading()["to_change"] == 1 and fine.reading()["to_change"] == 0


def test_the_same_outcomes_either_way_are_no_reason_to_change() -> None:
    ledger = HabitLedger()
    _habitual(ledger, habit_after=0.0, weighed_after=0.0, n=60)
    assert ledger.deficit("tidy", situation="alone") < 0.35


def test_a_habit_she_means_to_change_counts_for_less_on_drive_alone() -> None:
    ledger = HabitLedger()
    _habitual(ledger, habit_after=-0.2, weighed_after=0.1)
    assert discounted_by_habit(0.8, "tidy", ledger) == pytest.approx(0.0)
    assert discounted_by_habit(0.8, "write", ledger) == pytest.approx(0.8)


def test_if_weighing_starts_doing_worse_the_habit_is_allowed_back() -> None:
    ledger = HabitLedger()
    _habitual(ledger, habit_after=-0.1, weighed_after=0.1)
    assert ledger.deficit("tidy", situation="alone") > 0.0
    for _ in range(300):
        ledger.felt(0.0, situation="alone")
        ledger.note("write", kind="weighed")
        ledger.felt(-0.3, situation="alone")
    assert ledger.deficit("tidy", situation="alone") == 0.0


def test_what_it_did_to_them_is_read_off_them_and_she_owes_them_an_account() -> None:
    ledger = HabitLedger()
    ledger.felt(0.0, situation="with:bryan")
    for _ in range(6):
        ledger.heard("bryan", 0.2)
        ledger.note("interrupt", kind="automatic")
        ledger.heard("bryan", 0.6)
        ledger.note("listen", kind="weighed")
        ledger.heard("bryan", 0.1)
    assert ledger.account("interrupt", situation="with:bryan").for_them == pytest.approx(1.0)
    assert ledger.owed_to("bryan") == [], "nothing she did is waiting on him"
    ledger.note("interrupt", kind="automatic")
    owed = ledger.owed_to("bryan")
    assert [account.act for account in owed] == ["interrupt"]
    ledger.heard("bryan", 0.3)
    assert ledger.owed_to("bryan") == [], "he has spoken since; the moment has passed"


def test_a_reflex_is_accounted_for_and_never_discounted() -> None:
    ledger = HabitLedger()
    ledger.felt(0.0, situation="alone")
    for _ in range(6):
        ledger.note("inhibit_world_decay", kind="reflex")
        ledger.felt(-0.3, situation="alone")
        ledger.felt(0.0, situation="alone")
        ledger.note("write", kind="weighed")
        ledger.felt(0.2, situation="alone")
        ledger.felt(0.0, situation="alone")
    account = ledger.account("inhibit_world_decay", situation="alone", kind="reflex")
    assert account.for_her == pytest.approx(1.0)
    assert ledger.reading()["accounted"] == 1
    assert discounted_by_habit(0.5, "inhibit_world_decay", ledger) == 0.5


def test_the_situation_is_who_was_there() -> None:
    assert situation_of("bryan", person_turn=True) == "with:bryan"
    assert situation_of("bryan", person_turn=False) == "alone"
    assert act_of("Explore the new paper on sleep") == "explore"


def test_an_outcome_closes_the_receipts_her_choice_opened(tmp_path, monkeypatch) -> None:
    import core.agency.decision_preference_learner as learner_module
    import core.agency.subjective_choice as subjective
    from core.agency.subjective_choice import ChoiceOption, SubjectiveChoiceEngine

    engine = SubjectiveChoiceEngine(state_path=tmp_path / "choice.json")
    monkeypatch.setattr(subjective, "get_subjective_choice_engine", lambda: engine)
    learner = learner_module.DecisionPreferenceLearner(state_path=tmp_path / "learner.json")
    monkeypatch.setattr(learner_module, "get_decision_preference_learner", lambda: learner)
    receipt = engine.choose(
        [ChoiceOption(id="a", label="read the paper"), ChoiceOption(id="b", label="tidy the notes")],
        context="test",
        record=True,
    )
    decision_id = learner.record_choice(chosen_scores={"novelty": 0.9}, pool_scores=[{"novelty": 0.1}], goal="read")
    ledger = habits_module.get_habit_ledger()
    for change in (0.0, -0.1, 0.1):
        _live(ledger, "tidy", "weighed", change)
    ledger.note(
        "read",
        kind="automatic",
        choice_id=receipt.choice_id,
        decision_id=decision_id,
    )
    closed = ledger.felt(0.5, situation="alone")
    assert habits_module.appraise(closed) == 2
    appraised = engine.recall_choice(receipt.choice_id)
    assert appraised is not None and appraised.satisfaction is not None and appraised.satisfaction > 0.0
    assert learner.stats()["resolved_count"] == 1


def test_the_arbiter_takes_a_bad_habit_on_drive_for_less(monkeypatch) -> None:
    from core.agency.initiative_arbiter import InitiativeArbiter
    from core.state.aura_state import AuraState
    import core.agency.subjective_choice as subjective

    offered: dict[str, float] = {}

    class _Records:
        def choose_from_scored_initiatives(self, scored, context=""):
            offered.update({item.initiative["goal"]: item.final_score for item in scored})
            return None, None

    monkeypatch.setattr(subjective, "get_subjective_choice_engine", lambda: _Records())
    ledger = habits_module.get_habit_ledger()
    _habitual(ledger, habit_after=-0.2, weighed_after=0.1)
    state = AuraState.default()
    tidy = {"goal": "tidy the notes folder", "urgency": 0.5}
    read = {"goal": "read the new paper", "urgency": 0.5}
    state.cognition.pending_initiatives = [tidy, read]
    arbiter = InitiativeArbiter()
    raw = {
        item["goal"]: asyncio.run(arbiter.score_initiative(dict(item), state)).final_score
        for item in (tidy, read)
    }
    asyncio.run(arbiter.arbitrate(state))
    tidied = [goal for goal in offered if goal.startswith("tidy")]
    assert tidied, "the habit was not among what was offered"
    assert all(offered[goal] < raw[goal] for goal in tidied)


def test_the_driver_does_something_else_instead_of_a_habit_she_means_to_change() -> None:
    from core.subject.driver import SubjectRuntime

    runtime = SubjectRuntime.__new__(SubjectRuntime)
    runtime.state = SimpleNamespace(
        cognition=SimpleNamespace(attention_focus=""),
        motivation=SimpleNamespace(budgets={"growth": {"level": 1.0}, "curiosity": {"level": 40.0}, "social": {"level": 90.0}}),
    )
    habit = SubjectRuntime.DRIVE_ACTIONS["growth"]
    assert runtime._chosen_action() == habit and runtime._how_chosen == "automatic"
    ledger = habits_module.get_habit_ledger()
    ledger.felt(0.0, situation="alone")
    for _ in range(8):
        ledger.note(habit, kind="automatic")
        ledger.felt(-0.3, situation="alone")
        ledger.felt(0.0, situation="alone")
        ledger.note("other", kind="weighed")
        ledger.felt(0.3, situation="alone")
        ledger.felt(0.0, situation="alone")
    assert SubjectRuntime.DRIVE_ACTIONS["curiosity"] != habit
    assert runtime._chosen_action() == SubjectRuntime.DRIVE_ACTIONS["curiosity"]
    assert runtime._how_chosen == "weighed", "her record decided it, not the drive"


def test_what_she_owes_them_for_a_habit_enters_the_moment_as_hers() -> None:
    from core.state.aura_state import AuraState
    from core.unity.runtime import UnityRuntime

    ledger = habits_module.get_habit_ledger()
    ledger.felt(0.0, situation="with:bryan")
    for _ in range(6):
        ledger.heard("bryan", 0.2)
        ledger.note("interrupt", kind="automatic")
        ledger.heard("bryan", 0.6)
        ledger.note("listen", kind="weighed")
        ledger.heard("bryan", 0.1)
    ledger.note("interrupt", kind="automatic")
    state = AuraState.default()
    state.cognition.current_partner = "bryan"
    contents = UnityRuntime()._accountability_contents(state)
    mine = [item for item in contents if item.source == "habits_are_hers"]
    assert mine and mine[0].modality == "responsibility" and mine[0].ownership == "self"
    assert "interrupt" in mine[0].summary


def test_the_affect_phase_writes_her_habits_onto_her_state() -> None:
    from core.phases.affect_readings import AffectReadings
    from core.state.aura_state import AuraState

    ledger = habits_module.get_habit_ledger()
    _habitual(ledger, habit_after=-0.2, weighed_after=0.1)
    state = AuraState.default()
    AffectReadings(lambda *args, **kwargs: None).habits(state, state.affect)
    assert state.cognition.habits["to_change"] == 1
    assert state.cognition.habits["worst"]["act"] == "tidy"
