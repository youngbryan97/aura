"""Asking the impulse, wired into choice, state, schema and clamp.

The measure is pinned in test_an_impulse_asked_for_directions_has_a_record.py.
These pin the wiring: when her appraised choices say impulse has led her worse
than weighing, a risky option she can read no preference for loses to a safer
one it would otherwise beat, an impulse that served her as well changes
nothing, and the record reaches the action domain the clamp holds.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from core.agency.asking_the_impulse import MIN_SAMPLES
from core.agency.subjective_choice import ChoiceOption, SubjectiveChoiceEngine


@pytest.fixture
def engine(tmp_path: Path) -> SubjectiveChoiceEngine:
    return SubjectiveChoiceEngine(state_path=tmp_path / "choices.json", mirror_identity=False)


def _risky_impulse_and_safe_weighed() -> list[ChoiceOption]:
    """With no record: rush scores 0.55 * 0.95 - 0.35 * 0.3 = 0.4175 and plan 0.55 * 0.6 = 0.33.

    Where impulse-led choices satisfied her 1.2 less, rush's risk costs 2.2
    times as much and it scores 0.2915, below plan.
    """
    return [
        ChoiceOption(id="rush", label="rush", description="", drive_score=0.95, risk=0.3, features={"x": 0.0}),
        ChoiceOption(id="plan", label="plan", description="", drive_score=0.6, risk=0.0, features={"x": 0.0}),
    ]


def _history(engine: SubjectiveChoiceEngine, *, impulse: float, weighed: float) -> None:
    for satisfaction, features in ((impulse, {"x": 0.0}), (weighed, {"novelty": 0.8})):
        for _ in range(MIN_SAMPLES):
            receipt = engine.choose(
                [ChoiceOption(id="a", label="a", description="", drive_score=0.5, risk=0.0, features=features)],
                context="history",
            )
            engine.appraise_outcome(receipt.choice_id, outcome="done", satisfaction=satisfaction)


def test_with_no_record_the_stronger_impulse_wins(engine) -> None:
    receipt = engine.choose(_risky_impulse_and_safe_weighed(), context="now", record=False)
    assert receipt.chosen_id == "rush"


def test_an_impulse_that_led_her_worse_makes_its_risk_cost_more(engine) -> None:
    _history(engine, impulse=-0.6, weighed=0.6)
    before = engine.rank_options(_risky_impulse_and_safe_weighed(), context="now")
    receipt = engine.choose(_risky_impulse_and_safe_weighed(), context="now", record=False)
    assert receipt.chosen_id == "plan"
    assert before[0]["id"] == "plan"


def test_an_impulse_that_served_her_as_well_changes_nothing(engine) -> None:
    _history(engine, impulse=0.6, weighed=0.2)
    receipt = engine.choose(_risky_impulse_and_safe_weighed(), context="now", record=False)
    assert receipt.chosen_id == "rush"


def test_the_record_reaches_the_action_domain_and_the_clamp() -> None:
    from core.subject.clamp import CLAMPED_FIELDS
    from core.subject.state import feature_names

    assert "D.impulse_distrust" in feature_names("D")
    assert "cognition.impulse" in CLAMPED_FIELDS["D"]
