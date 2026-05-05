from core.environment.outcome_attribution import OutcomeAttributor
from core.environment.prediction_error import PredictionError


def test_success_without_visible_progress_for_information_gain():
    outcome = OutcomeAttributor().assess(
        action="inspect",
        expected_effect="",
        observed_events=[],
        information_gain=0.7,
    )
    assert outcome.success_score >= 0.8
    assert outcome.information_gain == 0.7


def test_visible_change_not_automatically_success_when_harmful():
    outcome = OutcomeAttributor().assess(
        action="submit",
        expected_effect="submitted",
        observed_events=["submitted"],
        resource_delta={"trust": -1.0},
    )
    assert outcome.harm_score > 0
    assert outcome.success_score <= 0.5


def test_prediction_surprise_creates_lesson():
    pe = PredictionError("a", ["pass"], ["fail"], ["pass"], ["fail"], 1.0, ["unexpected"])
    outcome = OutcomeAttributor().assess(action="run_test", expected_effect="pass", observed_events=["fail"], prediction_error=pe)
    assert outcome.surprise == 1.0
    assert outcome.lesson
