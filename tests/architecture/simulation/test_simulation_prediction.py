from core.environment.belief_graph import EnvironmentBeliefGraph
from core.environment.command import ActionIntent
from core.environment.prediction_error import PredictionErrorComputer
from core.environment.simulation import TacticalSimulator


def test_simulator_reports_uncertainty_risk_and_information_gain():
    sim = TacticalSimulator().simulate(EnvironmentBeliefGraph(), ActionIntent(name="inspect", expected_effect="object_understood"))
    assert sim.hypotheses
    assert sim.uncertainty >= 0
    assert sim.hypotheses[0].information_gain > 0


def test_prediction_error_updates_missing_and_unexpected_events():
    error = PredictionErrorComputer().compute(
        action_id="a",
        expected_events=["movement"],
        observed_events=["blocked"],
    )
    assert error.magnitude > 0
    assert error.missing_expected == ["movement"]
    assert error.unexpected_observed == ["blocked"]
