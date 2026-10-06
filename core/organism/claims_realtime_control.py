"""Register the measured separation of current readings from counter history."""
from typing import Any


def install_realtime_control_claims(suite: Any) -> None:
    from core.agency.what_meeting_things_does import _absent_counters_are_not_current
    from core.organism.model_validation import Claim, Evidence, Observation, ValidationTest, boolean_score

    name = "current_counters_exclude_absent_history"
    owner = "core/agency/what_meeting_things_does.py"
    suite.add_test(ValidationTest(
        name=name, description="a counter absent from a new reading remains history without becoming current",
        required_capability="",
        observation=Observation("absent_counter_is_history_only", True,
                                "tests/test_a_new_attempt_measures_its_own_state.py"),
        predict=lambda _model: _absent_counters_are_not_current(),
        score=lambda value, observation: boolean_score(value, expected=observation.value, subject=name),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Current counter readings exclude labels absent from the latest observation.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured counter history and presence; game wins and general environment coverage require separate runs.",
    ))
