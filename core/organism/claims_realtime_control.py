"""Register measured counter presence and control evidence."""
from typing import Any


def install_realtime_control_claims(suite: Any) -> None:
    from core.agency.what_meeting_things_does import _absent_counters_are_not_current
    from core.agency.which_one_answers_to_her import _ambiguous_controls_remain_unknown
    from core.organism.model_validation import Claim, Evidence, Observation, ValidationTest, boolean_score

    name = "current_counters_exclude_absent_history"
    owner = "core/agency/what_meeting_things_does.py"
    suite.add_test(ValidationTest(
        name=name, description="a counter absent from a new reading remains history without becoming current",
        required_capability="",
        observation=Observation("absent_counter_is_history_only", True,
                                "tests/test_a_new_attempt_measures_its_own_state.py"),
        predict=lambda _model: _absent_counters_are_not_current(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Current counter readings exclude labels absent from the latest observation.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured counter history and presence; game wins and general environment coverage require separate runs.",
    ))
    name = "ambiguous_controls_remain_unknown"
    owner = "core/agency/which_one_answers_to_her.py"
    suite.add_test(ValidationTest(
        name=name, description="blocked and unsettled observations leave a key available for another trial",
        required_capability="",
        observation=Observation("ambiguous_control_is_unknown", True,
                                "tests/test_which_thing_answers_to_her.py"),
        predict=lambda _model: _ambiguous_controls_remain_unknown(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Blocked and unsettled motion cannot establish a key's control response.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured ambiguous and free samples; actual task completion requires separate live evidence.",
    ))
