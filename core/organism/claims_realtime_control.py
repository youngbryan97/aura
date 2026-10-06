"""Register measured counter presence and control evidence."""
from typing import Any


def install_realtime_control_claims(suite: Any) -> None:
    from core.agency.what_meeting_things_does import _absent_counters_are_not_current
    from core.agency.which_one_answers_to_her import (
        _ambiguous_controls_remain_unknown,
        _fresh_control_experiment_forgets_old_rejections,
    )
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )
    from core.perception.the_drawing_as_objects import _scene_requires_a_matching_complete_frame

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
    name = "fresh_control_experiment_releases_old_evidence"
    owner = "core/agency/which_one_answers_to_her.py"
    suite.add_test(ValidationTest(
        name=name, description="a fresh experiment discards previous rejections and response calibration",
        required_capability="", observation=Observation("control_evidence_released", True,
                "tests/test_exact_executor_connections.py"),
        predict=lambda _model: _fresh_control_experiment_forgets_old_rejections(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A fresh control experiment releases prior object rejections and response calibration.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Evidence reset measured; actual reacquisition requires separate live runs.",
    ))
    name = "drawing_scene_requires_matching_frame"
    owner = "core/perception/the_drawing_as_objects.py"
    suite.add_test(ValidationTest(
        name=name, description="only complete, bounded scene descriptions of the captured frame are accepted",
        required_capability="", observation=Observation("matching_frame_required", True,
                "tests/test_the_drawing_can_describe_its_objects.py"),
        predict=lambda _model: _scene_requires_a_matching_complete_frame(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A structured drawing observation requires a complete description matching its captured pixels' dimensions.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Frame and metadata acceptance measured; supported paint and actual control are measured in separate browser runs.",
    ))
    from core.intent.capability_selection import _artifact_lane_is_owned_by_the_declaration

    name = "artifact_lane_follows_declared_effect"
    owner = "core/intent/capability_selection.py"
    suite.add_test(ValidationTest(
        name=name, description="an artifact effect takes precedence over a generic desktop shortcut",
        required_capability="", observation=Observation("declared_artifact_owner", True,
                "tests/test_artifact_work_precedes_a_desktop_shortcut.py"),
        predict=lambda _model: _artifact_lane_is_owned_by_the_declaration(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A strongest artifact declaration defers generic desktop shortcuts to capability selection.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Lane ownership measured; autonomous repair and use require separate live evidence.",
    ))
