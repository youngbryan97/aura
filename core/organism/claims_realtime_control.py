"""Register measured counter presence and control evidence."""
from typing import Any


def install_realtime_control_claims(suite: Any) -> None:
    from core.agency.causal_identification import (
        _stale_and_ambiguous_identity_witnesses_remain_unknown,
    )
    from core.agency.what_meeting_things_does import _absent_counters_are_not_current
    from core.agency.where_clicks_pay import _click_attribution_respects_stretch_boundaries
    from core.agency.which_one_answers_to_her import (
        _ambiguous_controls_remain_unknown,
        _emissions_require_distinct_input_receipts,
        _fresh_control_experiment_forgets_old_rejections,
    )
    from core.cognition.reading_the_rules import _procedure_completion_is_measured
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )
    from core.perception.eyes_of_their_own import _child_capture_custody_invariant
    from core.perception.observed_transfer import _transfer_receipts_require_evidence
    from core.perception.the_drawing_as_objects import _scene_requires_a_matching_complete_frame
    from core.perception.where_the_words_point import _named_places_invariant
    from core.runtime.executors import _interactive_and_receipt_workers_are_separate
    from core.runtime.what_she_learned import _indexed_knowledge_invariant

    for name, owner, fixture, probe, statement in (
        ("procedures_require_owned_effects", "core/cognition/reading_the_rules.py",
         "tests/test_procedure_steps_need_their_own_evidence.py", _procedure_completion_is_measured,
         "Procedure completion requires the selected step's own observed effect and preserves unresolved prerequisites."),
        ("named_places_require_current_scene", "core/perception/where_the_words_point.py",
         "tests/test_named_places_keep_their_scene.py", lambda: _named_places_invariant() == (),
         "Named visual destinations belong to the current surface, episode, viewport and image revision."),
        ("transfers_require_measured_occupancy", "core/perception/observed_transfer.py",
         "tests/test_observed_transfer.py", lambda: _transfer_receipts_require_evidence() == (),
         "Transfer verification requires fresh measured pixels showing unique new occupancy at the bound destination."),
        ("child_pixels_keep_capture_custody", "core/perception/eyes_of_their_own.py",
         "tests/test_other_process_pixels_keep_their_capture.py", lambda: _child_capture_custody_invariant() == (),
         "Other-process screen readings preserve the exact final captured pixels and reject mismatched capture metadata."),
    ):
        suite.add_test(ValidationTest(
            name=name, description=statement, required_capability="",
            observation=Observation(name, True, fixture),
            predict=lambda _model, check=probe: check(),
            score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
            owner=owner,
        ))
        suite.add_claim(Claim(
            statement=statement, test=name, owner=owner, asserted_in=owner,
            evidence=Evidence.MEASURED_SYNTHETIC,
            evidence_note="Offline causal and unknown-state regressions; ordinary live demo performance remains independently unproved.",
        ))

    name = "emissions_require_distinct_input_receipts"
    owner = "core/agency/which_one_answers_to_her.py"
    suite.add_test(ValidationTest(
        name=name, description="emissions require distinct, unambiguous delivered input trials",
        required_capability="", observation=Observation("distinct_input_effects", True,
                "tests/test_steering_and_triggering_use_separate_inputs.py"),
        predict=lambda _model: _emissions_require_distinct_input_receipts() == (),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Emission learning accepts effects during input delivery and requires distinct unambiguous trials.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Delivery windows, repeated births, ambiguity and failed fresh trials measured; live gameplay requires separate evidence.",
    ))

    name = "indexed_knowledge_keeps_relationships"
    owner = "core/runtime/what_she_learned.py"
    suite.add_test(ValidationTest(
        name=name, description="bounded retention preserves the subject and shape of referenced evidence",
        required_capability="", observation=Observation("indexed_relationships_preserved", True,
                "tests/test_indexed_knowledge_keeps_its_relationships.py"),
        predict=lambda _model: _indexed_knowledge_invariant() == (),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Bounded indexed-state retention remaps retained references together and protects structural vectors.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Compaction, roundtrip and invalid legacy references measured; broader schema coverage requires separate evidence.",
    ))

    name = "click_attribution_respects_stretch_boundaries"
    owner = "core/agency/where_clicks_pay.py"
    suite.add_test(ValidationTest(
        name=name, description="counter resets release pending click credit while retaining learned places",
        required_capability="", observation=Observation("click_credit_rebased", True,
                "tests/test_where_clicks_pay.py"),
        predict=lambda _model: _click_attribution_respects_stretch_boundaries(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="A new stretch cannot credit a pending click with the preceding stretch's counter reset.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Counter reset and retained learning measured; useful live control requires separate evidence.",
    ))

    name = "identity_requires_current_distinguishing_evidence"
    owner = "core/agency/causal_identification.py"
    suite.add_test(ValidationTest(
        name=name, description="stale experiments and equal responder evidence leave identity unknown",
        required_capability="", observation=Observation("causal_identity_unknown", True,
                "tests/test_causal_identification.py"),
        predict=lambda _model: _stale_and_ambiguous_identity_witnesses_remain_unknown(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Stale or indistinguishable causal witnesses cannot establish a control identity.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Measured evidence epochs and ambiguity; broader semantic identification requires separate evaluation.",
    ))

    name = "interactive_and_receipt_workers_are_separate"
    owner = "core/runtime/executors.py"
    suite.add_test(ValidationTest(
        name=name, description="observation and durable receipt workers have separate pool ownership",
        required_capability="", observation=Observation("separate_workers", True,
                "tests/test_interactive_work_survives_shared_pool_saturation.py"),
        predict=lambda _model: _interactive_and_receipt_workers_are_separate(),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Interactive perception and durable receipt work have separate workers from background work.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Pool ownership and default-pool saturation measured; end-to-end latency needs separate live runs.",
    ))

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
