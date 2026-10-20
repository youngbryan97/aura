"""Register measured counter presence and control evidence."""
from typing import Any


def _play_execution_claims() -> tuple:
    from core.perception.a_source_method_with_resolved_values import _resolved_source_step_invariant
    from core.agency.a_contrast_that_meets_both_tests import _joint_contrast_invariant
    from core.agency.evidence_for_a_requested_relation import _referent_evidence_invariant
    from core.agency.a_control_trial_and_its_observation_reserve import _trial_reserve_invariant
    from core.agency.motion_models_on_the_same_intervals import _forward_motion_models_invariant
    from core.perception.a_capture_owned_by_the_input_surface import _owned_input_surface_invariant
    from core.agency.looking_for_an_unbound_referent import _referent_attention_invariant
    from core.agency.a_requested_relation_in_play import _destination_reading_invariant
    from core.agency.why_a_motion_history_is_called_carried import _motion_family_reading_invariant
    from core.brain.llm.a_model_in_the_local_cache import _local_model_cache_invariant
    from core.perception.what_things_look_like import _visual_naming_clock_invariant
    from core.perception.material_displacement_from_named_pixels import _material_displacement_precision_invariant
    from core.agency.responses_from_measured_intervals import _measured_interval_trials_invariant
    from core.skills.screen_pursuit_as_it_happens import _coordinate_label_invariant
    from core.skills.known_navigation_after_discovery import _known_navigation_use_invariant
    from core.perception.recording_the_frames_and_inputs import (
        _keyboard_target_receipt_invariant,
        _source_reference_record_invariant,
    )
    from core.runtime.classical_compute_boundary import _classical_import_invariant
    from core.perception.guarded_effects_in_flash_methods import _guarded_source_graph_invariant
    from core.perception.source_geometry_in_current_pixels import _fresh_source_capture_invariant
    from core.perception.source_geometry_in_current_pixels import _reference_attention_invariant
    from core.perception.source_geometry_in_current_pixels import _optional_reference_share_invariant
    from core.perception.reference_material_near_a_measured_pose import _near_reference_material_invariant
    from core.perception.reference_material_near_a_measured_pose import _single_position_refit_invariant
    from core.perception.points_a_consumer_still_needs import _point_retention_visibility_invariant
    from core.perception.symbolic_motion_in_flash_methods import _source_deletion_invariant
    from core.perception.symbolic_motion_in_flash_methods import _source_boolean_arithmetic_invariant
    from core.perception.the_pixel_grid_a_mark_uses import _sample_grid_invariant
    from core.perception.named_material_in_a_consumed_frame import _named_material_capture_invariant
    from core.perception.a_source_point_on_visible_marks import _visible_reference_identity_invariant
    from core.perception.visual_references_for_current_sources import _current_reference_source_invariant
    from core.perception.source_references_while_play_advances import (
        _reference_catalog_readiness_invariant,
        _reference_worker_retirement_invariant,
    )
    from core.agency.enough_observations_for_a_trial import _trial_observation_goal_invariant
    from core.agency.time_to_observe_a_control import _phase_reservation_invariant
    from core.agency.trials_with_room_to_observe import _trial_priority_invariant
    from core.agency.which_one_answers_to_her import (
        _control_sample_custody_invariant,
        _inertial_axis_resolution_invariant,
        _later_control_interval_invariant,
    )
    from core.capabilities.browser_graphics_policy import _software_graphics_invariant
    from core.perception.a_material_observation_from_its_parts import _material_parts_invariant
    from core.perception.coordinates_in_a_visual_reference import _source_image_coordinates_invariant
    from core.perception.a_visual_reference_in_the_picture import _oriented_reference_invariant
    from core.perception.a_visual_reference_in_the_picture import _reduced_reference_centres_invariant
    from core.perception.a_visual_reference_in_the_picture import _scene_correlation_invariant
    from core.perception.names_for_observed_instances import _instance_name_invariant
    from core.perception.a_source_point_on_current_material import _source_point_basis_invariant
    from core.perception.pixels_consumed_by_the_observer import _consumed_pixel_invariant
    from core.perception.a_material_frame_kept_for_its_owner import _material_frame_owner_invariant
    from core.perception.affine_motion_from_named_points import _affine_material_invariant
    from core.perception.position_from_the_same_points import (
        _fresh_mark_capacity_invariant,
        _surviving_reference_invariant,
    )
    from core.perception.presses_on_the_same_material import _material_press_invariant
    from core.perception.recording_the_frames_and_inputs import (
        _navigation_port_recording_invariant,
        _recorded_pixel_invariant,
        _screen_options_invariant,
    )
    from core.perception.vision_request_compute import _cpu_vision_request_invariant
    from core.runtime.model_worker_policy import _model_worker_policy_invariant
    from core.skills.screen_pursuit_on_a_page import _read_before_scene_wait_invariant
    from core.skills.sovereign_browser_drawing import _cover_press_invariant, _notice_word_boundaries_invariant
    from core.skills.sovereign_browser_the_program import _current_program_bytes_invariant

    return (('source_arithmetic_requires_explicit_values_and_effects', 'core/perception/a_source_method_with_resolved_values.py',
             'tests/test_source_methods_resolve_only_supplied_values.py', lambda: _resolved_source_step_invariant() == (),
             'Bounded source arithmetic selects one known path from supplied primitive values, retains its input and instruction context, and publishes no partial update when values or effects remain unknown; it certifies no visual state or live outcome.'),
            ('contrast_ranking_preserves_joint_admission', 'core/agency/a_contrast_that_meets_both_tests.py',
             'tests/test_a_valid_contrast_survives_aggregate_ranking.py', lambda: _joint_contrast_invariant() == (),
             'A contrast meeting the caller\'s statistic and physical-effect gates survives aggregate ranking; invalid measurements do not qualify and thresholds and independent trial counts remain unchanged.'),
            ('referent_candidates_cannot_certify_contact', 'core/agency/evidence_for_a_requested_relation.py',
             'tests/test_missing_referents_request_specific_evidence.py', lambda: _referent_evidence_invariant() == (),
             'A destination refusal retains candidate identity and the specific missing observation; geometric approach regions preserve unknown collision, speed, angle and contact requirements.'),
            ('discovery_records_its_observation_deadline_and_owner', 'core/agency/a_control_trial_and_its_observation_reserve.py',
             'tests/test_trial_arbitration_keeps_one_input_owner.py', lambda: _trial_reserve_invariant() == (),
             'Discovery records one deadline and owner for predicted observation reserve; measured protection stays separate from calibration, and a new prepared planner cannot steal an owned trial.'),
            ('motion_models_share_forward_heldout_observations', 'core/agency/motion_models_on_the_same_intervals.py',
             'tests/test_motion_models_predict_the_same_future_intervals.py', lambda: _forward_motion_models_invariant() == (),
             'Acceleration and direct-speed fits predict the same later measured physical holds; one hold votes once, initial acceleration keeps its causal admission gates, and switching needs independent contrary holds.'),
            ('qualified_input_surfaces_keep_current_capture_custody', 'core/perception/a_capture_owned_by_the_input_surface.py',
             'tests/test_qualified_input_surfaces_survive_wordless_scenes.py', lambda: _owned_input_surface_invariant() == (),
             'A qualified input adapter can admit a fresh capture of a changed or wordless scene on the same active surface; copied metadata, stale captures, changed ownership and input-invalidated readings confer no authority.'),
            ('unbound_referents_request_current_instance_observation', 'core/agency/looking_for_an_unbound_referent.py',
             'tests/test_an_unbound_referent_reaches_visual_attention.py', lambda: _referent_attention_invariant() == (),
             'An unresolved destination prioritizes current owned instance queries while preserving query custody and the existing binder; geometry orders observations without assigning a target or a control role.'),
            ('play_refusals_keep_the_predicates_that_decided', 'core/agency/why_a_motion_history_is_called_carried.py',
             'tests/test_play_refusals_keep_the_predicates_that_decided.py',
             lambda: _motion_family_reading_invariant() == () and _destination_reading_invariant() == (),
             'Recorded motion-family and destination readings preserve the predicates and refusal branches used by selection; unvisited tests remain unavailable and diagnostic truncation cannot change a decision.'),
            ('current_reading_precedes_waiting_in_a_running_world', 'core/skills/screen_pursuit_on_a_page.py',
             'tests/test_current_reading_precedes_waiting_in_a_running_world.py', lambda: _read_before_scene_wait_invariant() == (),
             'A known running clock gives current reading priority over scene settling, while current words still decide navigation and no input authority is added.'),
            ('local_model_cache_cannot_supply_remote_authority', 'core/brain/llm/a_model_in_the_local_cache.py',
             'tests/test_a_cached_model_is_resolved_without_a_download.py', lambda: _local_model_cache_invariant() == (),
             'A read-only local cache lookup rejects path-shaped requests; a snapshot supplies neither a download nor serving-role or weight-completeness authority.'),
            ('visual_naming_accepts_measured_capture_times', 'core/perception/what_things_look_like.py',
             'tests/test_visual_naming_accepts_the_capture_clock.py', lambda: _visual_naming_clock_invariant() == (),
             'Finite real capture clocks reach the existing asynchronous instance naming path; invalid times cannot age it and box, birth, frame bytes and provider-answer custody remain required.'),
            ('material_displacement_separates_shared_reference_error', 'core/perception/material_displacement_from_named_pixels.py',
             'tests/test_material_displacement_keeps_its_reference_precision.py', lambda: _material_displacement_precision_invariant() == (),
             'Local named-pixel displacement cancels common translation bias and transports rotation and closure error; missing correspondence supplies no precision.'),
            ('measured_intervals_keep_one_physical_trial', 'core/agency/responses_from_measured_intervals.py',
             'tests/test_responses_keep_their_measured_intervals.py', lambda: _measured_interval_trials_invariant() == (),
             'Continuous material intervals date velocity at exposure midpoints and propagate estimated error; gaps cannot create independent physical trials and active contradictions retire their estimates.'),
            ('coordinates_cannot_rename_current_words', 'core/skills/screen_pursuit_as_it_happens.py',
             'tests/test_coordinates_cannot_rename_current_words.py', lambda: _coordinate_label_invariant() == (),
             'Earlier control coordinates cannot rename current counters or unread labels; pixel-backed recovery and explicit reset interpretation remain separate.'),
            ('reference_metadata_does_not_spend_match_cadence', 'core/perception/source_references_while_play_advances.py',
             'tests/test_reference_work_reaches_current_ordinary_observation.py', lambda: _reference_catalog_readiness_invariant() == (),
             'A completed reference catalog releases the first pixel search without spending its cadence; later searches retain their rate and one-worker capacity.'),
            ('keyboard_target_receipts_do_not_claim_acceptance', 'core/perception/recording_the_frames_and_inputs.py',
             'tests/test_drawing_focus_is_current_for_each_key.py', lambda: _keyboard_target_receipt_invariant() == (),
             'A current keyboard-target query retains its page and interval separately from input transport, game acceptance, controlled identity and task completion.'),
            ('navigation_use_is_separate_from_discovery', 'core/skills/known_navigation_after_discovery.py',
             'tests/test_known_navigation_is_useful_after_discovery.py', lambda: _known_navigation_use_invariant() == (),
             'A current offered navigation control can retain its recorded forward use after discovery; missing controls, reset labels and contradicted routes gain no new priority.'),
            ('later_control_intervals_keep_one_input', 'core/agency/which_one_answers_to_her.py',
             'tests/test_later_control_intervals_keep_one_physical_press.py', lambda: _later_control_interval_invariant() == (),
             'A later continuous measured interval can supply acceleration samples for the same physical input, without joining a missing stretch or counting another independent press.'),
            ('optional_reference_refits_one_position', 'core/perception/reference_material_near_a_measured_pose.py',
             'tests/test_optional_reference_refits_one_position.py', lambda: _single_position_refit_invariant() == (),
             'An ordinary budgeted reference visit fits one position of one pose; failed visits rotate across local and wide peaks, while every published match still needs current material and precision.'),
            ('optional_geometry_repays_observation_time', 'core/perception/source_geometry_in_current_pixels.py',
             'tests/test_optional_geometry_shares_observation_time.py', lambda: _optional_reference_share_invariant() == (),
             'Optional current-pixel refits repay measured elapsed work between essential observations; one operation can overdraw its allowance, and deferred fits supply no geometry or control evidence.'),
            ('optional_reference_attention_keeps_fair_audits', 'core/perception/source_geometry_in_current_pixels.py',
             'tests/test_fresh_source_geometry_reaches_ordinary_observation.py', lambda: _reference_attention_invariant() == (),
             'Optional reference work refits one pose per frame, retaining unsearched alternatives and fair audits; priority hints grant no current geometry, identity or control authority.'),
            ('fresh_source_geometry_keeps_the_consumed_capture', 'core/perception/source_geometry_in_current_pixels.py',
             'tests/test_fresh_source_geometry_reaches_ordinary_observation.py', lambda: _fresh_source_capture_invariant() == (),
             'Fresh source correspondence publishes geometry only for the exact consumed capture and current source; it does not supply continuing object identity or control authority.'),
            ('previous_reference_poses_need_fresh_pixel_support', 'core/perception/reference_material_near_a_measured_pose.py',
             'tests/test_previous_pose_only_proposes_current_reference_pixels.py', lambda: _near_reference_material_invariant() == (),
             'Previous geometry can propose a bounded fresh reference fit; current correspondence and reverse-patch checks must qualify again and missing material remains unavailable.'),
            ('play_records_keep_published_source_geometry', 'core/perception/recording_the_frames_and_inputs.py',
             'tests/test_recording_the_actual_play_ports.py', lambda: _source_reference_record_invariant() == (),
             'Opt-in control records copy bounded published reference geometry and its original capture; missing channels stay unmeasured and geometry supplies no selected body or task outcome.'),
            ('classical_probes_refuse_known_model_backend_imports', 'core/runtime/classical_compute_boundary.py',
             'tests/test_classical_compute_probe_refuses_inprocess_backends.py', lambda: _classical_import_invariant() == (),
             'A scoped classical Python probe refuses known model-backend imports and preloaded backends; this interpreter boundary complements worker admission and supplies no claim about native code or arbitrary child interpreters.'),
            ('guarded_source_graph_preserves_memory_and_control', 'core/perception/guarded_effects_in_flash_methods.py',
             'tests/test_guarded_source_effects_do_not_enumerate_paths.py', lambda: _guarded_source_graph_invariant() == (),
             'A bounded acyclic source graph shares branch values and retains ordered opaque memory effects; complete instruction coverage cannot certify an executable controller.'),
            ('source_boolean_arithmetic_keeps_type_boundaries', 'core/perception/symbolic_motion_in_flash_methods.py',
             'tests/test_symbolic_motion_in_flash_methods.py', lambda: _source_boolean_arithmetic_invariant() == (),
             'Known boolean values reduce to zero or one for numeric source operations; strict equality retains type distinctions and unknown input remains symbolic.'),
            ('source_deletions_remain_uncomposed_mutations', 'core/perception/symbolic_motion_in_flash_methods.py',
             'tests/test_symbolic_motion_in_flash_methods.py', lambda: _source_deletion_invariant() == (),
             'Property deletion preserves its reaching context and unknown stack result, invalidates cached property values and remains an uncomposed mutation; scope deletion and dynamic members stay unresolved.'),
            ('shared_scene_cost_matches_direct_pixels', 'core/perception/a_visual_reference_in_the_picture.py',
             'tests/test_reference_search_reuses_current_scene_transforms.py', lambda: _scene_correlation_invariant() == (),
             'One capture shares its Fourier transforms across reference kernels while preserving the direct masked squared-colour pixel objective; no previous capture supplies the scene.'),
            ('reduced_reference_matches_keep_pixel_centres', 'core/perception/a_visual_reference_in_the_picture.py',
             'tests/test_a_reference_keeps_coordinates_while_it_turns.py', lambda: _reduced_reference_centres_invariant() == (),
             'Reference matching composes the reduced scene sample-centre offset into original-image geometry; raster footprint checks cannot be replaced by agreement between two identically biased matches.'),
            ('point_retention_cannot_restore_hidden_marks', 'core/perception/points_a_consumer_still_needs.py',
             'tests/test_active_material_survives_capacity_allocation.py', lambda: _point_retention_visibility_invariant() == (),
             'Bounded expiring consumer requests can retain only current visually measured point identities within one third of the point budget; missing material cannot be restored by a retention request.'),
            ('point_grids_keep_their_actual_sampling', 'core/perception/the_pixel_grid_a_mark_uses.py',
             'tests/test_named_marks_keep_their_actual_sampling_grid.py', lambda: _sample_grid_invariant() == (),
             'A point stream maps its actual sampled and working dimensions back to source-image sample centres, including per-axis rounding and raster phase.'),
            ('named_material_keeps_its_consumed_capture', 'core/perception/named_material_in_a_consumed_frame.py',
             'tests/test_source_geometry_precedes_object_grouping.py', lambda: _named_material_capture_invariant() == (),
             'A bounded visible point stream retains its consumed RGB capture, sampling scale, epoch and sequence; stale or duplicate point identities cannot become current material.'),
            ('reference_geometry_does_not_create_object_identity', 'core/perception/a_source_point_on_visible_marks.py',
             'tests/test_source_geometry_precedes_object_grouping.py', lambda: _visible_reference_identity_invariant() == (),
             'A visible source point contains measured geometry without an object number, kind, controlled identity or task role.'),
            ('reference_catalog_requires_the_current_source', 'core/perception/visual_references_for_current_sources.py',
             'tests/test_registered_references_need_current_source_bytes.py', lambda: _current_reference_source_invariant() == (),
             'Registered source artwork is acquired only for a current complete source declaration and matching pixel identity; a source change or failed current read cannot keep historical references active.'),
            ('reference_workers_retire_publication', 'core/perception/source_references_while_play_advances.py',
             'tests/test_reference_work_reaches_current_ordinary_observation.py', lambda: _reference_worker_retirement_invariant() == (),
             'An optional reference worker closes publication when retired; its result grants no controlled identity, task role or gameplay success.'),
            ('detached_material_keeps_its_original_observer', 'core/perception/a_material_frame_kept_for_its_owner.py',
             'tests/test_source_points_follow_current_owned_material.py', lambda: _material_frame_owner_invariant() == (),
             'Detached reference work retains copied consumed pixels, current masks and the original observer instance receipts; a delayed binding reaches live geometry only through the same continuing measured material basis.'),
            ('observers_keep_their_consumed_pixel_receipt', 'core/perception/pixels_consumed_by_the_observer.py',
             'tests/test_material_attachment_uses_the_consumed_pixels.py', lambda: _consumed_pixel_invariant() == (),
             'A play observer retains the original consumed RGB geometry, capture time and byte digest across worker publication; unsupported arrays supply no pixel custody and later caller mutations cannot rewrite a consumed receipt.'),
            ('source_points_need_their_current_material_basis', 'core/perception/a_source_point_on_current_material.py',
             'tests/test_source_points_follow_current_owned_material.py', lambda: _source_point_basis_invariant() == (),
             'A calibrated source point attaches through unique current visible ownership and a measured continuing point basis; changed captures, source, observer, track or basis cannot carry it, and it grants no task role or control identity.'),
            ('rotating_references_keep_their_source_samples', 'core/perception/a_visual_reference_in_the_picture.py',
             'tests/test_a_reference_keeps_coordinates_while_it_turns.py', lambda: _oriented_reference_invariant() == (),
             'A bounded orientation search returns redundant matched samples in original reference coordinates; failed orientation proposals cannot suppress a current fit, while different visible copies remain competing appearances without control or task authority.'),
            ('visual_names_keep_the_queried_instance', 'core/perception/names_for_observed_instances.py',
             'tests/test_visual_names_stay_with_the_boxed_instance.py', lambda: _instance_name_invariant() == (),
             'A delayed semantic answer retains its observer epoch, tracked birth, visual kind and captured pixel identity; another instance of the same kind cannot inherit that name or receive control authority from it.'),
            ('source_reference_points_need_measured_raster_alignment', 'core/perception/coordinates_in_a_visual_reference.py',
             'tests/test_source_coordinates_survive_reference_export.py', lambda: _source_image_coordinates_invariant() == (),
             'Declared source coordinates transfer into reference-image pixels only after redundant independent raster alignment; changed source or image bytes cannot carry the projected point, which supplies no input authority or collision claim.'),
            ('current_programs_need_current_acquired_bytes', 'core/skills/sovereign_browser_the_program.py',
             'tests/test_a_saved_program_summary_cannot_replace_current_bytes.py', lambda: _current_program_bytes_invariant() == (),
             'A current program read requires bounded acquired bytes; saved URL summaries cannot bypass current source and visual-declaration extraction or supply a current read when acquisition fails.'),
            ('delivered_material_costs_require_the_current_capture', 'core/perception/presses_on_the_same_material.py',
             'tests/test_delivered_presses_follow_material_instead_of_point_labels.py', lambda: _material_press_invariant() == (),
             'A delivered press can reduce the exploration bonus of a uniquely corresponding anonymous local point in a current measured reading; it proves no object identity, accepted action, scene identity or progress.'),
            ('cover_press_keeps_its_actual_receiver', 'core/skills/sovereign_browser_drawing.py',
             'tests/test_a_cover_press_checks_its_actual_receiver.py', lambda: _cover_press_invariant() == (),
             'Startup retries distinguish a trusted physical click on the intended DOM control from a click received by another overlay; unmeasured delivery retains the finite startup allowance.'),
            ('normal_navigation_records_its_physical_port', 'core/perception/recording_the_frames_and_inputs.py',
             'tests/test_cursor_witness_reaches_the_physical_click.py', lambda: _navigation_port_recording_invariant() == (),
             'Opted-in normal page navigation preserves the actual port coordinates and delivery outcome without entering motor replay or supplying a task-completion claim.'),
            ("software_graphics_need_a_measured_cpu_renderer", "core/capabilities/browser_graphics_policy.py",
             "tests/test_browser_software_graphics_stays_software.py", lambda: _software_graphics_invariant() == (),
             "An explicit software browser request selects the CPU driver and requires the browser process's renderer report; missing measurements cannot certify that policy."),
            ("cpu_vision_requires_request_acknowledgement", "core/perception/vision_request_compute.py",
             "tests/test_vision_requests_keep_their_cpu_policy.py", lambda: _cpu_vision_request_invariant() == (),
             "An explicitly requested CPU policy reaches a Vision request before its handler executes; missing or refused acknowledgement cannot fall back to the platform's device choice."),
            ("model_worker_policy_refuses_before_start", "core/runtime/model_worker_policy.py",
             "tests/test_a_cpu_probe_cannot_start_a_model_worker.py", lambda: _model_worker_policy_invariant() == (),
             "An explicit process policy can refuse model-worker admission before spawn; the vision client and worker entry honor that refusal before loading accelerator code."),
            ('recorded_pixels_keep_their_bytes', 'core/perception/recording_the_frames_and_inputs.py',
             'tests/test_recording_the_actual_play_ports.py', lambda: _recorded_pixel_invariant() == (),
             'Opted-in pixel recording preserves raw channel bytes through a lossless image; storage gaps and input transport receipts do not qualify gameplay.'),
            ('normal_screen_options_keep_their_reading', 'core/perception/recording_the_frames_and_inputs.py',
             'tests/test_recording_the_actual_play_ports.py', lambda: _screen_options_invariant() == (),
             'Opted-in recording preserves the normal pursuit reading, offered controls and original reader snapshot without relabelling it as a motor capture or inferring its channel order.'),
            ('surviving_marks_need_target_precision', 'core/perception/position_from_the_same_points.py',
             'tests/test_surviving_marks_locate_the_retained_reference.py', lambda: _surviving_reference_invariant() == (),
             'Surviving original marks can retain a coordinate origin through a rigid fit with a finite target precision budget; new names, collinearity and an exhausted budget cannot bridge it.'),
            ('material_parts_need_one_current_owner', 'core/perception/a_material_observation_from_its_parts.py',
             'tests/test_material_pieces_keep_one_observed_body.py', lambda: _material_parts_invariant() == (),
             'Visible foreground pieces can supply one material observation through current named rigid support; competing owners and valid rival outlines refuse their union, and unseen gaps remain unfilled.'),
            ('notice_controls_keep_javascript_word_boundaries', 'core/skills/sovereign_browser_drawing.py',
             'tests/test_only_an_exposed_notice_close_is_pressed.py', lambda: _notice_word_boundaries_invariant() == (),
             'The embedded notice-close expression preserves JavaScript word boundaries; physical close coordinates require a current hit on the labelled element or its descendant.'),
            ('affine_material_requires_redundant_geometry', 'core/perception/affine_motion_from_named_points.py',
             'tests/test_affine_material_is_not_a_rigid_body_turn.py', lambda: _affine_material_invariant() == (),
             'Limited affine material deformation requires redundant named marks, leave-one-out prediction and a stable inverse; it supplies no rigid angular phase or control authority.'),
            ('fresh_marks_cannot_outvote_retained_material', 'core/perception/position_from_the_same_points.py',
             'tests/test_fresh_marks_do_not_invalidate_their_own_receipt.py', lambda: _fresh_mark_capacity_invariant() == (),
             'The point provider and retained pose share a fresh-mark budget that keeps admitted existing material at least two thirds of each next set, within 256 marks.'),
            ('control_samples_have_distinct_times', 'core/agency/which_one_answers_to_her.py',
             'tests/test_control_samples_do_not_share_mutable_history.py', lambda: _control_sample_custody_invariant() == (),
             'Timed control measurements reject duplicate and decreasing capture times; selecting a body copies its trial samples into an independently owned history.'),
            ('trial_priority_cannot_certify_a_response', 'core/agency/trials_with_room_to_observe.py',
             'tests/test_trials_retain_room_for_their_observation.py', lambda: _trial_priority_invariant() == (),
             'Provisional pulse and rest measurements rank finite trials by projected clearance, without certifying a response, selecting an identity or increasing the per-control allowance.'),
            ('new_phases_keep_their_observation_reservation', 'core/agency/time_to_observe_a_control.py',
             'tests/test_control_measurements_reserve_the_observed_slow_frames.py', lambda: _phase_reservation_invariant() == (),
             'A newly delivered counter phase keeps the current observation reservation; delivered input deadlines and the finite overall experiment end remain unchanged.'),
            ('trial_acquisition_requires_owned_samples', 'core/agency/enough_observations_for_a_trial.py',
             'tests/test_a_trial_waits_for_its_actual_observations.py', lambda: _trial_observation_goal_invariant() == (),
             'An identified inertial body can end a finite calibration phase after distinct settled observations meet its estimator requirement; missing evidence never renews the hard input deadline or its attempt budget.'),
            ('inertial_control_axes_use_displacement_resolution', 'core/agency/which_one_answers_to_her.py',
             'tests/test_inertial_controls_keep_their_resolved_axes.py', lambda: _inertial_axis_resolution_invariant() == (),
             'Established inertial control axes require measured pulse and neutral responses whose predicted displacement exceeds tracking resolution; discovery thresholds remain unchanged.'),
            *_task_grounding_claims(), *_reward_authority_claims())


def _task_grounding_claims() -> tuple:
    from core.agency.a_requested_relation_in_play import _requested_destination_invariant
    from core.agency.playing_a_well import _provisional_well_invariant
    from core.agency.reading_while_playing import _current_directions_invariant
    from core.agency.the_controls_a_game_names import _navigation_motion_invariant
    from core.language.how_a_game_ended import _damaged_repeat_invariant
    from core.perception.a_cursor_target_in_its_pixels import _pointer_feedback_invariant
    from core.perception.a_visual_reference_in_the_picture import _reference_appearance_invariant
    from core.perception.a_well_seen import _well_geometry_invariant
    from core.perception.recording_the_frames_and_inputs import _pointer_recording_invariant
    from core.perception.symbolic_motion_in_flash_methods import _source_effect_prefix_invariant
    from core.perception.the_latest_reading_at_handoff import _latest_reading_invariant
    from core.perception.visual_assets_in_a_flash_file import _visual_asset_custody_invariant
    from core.perception.where_am_i import _pointer_target_frame_invariant
    from core.perception.where_the_pointer_turns import (
        _corner_queries_invariant,
        _partial_cursor_invariant,
    )
    from core.skills.screen_pursuit_bearings import (
        _animated_navigation_invariant,
        _cursor_anchor_invariant,
    )
    from core.skills.screen_pursuit_on_a_page import _cursor_cache_custody_invariant
    from core.skills.sovereign_browser_drawing import _making_handoff_invariant

    return (('pointer_comparisons_keep_their_reference_pixels', 'core/perception/recording_the_frames_and_inputs.py',
             'tests/test_a_pointer_target_is_reached_through_pixels.py', lambda: _pointer_recording_invariant() == (),
             'Opted-in pointer comparisons retain separate RGB reference and current pictures with their relationship, including failed matches; these records supply no task result or motor-frame identity.'),
            ('unseen_well_geometry_is_not_a_piece', 'core/perception/a_well_seen.py',
             'tests/test_a_changing_picture_cannot_hold_well_control.py', lambda: _well_geometry_invariant() == (),
             'Invalid or absent well crop geometry supplies no piece; saved nonfinite coordinates, fractional cell counts and invalid color bytes cannot enter the retained grid.'),
            ('provisional_well_motion_cannot_supply_a_played_result', 'core/agency/playing_a_well.py',
             'tests/test_a_changing_picture_cannot_hold_well_control.py', lambda: _provisional_well_invariant() == (),
             'A candidate grid has a bounded current-control admission period across every continuation branch; unconfirmed motion supplies no played-piece or clear count, and recalled key names require current responses.'),
            ('pointer_feedback_needs_current_distinct_material', 'core/perception/a_cursor_target_in_its_pixels.py',
             'tests/test_a_pointer_target_is_reached_through_pixels.py', lambda: _pointer_feedback_invariant() == (),
             'A physically sampled cursor target is followed through fresh pixel correspondence; continuing motion under a held pointer cannot train a static inverse, and ambiguous, absent or unreachable material cannot authorize a corrected press.'),
            ('handoff_words_belong_to_the_latest_reading', 'core/perception/the_latest_reading_at_handoff.py',
             'tests/test_handoff_retains_the_latest_reading.py', lambda: _latest_reading_invariant() == (),
             'Wordless or unavailable observations retire current text while retaining separately labelled historical words; an older capture cannot replace a newer handoff observation.'),
            ('current_pointer_targets_do_not_depend_on_a_grid', 'core/perception/where_am_i.py',
             'tests/test_current_pointer_target_survives_an_unrelated_grid.py', lambda: _pointer_target_frame_invariant() == (),
             'A unique pointer-responsive target in a current scoped pixel capture can admit its click despite an unrelated remembered grid; missing or ambiguous targets and keyboard moves retain the prior frame gate.'),
            ('view_navigation_does_not_name_a_following_body', 'core/agency/the_controls_a_game_names.py',
             'tests/test_pointer_navigation_does_not_invent_a_body.py', lambda: _navigation_motion_invariant() == (),
             'An explicit route between rooms or software views supplies no following-body evidence; separate direct body movement, aiming, firing and keyboard clauses remain available.'),
            ('interrupted_cursor_sweeps_keep_only_physical_evidence', 'core/perception/where_the_pointer_turns.py',
             'tests/test_cursor_inspection_returns_its_finished_samples.py', lambda: _partial_cursor_invariant() == (),
             'A bounded cursor sweep retains actual finished physical hit samples only with a local negative and unchanged page geometry; an incomplete pass cannot supply a census or suppress external cancellation.'),
            ('picture_corners_propose_cursor_queries', 'core/perception/where_the_pointer_turns.py',
             'tests/test_cursor_witness_reaches_the_physical_click.py', lambda: _corner_queries_invariant() == (),
             'Current captured corners provide bounded coordinate proposals; the sparse cursor path requires local current hand evidence with a negative control and retains the existing sweep when that evidence is missing.'),
            ('cursor_cache_keeps_pixels_and_coordinate_frames', 'core/skills/screen_pursuit_on_a_page.py',
             'tests/test_cursor_witness_reaches_the_physical_click.py', lambda: _cursor_cache_custody_invariant() == (),
             'Cursor sweep reuse requires the same immutable picture and source geometry; changed pixels, absent pictures and input attempts cannot retain old negative or cross-crop affordance evidence.'),
            ('named_clicks_keep_their_cursor_candidate', 'core/skills/screen_pursuit_bearings.py',
             'tests/test_cursor_witness_reaches_the_physical_click.py', lambda: _cursor_anchor_invariant() == (),
             'A uniquely associated hover caption retains its cursor candidate in normal click geometry; ambiguous or invalid anchors refuse a named click, and the page revalidates cursor affordance before delivery.'),
            ('making_deadlines_cannot_supply_a_creation_handoff', 'core/skills/sovereign_browser_drawing.py',
             'tests/test_making_timeout_does_not_make_an_artifact.py', lambda: _making_handoff_invariant() == (),
             'A browser task deadline or making classification cannot replace the current run creation handoff before reporting a made result or setting completion.'),
            ('known_navigation_survives_pixel_animation', 'core/skills/screen_pursuit_bearings.py',
             'tests/test_animated_navigation_still_offers_its_control.py', lambda: _animated_navigation_invariant() == (),
             'A current whole navigation label remains an input candidate when its pixels animate; changing scene prose and readouts remain excluded.'),
            ('source_effects_keep_their_reaching_prefix', 'core/perception/symbolic_motion_in_flash_methods.py',
             'tests/test_symbolic_motion_in_flash_methods.py', lambda: _source_effect_prefix_invariant() == (),
             'Each supported bytecode call or write retains its own reaching guards, numeric assumptions and supplied input context; later branches cannot retroactively guard an earlier effect.'),
            ('visual_asset_names_keep_source_dictionary_custody', 'core/perception/visual_assets_in_a_flash_file.py',
             'tests/test_visual_assets_keep_their_program_and_declaration.py', lambda: _visual_asset_custody_invariant() == (),
             'Exported visual names retain their program digest, dictionary character and declaration addresses; incomplete or conflicting metadata supplies no unique asset, observed object or task role.'),
            ('current_directions_follow_capture_order', 'core/agency/reading_while_playing.py',
             'tests/test_new_motion_directions_reach_the_running_actor.py', lambda: _current_directions_invariant() == (),
             'New explicit spatial directions reach the running meeting state in capture order, even when their wording was heard earlier; counters and stale or invalid capture times cannot replace them.'),
            ('reference_appearance_is_not_temporal_identity', 'core/perception/a_visual_reference_in_the_picture.py',
             'tests/test_a_supplied_reference_needs_current_matching_material.py', lambda: _reference_appearance_invariant() == (),
             'Supplied visual references require current bidirectional patch correspondence and redundant geometry; static asset appearance supplies neither a task role nor a controlled identity, and repeated instances remain competing matches.'),
            ('requested_destinations_require_current_unique_referents', 'core/agency/a_requested_relation_in_play.py',
             'tests/test_requested_relations_reach_the_actual_target.py', lambda: _requested_destination_invariant() == (),
             'An explicit spatial request binds a unique currently observed named destination; ambiguous names, stale extents and narration-only role names cannot select a replacement contact target.'),
            ('damaged_repeat_labels_supply_no_winner', 'core/language/how_a_game_ended.py',
             'tests/test_a_repeat_label_read_badly_hands_back_the_screen.py', lambda: _damaged_repeat_invariant() == (),
             'A bounded OCR comparison can return a newly repeated end screen to navigation without declaring a winner; current turns and conditional instructions still refuse termination.'))


def _reward_authority_claims() -> tuple:
    from core.perception.how_full_a_bar_is import _bar_discovery_budget_invariant
    from core.agency.reading_while_playing import _unowned_bar_reward_invariant
    from core.agency.what_meeting_things_does import _unlabelled_counter_reference_invariant
    from core.agency.what_she_has_left import _bar_role_invariant

    return (("bar_discovery_keeps_time_for_the_next_picture", "core/perception/how_full_a_bar_is.py",
             "tests/test_bar_discovery_keeps_time_for_the_next_picture.py", lambda: _bar_discovery_budget_invariant() == (),
             "Bar discovery visits at most eight sampled rows per picture; delayed candidates need current strip measurements, capture changes retire pending work, and established bars remain sampled each picture."),
            ("unowned_strips_cannot_teach_reward", "core/agency/reading_while_playing.py",
             "tests/test_unowned_strips_do_not_teach_rewards.py", lambda: _unowned_bar_reward_invariant() == (),
             "Before a body is selected or a response is measured, nameless changing strips remain measurements and cannot supply rewards or resource scarcity."),
            ("unlabelled_counters_require_a_reference", "core/agency/what_meeting_things_does.py",
             "tests/test_unowned_strips_do_not_teach_rewards.py", lambda: _unlabelled_counter_reference_invariant() == (),
             "An unlabelled numeric change remains measured without supplying reward until an own-side reference relates the two counters."),
            ("measured_strips_require_a_known_role", "core/agency/what_she_has_left.py",
             "tests/test_unowned_strips_do_not_teach_rewards.py", lambda: _bar_role_invariant() == (),
             "Measured strip changes require a known resource, reward or side role before teaching reward or scarcity; selecting a controlled body supplies no meter semantics."))


def install_realtime_control_claims(suite: Any) -> None:
    from core.agency.a_body_seen_then import _body_seen_then_invariant
    from core.agency.a_response_worth_measuring_again import _provisional_response_invariant
    from core.agency.body_trials_keep_the_world import _body_trial_phase_invariant
    from core.agency.causal_identification import (
        _stale_and_ambiguous_identity_witnesses_remain_unknown,
    )
    from core.agency.controls_measured_on_one_thing import (
        _own_control_witness_invariant,
        _resolved_motion_invariant,
        _trajectory_control_invariant,
    )
    from core.agency.controls_ready_for_the_motion_model import _model_readiness_invariant
    from core.agency.countering_motion_as_it_appears import _counter_trial_measurement_invariant
    from core.agency.deliberate_action import _unmeasured_response_invariant
    from core.agency.facing_seen_under_its_input import _facing_input_invariant
    from core.agency.giving_a_trial_time import _control_trial_time_invariant
    from core.agency.how_a_game_works import _known_body_controls_invariant
    from core.agency.keeping_a_trial_observable import _observed_viability_invariant
    from core.agency.using_a_control_on_a_thing import _object_use_target_invariant
    from core.agency.what_meeting_things_does import _absent_counters_are_not_current
    from core.agency.where_clicks_pay import _click_attribution_respects_stretch_boundaries
    from core.agency.where_things_lead import _unknown_selector_invariant
    from core.agency.which_one_answers_to_her import (
        _ambiguous_controls_remain_unknown,
        _emissions_require_distinct_input_receipts,
        _fresh_control_experiment_forgets_old_rejections,
    )
    from core.agency.which_way_she_faces import _turn_observation_invariant
    from core.agency.words_read_with_their_picture import _word_picture_start_invariant
    from core.cognition.action_refinement import _bound_refinement_invariant
    from core.cognition.guidance_lifetime import _completed_guidance_invariant
    from core.cognition.procedure_binding import _choice_binding_invariant
    from core.cognition.reading_the_rules import _procedure_completion_is_measured
    from core.organism.model_validation import (
        Claim,
        Evidence,
        Observation,
        ValidationTest,
        boolean_score,
    )
    from core.perception.a_small_thing_in_a_joined_patch import _small_joined_thing_invariant
    from core.perception.drawn_objects import _object_candidates_invariant
    from core.perception.eyes_of_their_own import _child_capture_custody_invariant
    from core.perception.foreground_appearance import _foreground_mask_invariant
    from core.perception.observed_transfer import _transfer_receipts_require_evidence
    from core.perception.parts_when_things_touch import _joined_parts_invariant
    from core.perception.relations_between_things import _relations_require_unique_finite_entities
    from core.perception.the_drawing_as_objects import _scene_requires_a_matching_complete_frame
    from core.perception.visual_anchor import _remembered_labels_invariant
    from core.perception.visual_probe_cache import _visual_probe_miss_invariant
    from core.perception.watching_it_happen import _response_window_keeps_its_evidence
    from core.perception.where_the_words_point import _named_places_invariant
    from core.perception.word_labels_seen_again import _visible_word_labels_invariant
    from core.perception.words_kept_with_their_pixels import _kept_word_pixels_invariant
    from core.runtime.action_executor import _terminal_effect_flags_invariant
    from core.runtime.executors import _interactive_and_receipt_workers_are_separate
    from core.runtime.what_she_learned import (
        _indexed_knowledge_invariant,
        _retained_structure_invariant,
    )
    from core.runtime.what_stops_it import _stop_listener_lifetime_invariant
    from core.skills.current_words_before_play import _current_play_words_invariant, _menu_preparation_invariant
    from core.skills.screen_step_evidence import (
        _construction_readiness_invariant,
        _creation_completion_invariant,
        _response_dispatch_invariant,
    )
    from core.skills.sovereign_browser_drawing import (
        _current_play_veto_invariant,
        _stalled_making_invariant,
    )
    from core.skills.sovereign_browser_picking import _picked_outcomes_invariant

    for name, owner, fixture, probe, statement in (
        *_play_execution_claims(),
        *_visual_motion_claims(),
        *_body_state_claims(),
        *_control_discovery_claims(),
        *_bounded_pixel_claims(),
        *_travel_turn_claims(),
        ("pixel_control_requires_resolved_displacement", "core/agency/controls_measured_on_one_thing.py",
         "tests/test_which_thing_answers_to_her.py", lambda: _resolved_motion_invariant() == (),
         "Pixel-derived control evidence requires observed displacement beyond the tracker resolution before assigning a new body identity."),
        ("entity_control_cannot_borrow_kind_evidence", "core/agency/controls_measured_on_one_thing.py",
         "tests/test_a_kind_cannot_lend_a_controls_identity.py", lambda: _own_control_witness_invariant() == (),
         "A kind's control prior cannot replace an entity's missing input trials or measured exclusions."),
        ("returning_paths_keep_their_entity_trials", "core/agency/controls_measured_on_one_thing.py",
         "tests/test_entity_press_trajectories.py", lambda: _trajectory_control_invariant() == (),
         "Returning trajectories require repeated contrasting input receipts on the same observed entity and experiment."),
        ("controls_are_ready_for_the_motion_model", "core/agency/controls_ready_for_the_motion_model.py",
         "tests/test_control_readiness_matches_the_motion_model.py", lambda: _model_readiness_invariant() == (),
         "A control remains unknown until the active motion model has measured its response."),
        ("control_trials_preserve_observation_time", "core/agency/keeping_a_trial_observable.py",
         "tests/test_a_control_trial_preserves_its_view.py", lambda: _observed_viability_invariant() == (),
         "Measured acceleration can interrupt an unknown control trial when waiting and braking would leave the observed field."),
        ("object_use_requires_positive_target_evidence", "core/agency/using_a_control_on_a_thing.py",
         "tests/test_a_control_is_used_on_what_it_is_for.py", lambda: _object_use_target_invariant() == (),
         "Unknown action text cannot authorize using a key on an unnamed object without a stated target or measured effect."),
        ("remembered_labels_keep_current_word_pixels", "core/perception/word_labels_seen_again.py",
         "tests/test_visible_word_labels.py", lambda: _visible_word_labels_invariant() == (),
         "A retained label requires a unique fresh match to its measured word pixels within the same view and bounded lifetime."),
        ("provisional_responses_do_not_establish_identity", "core/agency/a_response_worth_measuring_again.py",
         "tests/test_a_cold_control_loop_preserves_time_to_measure.py", lambda: _provisional_response_invariant() == (),
         "A fresh pulse and neutral interval can select a bounded independent repeat without granting body identity; missing or competing measurements abstain."),
        ("visual_avatar_guess_keeps_its_observed_track", "core/agency/a_body_seen_then.py",
         "tests/test_a_body_guess_keeps_its_observed_track.py", lambda: _body_seen_then_invariant() == (),
         "A visual body guess follows its captured observed track; absence or changed appearance retires it and causal identity takes precedence."),
        ("an_empty_reading_still_measures_the_initial_screen", "core/agency/words_read_with_their_picture.py",
         "tests/test_an_empty_reading_still_saw_the_world.py", lambda: _word_picture_start_invariant() == (),
         "A completed empty word reading establishes the initial picture; pending or unavailable readings remain unmeasured."),
        ("counter_trials_wait_for_measured_neutral_motion", "core/agency/countering_motion_as_it_appears.py",
         "tests/test_counter_trials_wait_for_measured_motion.py", lambda: _counter_trial_measurement_invariant() == (),
         "Initial counter trials wait for fresh settled unheld motion, preserve ordinary exploration and terminate within their finite input and observation budget."),
        ("joined_foreground_keeps_a_distinct_small_thing", "core/perception/a_small_thing_in_a_joined_patch.py",
         "tests/test_a_small_thing_survives_a_joined_foreground.py", lambda: _small_joined_thing_invariant() == (),
         "Joined foreground is separated before an area rejection; floor correction leaves visibly distinct minority pixels outside the backdrop."),
        ("body_trials_preserve_their_interaction_phase", "core/agency/body_trials_keep_the_world.py",
         "tests/test_body_trials_leave_interface_controls_available.py", lambda: _body_trial_phase_invariant() == (),
         "Known body trials exclude explicit resets and text-entry controls while preserving unknown inputs and interface availability."),
        ("kept_words_need_their_captured_pixels", "core/perception/words_kept_with_their_pixels.py",
         "tests/test_old_menu_words_keep_their_pixels.py", lambda: _kept_word_pixels_invariant() == (),
         "Retained word readings can be contradicted by measured appearance changes; absent pixels and changed view geometry stay unknown."),
        ("turns_require_settled_observations", "core/agency/which_way_she_faces.py",
         "tests/test_a_turn_is_not_learned_from_unseen_motion.py", lambda: _turn_observation_invariant() == (),
         "Stationary turning samples require settled observations; missing pictures cannot establish that a press moved nothing."),
        ("facing_keeps_observed_motion_and_input_custody", "core/agency/facing_seen_under_its_input.py",
         "tests/test_facing_uses_observed_motion_and_its_input.py", lambda: _facing_input_invariant() == (),
         "Facing measurements require current observed motion under the input delivered before the picture; predictions and overlapping inputs teach no response."),
        ("classical_preparation_requires_a_captured_menu", "core/skills/current_words_before_play.py",
         "tests/test_classical_observation_is_prepared_on_a_menu.py", lambda: _menu_preparation_invariant() == (),
         "Empty classical observation preparation requires a captured current menu in a known running world, with no picture consumption, input or body evidence."),
        ("current_play_words_require_a_measured_frame", "core/skills/current_words_before_play.py",
         "tests/test_current_words_precede_play_interpretation.py", lambda: _current_play_words_invariant() == (),
         "Fast play handoff reads use current measured words; empty, old and future frames cannot remove a menu."),
        ("control_trials_keep_their_measurement_time", "core/agency/giving_a_trial_time.py",
         "tests/test_a_control_trial_gets_time_to_answer.py", lambda: _control_trial_time_invariant() == (),
         "A control trial retains its finite measurement interval while input and observation identity remain unchanged."),
        ("joined_parts_keep_measured_pixels", "core/perception/parts_when_things_touch.py",
         "tests/test_touching_things_keep_their_parts.py", lambda: _joined_parts_invariant() == (),
         "Touching foreground parts retain measured pixels and separate only where recent appearance and motion distinguish them."),
        ("known_body_controls_set_trial_priority", "core/agency/how_a_game_works.py",
         "tests/test_known_body_controls_set_trial_priority.py", lambda: _known_body_controls_invariant() == (),
         "Known body inputs set control trial priority before incidental interface input guesses; unknown roles preserve uncertainty."),
        ("foreground_masks_require_measured_support", "core/perception/foreground_appearance.py",
         "tests/test_transfer_identity_survives_background_changes.py", lambda: _foreground_mask_invariant() == (),
         "Appearance masks exclude only a measured flat exterior while retaining bounded foreground support and enclosed highlights."),
        ("creation_completion_requires_current_work", "core/skills/screen_step_evidence.py",
         "tests/test_creation_completion_needs_current_work.py", lambda: _creation_completion_invariant() == (),
         "Construction-based making completion requires independently reobserved placed work; navigation and historical activity cannot establish it."),
        ("visual_probe_misses_require_current_pixels", "core/perception/visual_probe_cache.py",
         "tests/test_visual_probe_misses_require_current_pixels.py", lambda: _visual_probe_miss_invariant() == (),
         "A failed visual probe suppresses repeated work only for matching measured pixels and interpretation inputs within its lifetime."),
        ("stalled_making_is_not_completion", "core/skills/sovereign_browser_drawing.py",
         "tests/test_making_timeout_does_not_make_an_artifact.py", lambda: _stalled_making_invariant() == (),
         "An unchanged making task's timeout ends its attempt without establishing a produced artifact."),
        ("receipts_keep_terminal_effect_flags", "core/runtime/action_executor.py",
         "tests/test_external_completion_keeps_its_terminal_outcome.py", lambda: _terminal_effect_flags_invariant() == (),
         "Post-action receipt flags preserve the durable transaction's effect and reconciliation outcome."),
        ("relations_require_unique_finite_entities", "core/perception/relations_between_things.py",
         "tests/test_spatial_relations_require_unique_current_entities.py", lambda: _relations_require_unique_finite_entities() == (),
         "Spatial completion evidence abstains when object names are ambiguous or their geometry is unmeasured."),
        ("picked_objectives_require_each_outcome", "core/skills/sovereign_browser_picking.py",
         "tests/test_picked_objectives_require_each_outcome.py", lambda: _picked_outcomes_invariant() == (),
         "A collection's completion requires each requested objective rather than each pursuit merely ending."),
        ("current_play_vetoes_predicted_completion", "core/skills/sovereign_browser_drawing.py",
         "tests/test_play_completion_requires_terminal_evidence.py", lambda: _current_play_veto_invariant() == (),
         "Current turn and ongoing-play evidence vetoes an inferred win or loss."),
        ("refined_obligations_require_their_bound_transfer", "core/cognition/action_refinement.py",
         "tests/test_source_choice_and_placement_share_one_owned_action.py", lambda: _bound_refinement_invariant() == (),
         "A combined choice and placement requires the exact source-use transfer bound before delivery."),
        ("unknown_selector_views_do_not_create_navigation_identity", "core/agency/where_things_lead.py",
         "tests/test_selector_views_keep_visual_navigation_states.py", lambda: _unknown_selector_invariant() == (),
         "Unmeasured selector contents cannot create a navigation state or an edge between states."),
        ("response_streams_keep_owned_capture_windows", "core/perception/watching_it_happen.py",
         "tests/test_action_response_windows.py", _response_window_keeps_its_evidence,
         "Response streams retain delivery intervals and capture timestamps, excluding prior owned changes from background."),
        ("responses_belong_to_one_delivered_dispatch", "core/skills/screen_step_evidence.py",
         "tests/test_visual_responses_keep_their_own_dispatch.py", lambda: _response_dispatch_invariant() == (),
         "Visual responses belong to exactly one delivered input and a fresh observation of the bound surface."),
        ("unmeasured_responses_are_not_graded", "core/agency/deliberate_action.py",
         "tests/test_visual_responses_keep_their_own_dispatch.py", lambda: _unmeasured_response_invariant() == (),
         "Unmeasured responses cannot resolve action episodes or teach success and failure consequences."),
        ("object_candidates_require_measured_extents", "core/perception/drawn_objects.py",
         "tests/test_drawn_objects_have_measured_extents.py", lambda: _object_candidates_invariant() == (),
         "Anonymous object candidates have current measured extents and appearance witnesses, without assumed affordances."),
        ("completed_guidance_cannot_publish", "core/cognition/guidance_lifetime.py",
         "tests/test_guidance_lifetime.py", lambda: _completed_guidance_invariant() == (),
         "Retired guidance operations reject delayed publication while preserving their parent execution owner."),
        ("remembered_labels_require_current_pixels", "core/perception/visual_anchor.py",
         "tests/test_visual_anchor.py", lambda: _remembered_labels_invariant() == (),
         "Remembered control labels require fresh current appearance in the same surface and capture context."),
        ("choices_require_current_observed_controls", "core/cognition/procedure_binding.py",
         "tests/test_procedure_bindings_follow_observed_controls.py", lambda: _choice_binding_invariant() == (),
         "Choice bindings require an observed selector effect and currently visible choices in its capture context."),
        ("construction_controls_require_current_work", "core/skills/screen_step_evidence.py",
         "tests/test_procedure_bindings_follow_observed_controls.py", lambda: _construction_readiness_invariant() == (),
         "Construction execution controls cannot use narrative or historical drag records as current readiness."),
        ("stop_listeners_have_scoped_lifetimes", "core/runtime/what_stops_it.py",
         "tests/test_stopped_execution_cannot_keep_acting.py", lambda: _stop_listener_lifetime_invariant() == (),
         "Cancellation listeners detach without stopping sibling work and retain the parent's stop direction."),
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

    name = "retained_structures_reject_damaged_closures"
    owner = "core/runtime/what_she_learned.py"
    suite.add_test(ValidationTest(
        name=name, description="damaged declared positional structures exclude their closure and preserve independent knowledge",
        required_capability="", observation=Observation("damaged_structures_excluded", True,
                "tests/test_retained_positional_structures_keep_their_shape.py"),
        predict=lambda _model: _retained_structure_invariant() == (),
        score=lambda value, observation, subject=name: boolean_score(value, expected=observation.value, subject=subject),
        owner=owner,
    ))
    suite.add_claim(Claim(
        statement="Declared retained positional structures are validated before reuse, excluding damaged closures while preserving independent knowledge.",
        test=name, owner=owner, asserted_in=owner, evidence=Evidence.MEASURED_SYNTHETIC,
        evidence_note="Fixed records, vectors, schema evolution and bounded roundtrips measured in independent synthetic domains; live use requires separate evidence.",
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


def _body_state_claims() -> tuple:
    from core.agency.acting_on_a_prepared_motion import _prepared_state_invariant
    from core.agency.facing_seen_under_its_input import _heading_lineage_invariant

    return (
        ("a_continuing_body_keeps_heading_when_its_look_changes", "core/agency/facing_seen_under_its_input.py",
         "tests/test_a_continuing_body_keeps_its_heading.py", lambda: _heading_lineage_invariant() == (),
         "A current track with the same birth and experiment retains heading through visual reclassification; a replacement clears the earlier measurements."),
        ("prepared_motion_requires_its_current_qualified_state", "core/agency/acting_on_a_prepared_motion.py",
         "tests/test_prepared_motion_keeps_its_input_set.py", lambda: _prepared_state_invariant() == (),
         "Prepared motion requires fresh measured state with the same body and model context; a pixel backdrop is not a state measurement."),
    )


def _travel_turn_claims() -> tuple:
    from core.agency.a_model_for_steering import (
        _forecast_window_invariant,
        _provisional_steering_invariant,
    )
    from core.agency.a_trial_has_a_physical_end import _physical_trial_deadline_invariant
    from core.agency.calibration_for_a_current_body import _current_body_calibration_invariant
    from core.agency.direction_linked_to_observed_pose import (
        _pose_direction_invariant,
    )
    from core.agency.facing_seen_under_its_input import _fresh_heading_invariant
    from core.agency.response_in_the_body_frame import _body_heading_invariant
    from core.agency.steering_from_a_model import (
        _provisional_steering_input_invariant,
        _source_turn_role_invariant,
    )
    from core.agency.turning_while_travelling import (
        _travel_segments_invariant,
        _travel_turns_invariant,
    )
    from core.agency.which_way_she_faces import _past_travel_invariant
    from core.perception.orientation_from_continuing_marks import _angular_phase_invariant
    from core.perception.texture_that_continues_when_outlines_join import (
        _rotating_texture_invariant,
    )
    from core.runtime.task_ownership import _unstarted_task_cleanup_invariant

    return (
        ("observed_travel_is_not_erased_by_a_later_missing_picture", "core/agency/which_way_she_faces.py",
         "tests/test_observed_travel_survives_a_missing_picture.py", lambda: _past_travel_invariant() == (),
         "One delivered press retains its earlier observed motion segment through later missing pictures; gaps do not combine samples or establish stationary turns."),
        ("rotating_texture_requires_measured_rigid_pose", "core/perception/texture_that_continues_when_outlines_join.py",
         "tests/test_a_rotating_texture_keeps_its_visible_support.py", lambda: _rotating_texture_invariant() == (),
         "A rotated texture proposal uses a current rigid pose, continuing owned marks and matching current pixels; translation alone cannot supply an unmeasured rotation."),
        ("angular_phase_requires_current_material_marks", "core/perception/orientation_from_continuing_marks.py",
         "tests/test_direction_can_follow_current_material_marks.py", lambda: _angular_phase_invariant() == (),
         "A current angular phase requires spread proper-rigid correspondence on the same visible point pose; stale, replaced or reflected geometry remains unknown."),
        ("pose_direction_requires_observed_travel_agreement", "core/agency/direction_linked_to_observed_pose.py",
         "tests/test_a_pose_path_pairs_direction_at_the_measured_time.py", lambda: _pose_direction_invariant() == (),
         "A pose phase supplies a facing only after spread direction samples agree within measured uncertainty; changed ownership or pose basis requires new qualification."),
        ("source_turn_roles_only_arm_a_body_hypothesis", "core/agency/steering_from_a_model.py",
         "tests/test_steering_hypotheses_reach_the_actor.py", lambda: _source_turn_role_invariant() == (),
         "A resolved source rotation method on the controlled class can arm a provisional turn; unrelated calls cannot confer a turn role or measured control."),
        ("a_forecast_ends_at_its_actual_input_window", "core/agency/a_model_for_steering.py",
         "tests/test_a_forecast_keeps_its_actual_observation_window.py", lambda: _forecast_window_invariant() == (),
         "A provisional forecast retains only measured pictures inside its actual delivered input interval; a physical release preserves partial coverage without treating a later picture as evidence of the old press."),
        ("a_control_trial_has_a_physical_deadline", "core/agency/a_trial_has_a_physical_end.py",
         "tests/test_a_control_trial_ends_without_another_picture.py", lambda: _physical_trial_deadline_invariant() == (),
         "A discovery or facing press has a receipt-owned physical deadline; its release does not wait for another picture, cannot release a different press, and preserves a current discovery trial's planned neutral observation phase."),
        ("completed_task_closes_its_unstarted_coroutine", "core/runtime/task_ownership.py",
         "tests/test_a_cancelled_wrapper_closes_unstarted_work.py", lambda: _unstarted_task_cleanup_invariant() == (),
         "A completed tracked task closes its still-unstarted native coroutine, including cancellation before its wrapper starts; running work keeps its cancellation and completion semantics."),
        ("provisional_steering_keeps_its_observed_owner", "core/agency/a_model_for_steering.py",
         "tests/test_a_steering_model_can_be_used_before_calibration.py", lambda: _provisional_steering_invariant() == (),
         "A structural steering model can propose finite physical predictions from current owned motion; its assumed signs and rates remain provisional, and changed birth or epoch invalidates its parameters and forecast."),
        ("provisional_steering_keeps_input_ownership", "core/agency/steering_from_a_model.py",
         "tests/test_steering_hypotheses_reach_the_actor.py", lambda: _provisional_steering_input_invariant() == (),
         "Provisional physical forecasts can deliver bounded goal-directed inputs before complete steering calibration, without granting body identity or measured control; ownership changes and contradictions end their input."),
        ("calibration_belongs_to_the_current_owned_body", "core/agency/calibration_for_a_current_body.py",
         "tests/test_a_current_body_has_its_own_calibration_budget.py", lambda: _current_body_calibration_invariant() == (),
         "A current matched body receives its own finite calibration programme; missing ownership cannot renew a budget, and issued requests do not establish measured control."),
        ("travel_segments_keep_one_input_trial", "core/agency/turning_while_travelling.py",
         "tests/test_a_travel_press_can_be_observed_again.py", lambda: _travel_segments_invariant() == (),
         "Missing motion readings split observed arcs without merging samples or multiplying input trials; contradictory qualified arcs retire their press and require new independent repeats."),
        ("a_new_heading_requires_a_current_direction_experiment", "core/agency/facing_seen_under_its_input.py",
         "tests/test_a_new_body_requests_a_fresh_heading.py", lambda: _fresh_heading_invariant() == (),
         "Earlier owned responses can request bounded direction trials after a body is reacquired; they grant no heading or turning measurements, and unavailable controls cannot be requested."),
        ("travel_turns_require_repeated_observed_input_contrasts", "core/agency/turning_while_travelling.py",
         "tests/test_turning_while_travelling.py", lambda: _travel_turns_invariant() == (),
         "Turning during travel requires two coherent measured input intervals and two neutral contrasts; its rate cannot establish turning at rest."),
        ("heading_response_belongs_to_its_body_and_experiment", "core/agency/response_in_the_body_frame.py",
         "tests/test_body_heading_measurements_keep_their_owner.py", lambda: _body_heading_invariant() == (),
         "A heading response belongs to its current body and identification experiment; predicting a travel tangent requires fresh observed travel."),
    )


def _bounded_pixel_claims() -> tuple:
    from core.agency.an_observed_scene_transition import _observed_transition_invariant
    from core.agency.collecting_independent_pixel_trials import (
        _pixel_sampling_time_invariant,
        _settled_pixel_capture_invariant,
    )
    from core.agency.discovering_control_in_current_pixels import _pixel_discovery_invariant
    from core.agency.navigation_before_body_experiments import (
        _current_navigation_invariant,
        _current_round_end_invariant,
        _navigation_occurrence_invariant,
    )
    from core.agency.pixels_inside_control_trials import _pixel_trial_custody_invariant
    from core.agency.probes_from_source_arithmetic import _conditional_source_probe_invariant
    from core.agency.searching_before_a_body_is_known import _pixel_search_phase_invariant
    from core.perception.a_region_that_answers_independent_inputs import _pixel_cohort_invariant
    from core.perception.a_search_hint_for_these_pixels import _pixel_search_hint_invariant
    from core.perception.boundary_support_for_translation import _bounded_edge_samples_invariant
    from core.perception.flash_actions_with_source import _action_address_invariant
    from core.perception.frames_as_they_are_drawn import _canvas_capture_bounds_invariant
    from core.perception.learned_tracking.continuity_backend import _continuity_invariant
    from core.perception.learned_tracking.points_for_the_motion_eyes import (
        _query_lifetime_invariant,
    )
    from core.perception.motion_inside_source_methods import _source_motion_conditions_invariant
    from core.perception.pixel_motion_before_object_identity import (
        _pixel_motion_bound_invariant,
        _translation_layer_invariant,
    )
    from core.perception.points_before_the_picture_is_reduced import (
        _pixel_search_capacity_invariant,
        _sampling_hint_receipt_invariant,
    )
    from core.perception.rotation_inside_a_matched_pixel_patch import (
        _patch_rotation_uncertainty_invariant,
    )
    from core.perception.streamed_pixels_keep_their_surface import _stream_surface_invariant
    from core.runtime.optional_compute import _optional_headroom_invariant

    return (
        ("conditional_source_arithmetic_only_orders_body_probes", "core/agency/probes_from_source_arithmetic.py",
         "tests/test_source_arithmetic_reaches_body_probes.py", lambda: _conditional_source_probe_invariant() == (),
         "Conditional source vector increments can order already authorized body probes without supplying physical units, signed controls or a visible identity."),
        ("flash_action_branches_keep_exact_byte_boundaries", "core/perception/flash_actions_with_source.py",
         "tests/test_flash_actions_keep_their_control_flow.py", lambda: _action_address_invariant() == (),
         "Flash action source records retain exact branch destinations and refuse branches into instruction payloads."),
        ("source_motion_keeps_unresolved_branch_conditions", "core/perception/motion_inside_source_methods.py",
         "tests/test_motion_inside_source_methods.py", lambda: _source_motion_conditions_invariant() == (),
         "Source arithmetic keeps unresolved branch conditions separate from unconditional physical models."),
        ("experiment_observations_preserve_scene_transitions", "core/agency/an_observed_scene_transition.py",
         "tests/test_pixel_search_observes_its_physical_trials.py", lambda: _observed_transition_invariant() == (),
         "A screen transition observed during an input experiment retires the partial cohort, releases input and clears the prior scene's spatial plans."),
        ("pixel_sampling_keeps_finite_selected_capture_times", "core/agency/collecting_independent_pixel_trials.py",
         "tests/test_collecting_independent_pixel_trials.py", lambda: _pixel_sampling_time_invariant() == (),
         "An isolated press can supply at most five selected capture times to its existing observation path; early feedback retires the incomplete pixel cohort and releases its input."),
        ("a_later_menu_occurrence_needs_measured_retirement", "core/agency/navigation_before_body_experiments.py",
         "tests/test_current_navigation_precedes_body_experiments.py", lambda: _navigation_occurrence_invariant() == (),
         "Two fresh captures of changed word pixels can retire a menu occurrence and renew a later identical control; unchanged colour, duplicate captures and missing readings cannot."),
        ("current_round_ends_prevent_body_experiments", "core/agency/navigation_before_body_experiments.py",
         "tests/test_a_round_end_cannot_supply_a_new_body.py", lambda: _current_round_end_invariant() == (),
         "Current asserted terminal word pixels return play even at its start; menu budgets, existing bodies and unchanged readouts cannot turn that screen into another body experiment."),
        ("local_translation_needs_its_own_moving_layer", "core/perception/pixel_motion_before_object_identity.py",
         "tests/test_pixel_motion_before_object_identity.py", lambda: _translation_layer_invariant() == (),
         "A local translation proposal can separate coherent moving matches from still background only with current inlier support inside the reported tile; competing layers remain unavailable."),
        ("pixel_exposure_keeps_its_selected_capture_start", "core/agency/collecting_independent_pixel_trials.py",
         "tests/test_collecting_independent_pixel_trials.py", lambda: _settled_pixel_capture_invariant() == (),
         "Paired pixel trials require captures after their selected neutral and settled input intervals; late arrival of an earlier capture cannot supply that exposure."),
        ("current_navigation_precedes_body_experiments", "core/agency/navigation_before_body_experiments.py",
         "tests/test_current_navigation_precedes_body_experiments.py", lambda: _current_navigation_invariant() == (),
         "Current owned word pixels can return an unassigned body task to navigation before control experiments; old words, changed surfaces and unavailable captures cannot."),
        ("patch_rotation_uncertainty_needs_rgb_gradient_rank", "core/perception/rotation_inside_a_matched_pixel_patch.py",
         "tests/test_rotation_inside_a_matched_pixel_patch.py", lambda: _patch_rotation_uncertainty_invariant() == (),
         "Local RGB patch rotation keeps fit uncertainty separate from rigid-body pose; flat or rank-deficient patches cannot resolve its angle."),
        ("streamed_pixels_keep_session_document_and_crop", "core/perception/streamed_pixels_keep_their_surface.py",
         "tests/test_streamed_pixels_keep_their_surface.py", lambda: _stream_surface_invariant() == (),
         "Accepted streamed frames retain their actual session, document and crop with source capture bounds; missing clocks and geometry cannot supply causal capture custody."),
        ("sampling_hint_receipt_comes_from_kept_points", "core/perception/points_before_the_picture_is_reduced.py",
         "tests/test_pixel_search_hints_reach_owned_sampling.py", lambda: _sampling_hint_receipt_invariant() == (),
         "A sampling receipt retains exact capture custody and counts points after actual allocation; skipped or refused sampling supplies no consumed receipt."),
        ("pixel_search_follows_completed_unresolved_body_trials", "core/agency/searching_before_a_body_is_known.py",
         "tests/test_pixel_search_follows_unresolved_body_trials.py", lambda: _pixel_search_phase_invariant() == (),
         "A bounded pixel search follows a completed unresolved body experiment with current body controls; known bodies and other action phases keep ordinary play."),
        ("pixel_search_hints_belong_to_the_exact_capture", "core/perception/a_search_hint_for_these_pixels.py",
         "tests/test_pixel_search_hints_reach_owned_sampling.py", lambda: _pixel_search_hint_invariant() == (),
         "A sampling search hint requires the exact RGB capture and its owner through thread and child transport; changed pixels or receipts refuse priority."),
        ("pixel_search_preserves_other_point_capacity", "core/perception/points_before_the_picture_is_reduced.py",
         "tests/test_pixel_search_hints_reach_owned_sampling.py", lambda: _pixel_search_capacity_invariant() == (),
         "A validated region reserves part of existing point capacity while preserving quiet references; newly sampled corners supply no prior motion or identity."),
        ("pixel_discovery_requires_its_complete_admitted_experiment", "core/agency/discovering_control_in_current_pixels.py",
         "tests/test_discovering_control_in_current_pixels.py", lambda: _pixel_discovery_invariant() == (),
         "Pixel response discovery requires every planned admitted press and its observed motion; missing or changed references retire the experiment without assigning a body."),
        ("pixel_trials_keep_prespecified_input_and_capture", "core/agency/pixels_inside_control_trials.py",
         "tests/test_pixels_inside_control_trials.py", lambda: _pixel_trial_custody_invariant() == (),
         "Independent visual trials retain randomized cohorts, immutable captured pixels and acknowledged isolated input windows; replayed captures and changed exposures are refused."),
        ("pixel_response_uses_distinct_confirmation_cohorts", "core/perception/a_region_that_answers_independent_inputs.py",
         "tests/test_a_region_answers_independent_inputs.py", lambda: _pixel_cohort_invariant() == (),
         "Visual response regions require separate prespecified confirmation trials and measured feature support; they supply search priority without player identity."),
        ("canvas_capture_bounds_keep_their_surface", "core/perception/frames_as_they_are_drawn.py",
         "tests/test_canvas_frames_keep_their_capture_bounds.py", lambda: _canvas_capture_bounds_invariant() == (),
         "A canvas capture retains its read interval, surface generation and page sequence through decoding; missing custody metadata leaves capture time unknown."),
        ("pixel_motion_proposals_have_bounded_measured_support", "core/perception/pixel_motion_before_object_identity.py",
         "tests/test_pixel_motion_before_object_identity.py", lambda: _pixel_motion_bound_invariant() == (),
         "Bounded RGB correspondences retain current forward and reverse patch support before object discovery; absent scene references remain unavailable."),
        ('queries_keep_their_first_observation_and_live_budget',
         'core/perception/learned_tracking/points_for_the_motion_eyes.py',
         'tests/test_original_continuity_points.py', lambda: _query_lifetime_invariant() == (),
         'Point queries receive a first observation before replacement; replenishment uses the actual live query count.'),
        ('continuity_points_need_fresh_unique_pixels', 'core/perception/learned_tracking/continuity_backend.py',
         'tests/test_original_continuity_points.py', lambda: _continuity_invariant() == (),
         'Original point tracking retains bounded appearance memory; absent or competing current patches supply no visible motion.'),
        ('optional_compute_preserves_unobserved_headroom', 'core/runtime/optional_compute.py',
         'tests/test_tracking_models_share_host_capacity.py', lambda: _optional_headroom_invariant() == (),
         'Optional numerical work defers when host headroom is unknown and cannot queue behind an occupied process permit.'),
        ("bounded_edge_samples_keep_their_observed_median", "core/perception/boundary_support_for_translation.py",
         "tests/test_bounded_edge_samples_keep_their_pixels.py", lambda: _bounded_edge_samples_invariant() == (),
         "Exterior edge samples retain the same observed colour medians through support holes; missing samples remain unavailable."),
    )


def _control_discovery_claims() -> tuple:
    from core.agency.a_source_motion_model_checked_in_view import _source_motion_agreement_invariant
    from core.agency.an_effect_contact_cannot_avoid import _nonlocal_countermeasure_invariant
    from core.agency.body_trials_keep_the_world import _control_source_invariant
    from core.agency.how_a_game_works import (
        _cold_translation_controls_invariant,
        _motion_and_emission_trials_invariant,
    )
    from core.agency.inputs_while_measuring_control import (
        _body_probe_phase_invariant,
        _body_trial_rest_invariant,
    )
    from core.agency.reflexes_from_measured_consequences import (
        _reflex_consequence_invariant,
        _reflex_effect_invariant,
    )
    from core.agency.unknown_controls_get_their_turn import _unknown_control_fairness_invariant
    from core.agency.control_trials_keep_their_delivered_counts import _delivered_probe_count_invariant
    from core.agency.what_acknowledged_inputs_predict import _acknowledged_prediction_invariant
    from core.perception.a_world_as_its_code_makes_it import _saved_world_effects_invariant

    return (
        ("discovery_strategies_share_delivered_probe_counts", "core/agency/control_trials_keep_their_delivered_counts.py",
         "tests/test_control_discovery_keeps_its_delivered_probes.py", lambda: _delivered_probe_count_invariant() == (),
         "Unknown-body discovery retains current physically delivered probe counts across strategies for ordering only; counts grant no measured response or identity."),
        ("unknown_controls_share_a_finite_current_body_budget", "core/agency/unknown_controls_get_their_turn.py",
         "tests/test_unknown_controls_get_their_turn.py", lambda: _unknown_control_fairness_invariant() == (),
         "Unknown controls share bounded issued trials for the current body; old discovery counts cannot spend a new owner's allowance, and reordered inputs or unavailable outcomes cannot renew it."),
        ("cold_translation_trials_keep_current_source_channels", "core/agency/how_a_game_works.py",
         "tests/test_pixel_search_follows_unresolved_body_trials.py", lambda: _cold_translation_controls_invariant() == (),
         "Cold pixel experiments prioritize current nonzero translation channels before turning inputs; key names and obsolete controls cannot supply that priority."),
        ("acknowledged_inputs_cover_the_response_interval", "core/agency/what_acknowledged_inputs_predict.py",
         "tests/test_acknowledged_inputs_cover_visual_latency.py", lambda: _acknowledged_prediction_invariant() == (),
         "Direct-response predictions integrate acknowledged control intervals with bounded state channels; missing model intervals preserve uncertainty."),
        ("reflex_motion_feedback_needs_its_owned_response", "core/agency/reflexes_from_measured_consequences.py",
         "tests/test_reflexes_act_before_slower_planning.py", lambda: _reflex_effect_invariant() == (),
         "Observed response checks belong to an acknowledged input and its current model context; contradictory motion suspends that response even when risk falls."),
        ("fresh_consequences_can_interrupt_slower_planning", "core/agency/reflexes_from_measured_consequences.py",
         "tests/test_reflexes_act_before_slower_planning.py", lambda: _reflex_consequence_invariant() == (),
         "Short responses require fresh improving consequence estimates, acknowledged inputs and later observed outcomes; absent observations remain unknown."),
        ("nonlocal_cost_keeps_a_known_countermeasure_available", "core/agency/an_effect_contact_cannot_avoid.py",
         "tests/test_an_unavoidable_cost_is_not_hidden_by_an_appearance.py",
         lambda: _nonlocal_countermeasure_invariant() == (),
         "An effect that avoiding contact cannot prevent retains a known countermeasure; missing actions and measured beneficial meetings preserve uncertainty."),
        ("source_motion_needs_current_observed_agreement", "core/agency/a_source_motion_model_checked_in_view.py",
         "tests/test_a_source_motion_model_is_checked_in_current_view.py",
         lambda: _source_motion_agreement_invariant() == (),
         "Complete source-bound direct motion can guide actions after current observed response agrees; missing, contradictory, carried and active-trial states retain discovery."),
        ("saved_world_readings_preserve_action_effects", "core/perception/a_world_as_its_code_makes_it.py",
         "tests/test_body_motion_trials_keep_emission_controls_separate.py",
         lambda: _saved_world_effects_invariant() == (),
         "Saving and loading a program reading preserves its source-bound emissions and effects at an edge."),
        ("current_controls_keep_their_sources", "core/agency/body_trials_keep_the_world.py",
         "tests/test_body_motion_trials_keep_emission_controls_separate.py",
         lambda: _control_source_invariant() == (),
         "Current local control bindings take precedence over remembered names and controls parsed from unqualified prose."),
        ("body_motion_trials_keep_emission_controls_separate", "core/agency/how_a_game_works.py",
         "tests/test_body_motion_trials_keep_emission_controls_separate.py",
         lambda: _motion_and_emission_trials_invariant() == (),
         "Source-bound emissions retain their action channel while body trials measure movement; combined effects and unknown controls remain eligible."),
        ("control_discovery_keeps_its_finite_observation_phase", "core/agency/inputs_while_measuring_control.py",
         "tests/test_control_discovery_has_its_own_finite_phase.py", lambda: _body_probe_phase_invariant() == (),
         "Control discovery reserves two isolated press and rest cycles and each delivered trial's finite observation interval before ordinary favored actions."),
        ("an_observed_body_trial_finishes_with_its_rest", "core/agency/inputs_while_measuring_control.py",
         "tests/test_identifying_a_body_does_not_skip_the_trial_rest.py", lambda: _body_trial_rest_invariant() == (),
         "An identified body still completes the delivered control trial and its finite rest before ordinary planning takes over."),
    )


def _visual_motion_claims() -> tuple:
    from core.perception.a_texture_copy_with_public_state import _effective_texture_copy_invariant
    """Measured foreground and point continuity claims."""
    from core.agency.a_body_still_expected_when_it_is_unseen import _expected_control_invariant
    from core.agency.a_model_for_steering import (
        _slow_motion_invariant,
        _velocity_uncertainty_invariant,
    )
    from core.agency.acting_on_a_prepared_motion import _prepared_input_set_invariant
    from core.agency.an_input_answers_relative_to_motion import (
        _motion_coordinates_invariant,
        _travel_speed_invariant,
    )
    from core.agency.arriving_by_her_pushes import _arrival_braking_invariant
    from core.agency.controls_that_take_time_to_build import _stored_response_invariant
    from core.agency.direction_linked_to_observed_pose import _pose_coverage_invariant
    from core.agency.finding_her_another_way import _current_body_motion_invariant
    from core.agency.following_a_prepared_path import _prepared_path_invariant
    from core.agency.how_a_game_works import _parsed_body_trial_invariant, _picture_checks_invariant
    from core.agency.inputs_while_measuring_control import _single_input_interval_invariant
    from core.agency.room_to_finish_a_required_motion import _preparation_room_invariant
    from core.agency.solving_for_a_required_state import _required_state_invariant
    from core.agency.time_from_picture_to_control import _picture_age_invariant
    from core.agency.time_to_observe_a_control import _observation_time_invariant
    from core.agency.what_a_press_does import _settled_trial_invariant, _unchanged_input_invariant
    from core.agency.what_follows_the_pointer import _pointer_body_continuity_invariant
    from core.agency.which_one_answers_to_her import (
        _measured_prefix_invariant,
        _motion_segments_invariant,
    )
    from core.cognition.what_its_code_says import _parsed_control_priority_invariant
    from core.perception.a_board_seen import _lattice_phase_invariant
    from core.perception.a_browser_picture_has_a_deadline import _picture_request_budget_invariant
    from core.perception.a_clock_for_drawn_frames import _drawing_clock_invariant
    from core.perception.a_picture_examined_without_holding_the_loop import (
        _picture_probe_inputs_invariant,
    )
    from core.perception.a_reading_before_more_detail import _finished_reading_invariant
    from core.perception.a_reference_kept_by_visible_support import _reference_bridge_invariant, _resting_material_invariant
    from core.perception.a_texture_copied_by_its_shader import _texture_copy_invariant
    from core.perception.activity_in_the_picture import _picture_activity_invariant
    from core.perception.background_visible_beside_a_patch import _exposed_background_invariant
    from core.perception.boundary_support_for_translation import _visible_boundary_invariant
    from core.perception.continuing_material_before_outlines import _material_identity_invariant
    from core.perception.frames_as_they_are_drawn import (
        _frame_session_invariant,
        _frame_time_invariant,
        _lossless_frame_invariant,
    )
    from core.perception.graphics_presented_on_canvas import _presentation_custody_invariant
    from core.perception.learned_tracking.contracts import _tracking_visibility_invariant
    from core.perception.learned_tracking.corrections_while_points_keep_moving import (
        _capture_batch_invariant,
        _visible_correction_invariant,
        _visible_mark_retention_invariant,
    )
    from core.perception.learned_tracking.points_for_the_motion_eyes import (
        _learned_point_capture_invariant,
    )
    from core.perception.motion_kept_between_pictures import (
        _motion_belief_invariant,
        _motion_proposal_invariant,
        _shared_motion_arithmetic_invariant,
    )
    from core.perception.play_observation_without_holding_the_loop import (
        _owned_play_observation_invariant,
    )
    from core.perception.points_before_the_picture_is_reduced import (
        _independent_patch_fit_invariant,
        _source_point_geometry_invariant,
        _source_point_precision_invariant,
    )
    from core.perception.points_shared_with_changing_regions import _point_capacity_invariant
    from core.perception.points_that_keep_their_motion import (
        _owned_material_motion_invariant,
        _point_history_invariant,
        _point_identity_invariant,
    )
    from core.perception.position_from_the_same_points import (
        _point_handoff_invariant,
        _point_path_invariant,
    )
    from core.perception.reading_again_after_loss import _reading_renewal_invariant
    from core.perception.rigid_motion_from_named_points import _rigid_motion_invariant
    from core.perception.sampling_the_drawings_pace import _drawing_cadence_invariant
    from core.perception.shader_attributes_from_buffer_bytes import _attribute_addressing_invariant
    from core.perception.shader_position_math import _shader_position_invariant
    from core.perception.shader_uniforms_from_buffer_bytes import _uniform_resource_invariant
    from core.perception.texture_coordinates_to_a_surface import _texture_projection_invariant
    from core.perception.texture_that_continues_when_outlines_join import (
        _continuing_texture_invariant,
    )
    from core.perception.the_whole_patch_around_a_visible_corner import _whole_patch_invariant
    from core.perception.translation_from_visible_support import _translation_support_invariant
    from core.perception.triangles_from_public_draws import _triangle_assembly_invariant
    from core.perception.what_moves_in_the_picture import (
        _material_trail_invariant,
        _missing_points_invariant,
        _point_pose_birth_invariant,
    )
    from core.perception.what_the_points_on_a_thing_say import _point_meaning_invariant
    from core.skills.a_new_attempt_must_be_seen import _new_attempt_invariant

    return (
        ("retained_material_precedes_a_joined_outline_majority", "core/perception/points_that_keep_their_motion.py",
         "tests/test_owned_marks_survive_a_stationary_majority.py", lambda: _owned_material_motion_invariant() == (),
         "A continuing body's retained named marks establish its current motion before admitting coherent new marks; stationary texture in a larger joined outline cannot replace that measured motion."),
        ("browser_picture_requests_have_a_finite_budget", "core/perception/a_browser_picture_has_a_deadline.py",
         "tests/test_a_stalled_browser_cannot_own_the_picture_wait.py", lambda: _picture_request_budget_invariant() == (),
         "Stream startup, frame acknowledgements, resampling, direct canvas reads and screenshot requests have finite browser waits; an unanswered request supplies no fresh picture."),
        ("local_frame_encoding_keeps_colour_samples", "core/perception/frames_as_they_are_drawn.py",
         "tests/test_local_capture_preserves_colour_samples.py", lambda: _lossless_frame_invariant() == (),
         "Local stream and direct canvas capture use lossless encoding; fine colour samples survive decoding and cropping at the chosen capture extent."),
        ("shared_motion_arithmetic_keeps_its_owned_state", "core/perception/motion_kept_between_pictures.py",
         "tests/test_motion_prediction_reuses_its_arithmetic.py", lambda: _shared_motion_arithmetic_invariant() == (),
         "Motion estimates share bounded immutable time-step matrices while each object's state remains owned; diagonal mixture arithmetic retains the full covariance reference's reported estimates."),
        ("a_missing_outline_requires_continuous_observed_marks", "core/perception/points_that_keep_their_motion.py",
         "tests/test_point_paths_bridge_an_unmatched_outline.py", lambda: _point_history_invariant() == (),
         "A briefly missing object outline can use continuously observed named marks from its last owned pixels; broken, stale or replaced paths cannot bridge the gap."),
        ("texture_copy_uses_effective_public_draw_state", "core/perception/a_texture_copy_with_public_state.py",
         "tests/test_a_texture_copy_with_public_state.py", lambda: _effective_texture_copy_invariant() == (),
         "Texture-copy geometry requires the current linked shader, effective sampler, replacement raster state and exact attachment image generations; geometry grants no visibility or control identity."),
        ("source_marks_return_to_one_working_geometry", "core/perception/points_before_the_picture_is_reduced.py",
         "tests/test_source_points_keep_their_visible_detail.py", lambda: _source_point_geometry_invariant() == (),
         "Source-resolution marks retain their identities and observation times when mapped to working coordinates; changed geometry breaks their correspondence."),
        ("local_patch_fits_finish_independently", "core/perception/points_before_the_picture_is_reduced.py",
         "tests/test_patch_refinement_finishes_independently.py", lambda: _independent_patch_fit_invariant() == (),
         "Each local rigid patch fit reaches its own numerical stopping condition; batching unrelated patches preserves the isolated fit's result and final colour residual."),
        ("graphics_hints_require_current_presentation", "core/perception/graphics_presented_on_canvas.py",
         "tests/test_graphics_presented_on_canvas.py", lambda: _presentation_custody_invariant() == (),
         "Geometric search hints follow recorded image versions and ordered public copies to a measured current canvas; visibility and object identity remain unverified."),
        ("shader_positions_use_declared_operations", "core/perception/shader_position_math.py",
         "tests/test_shader_position_math.py", lambda: _shader_position_invariant() == (),
         "The admitted position subset follows declared arithmetic and simultaneous swizzle writes; missing and unsupported dependencies return unknown."),
        ("uniform_bytes_keep_resource_versions", "core/perception/shader_uniforms_from_buffer_bytes.py",
         "tests/test_shader_uniforms_from_buffer_bytes.py", lambda: _uniform_resource_invariant() == (),
         "Uniform fields use declared names, reflected layouts and matching captured buffer generations; missing or reused bytes remain unknown."),
        ("texture_copy_requires_unchanged_dataflow", "core/perception/a_texture_copied_by_its_shader.py",
         "tests/test_a_texture_copied_by_its_shader.py", lambda: _texture_copy_invariant() == (),
         "An admitted fragment copy requires one unchanged texture sample and declared smooth coordinates; altered color, coordinates and multiple outputs stay unknown."),
        ("vertex_bytes_keep_draw_addressing", "core/perception/shader_attributes_from_buffer_bytes.py",
         "tests/test_shader_attributes_from_buffer_bytes.py", lambda: _attribute_addressing_invariant() == (),
         "Vertex values retain current buffer versions, stride, offset, normalization and instance divisor; missing bytes and unsupported formats remain unknown."),
        ("draw_triangles_keep_assembly_and_versions", "core/perception/triangles_from_public_draws.py",
         "tests/test_triangles_from_public_draws.py", lambda: _triangle_assembly_invariant() == (),
         "Admitted triangle assembly preserves winding, first vertex and fixed restart within captured current element ranges; unsupported draws stay unknown."),
        ("texture_regions_use_measured_projective_coordinates", "core/perception/texture_coordinates_to_a_surface.py",
         "tests/test_texture_coordinates_to_a_surface.py", lambda: _texture_projection_invariant() == (),
         "Texture search regions follow measured smooth UV and homogeneous positions through clipping; unknown sampling and degenerate coordinates stay unknown."),
        ("point_resolution_keeps_tracked_grid_units", "core/perception/points_before_the_picture_is_reduced.py",
         "tests/test_point_precision_keeps_the_tracked_grids_units.py", lambda: _source_point_precision_invariant() == (),
         "Point coordinates and their resolution term use the same tracked grid mapping; unknown resolution retains the working-grid default and invalid resolution supplies no pose."),
        ("pose_calibration_keeps_measured_phase_coverage", "core/agency/direction_linked_to_observed_pose.py",
         "tests/test_direction_calibration_keeps_measured_phase_coverage.py", lambda: _pose_coverage_invariant() == (),
         "A bounded direction calibration retains measured phase coverage and the latest disagreement; repeated nearby observations cannot remove all excitation or hide a contradictory pair."),
        ("prepared_motion_releases_only_its_owned_input_set", "core/agency/acting_on_a_prepared_motion.py",
         "tests/test_prepared_motion_keeps_its_input_set.py", lambda: _prepared_input_set_invariant() == (),
         "A prepared model action owns its acknowledged key set and finite physical hold; interruption releases its own inputs without advancing an incomplete path."),
        ("parsed_body_controls_reach_finite_discovery", "core/agency/how_a_game_works.py",
         "tests/test_a_parsed_body_has_its_named_control_trials.py", lambda: _parsed_body_trial_invariant() == (),
         "A resolved current source method on the controlled class reaches finite input discovery when a caller omits its key; that experiment cannot replace measured role or body identity."),
        ("parsed_controls_keep_their_source_priority", "core/cognition/what_its_code_says.py",
         "tests/test_a_code_control_survives_lower_priority_readings.py", lambda: _parsed_control_priority_invariant() == (),
         "A control parsed from the current program retains its known role and source when a lower-priority reading names a conflicting action."),
        ("worker_observations_keep_their_private_state", "core/perception/play_observation_without_holding_the_loop.py",
         "tests/test_play_observation_keeps_input_deadlines_alive.py", lambda: _owned_play_observation_invariant() == (),
         "A pending picture leaves input deadlines runnable; worker-owned mutable state reaches the public view only through a detached completed snapshot."),
        ("rotated_control_trials_keep_their_own_motion", "core/agency/an_input_answers_relative_to_motion.py",
         "tests/test_an_input_answers_relative_to_the_bodys_motion.py", lambda: _motion_coordinates_invariant() == (),
         "Repeated isolated inputs distinguish speed change and signed curvature across headings; autonomous motion or absent neutral trials cannot supply the witness."),
        ("travel_keeps_its_speed_when_it_changes_direction", "core/agency/an_input_answers_relative_to_motion.py",
         "tests/test_travel_speed_can_answer_without_one_direction.py", lambda: _travel_speed_invariant() == (),
         "Repeated active and neutral trials compare measured travel speed before averaging directions; a closed curve does not become stationary."),
        ("picture_checks_follow_the_activity", "core/agency/how_a_game_works.py",
         "tests/test_picture_checks_follow_the_activity.py", lambda: _picture_checks_invariant() == (),
         "Known continuous movement reaches control before incompatible board and well discovery; shared keys and grids keep discovery unresolved."),
        ("lattice_phases_keep_their_observed_teeth", "core/perception/a_board_seen.py",
         "tests/test_lattice_period_can_be_measured_without_phase_loops.py", lambda: _lattice_phase_invariant() == (),
         "Lattice phase grouping preserves observed neighbor pixels, tooth counts, phase ties and the smallest nearly best spacing."),
        ("picture_probes_own_detached_inputs", "core/perception/a_picture_examined_without_holding_the_loop.py",
         "tests/test_optional_picture_probes_keep_the_loop_free.py", lambda: _picture_probe_inputs_invariant() == (),
         "Optional picture workers use copied arguments and detached answers; timeouts remain unmeasured and preserve ownership until actual completion."),
        ("missing_body_observations_keep_the_control_role", "core/agency/a_body_still_expected_when_it_is_unseen.py",
         "tests/test_missing_body_observations_keep_the_control_role.py", lambda: _expected_control_invariant() == (),
         "Missing visual identity preserves established body control; timing roles and measured payoffs retain their own evidence."),
        ("occluded_tracking_is_not_a_visible_measurement", "core/perception/learned_tracking/contracts.py",
         "tests/test_learned_tracking_capture_contract.py", lambda: _tracking_visibility_invariant() == (),
         "Tracking results keep capture identity and geometry; hidden coordinates and seeded masks cannot supply visible observations."),
        ("learned_points_keep_their_capture", "core/perception/learned_tracking/points_for_the_motion_eyes.py",
         "tests/test_learned_points_reach_the_motion_eyes.py", lambda: _learned_point_capture_invariant() == (),
         "Learned point measurements reach actual motion support in working coordinates; other captures and hidden predictions cannot extend a visible path."),
        ("late_points_need_visible_transport", "core/perception/learned_tracking/corrections_while_points_keep_moving.py",
         "tests/test_learned_corrections_do_not_hold_up_visible_tracking.py", lambda: _visible_correction_invariant() == (),
         "Late learned positions reach the current view only through retained actual captures and fresh correspondence; expired or reset history supplies no correction."),
        ("grouped_points_use_actual_capture_sequences", "core/perception/learned_tracking/corrections_while_points_keep_moving.py",
         "tests/test_grouped_points_receive_actual_capture_sequences.py", lambda: _capture_batch_invariant() == (),
         "Grouped point tracking receives bounded sequences of actual captures; missing history restarts queries and a late answer retains the capture it measured."),
        ("refinement_keeps_current_visible_marks", "core/perception/learned_tracking/corrections_while_points_keep_moving.py",
         "tests/test_grouped_points_receive_actual_capture_sequences.py", lambda: _visible_mark_retention_invariant() == (),
         "Refinement preserves continuing marks measured in the current capture, within a bounded point budget; another reference or duplicate location supplies no additional support."),
        ("changing_regions_receive_point_capacity", "core/perception/points_shared_with_changing_regions.py",
         "tests/test_changing_regions_receive_point_capacity.py", lambda: _point_capacity_invariant() == (),
         "Quiet reference marks leave bounded capacity for new changing regions; fresh corner admission supplies no motion until later pixels match."),
        ("an_ending_after_reset_needs_new_activity", "core/skills/a_new_attempt_must_be_seen.py",
         "tests/test_a_retry_restarts_the_page_and_observes_a_new_attempt.py", lambda: _new_attempt_invariant() == (),
         "A new attempt requires current activity after reset; an old capture or another reading of its ending supplies no completed run."),
        ("steering_preserves_measured_velocity_uncertainty", "core/agency/a_model_for_steering.py",
         "tests/test_steering_keeps_measured_velocity_uncertainty.py", lambda: _velocity_uncertainty_invariant() == (),
         "A current measured velocity keeps its uncertainty across the steering boundary; invalid uncertainty retires the current pose."),
        ("slow_motion_keeps_its_input_and_pixel_uncertainty", "core/agency/a_model_for_steering.py",
         "tests/test_slow_observed_motion_can_anchor_a_steering_model.py", lambda: _slow_motion_invariant() == (),
         "Slow motion accumulates displacement within one observed input receipt; unresolved pixel jitter cannot supply a driver, and chord uncertainty stays in provisional forecasts."),
        ("a_stream_frame_keeps_its_session_owner", "core/perception/frames_as_they_are_drawn.py",
         "tests/test_a_stream_callback_keeps_its_producing_session.py", lambda: _frame_session_invariant() == (),
         "A browser callback, calibration and decoded frame belong to their producing stream generation; an old session cannot publish or acknowledge through its replacement."),
        ("motion_proposals_keep_the_last_measured_box", "core/perception/motion_kept_between_pictures.py",
         "tests/test_measured_motion_proposes_visual_correspondence.py", lambda: _motion_proposal_invariant() == (),
         "Correspondence proposals retain the last measured box and uncertainty without refreshing observation or control identity."),
        ("motion_forecasts_do_not_renew_observations", "core/perception/motion_kept_between_pictures.py",
         "tests/test_motion_is_kept_between_measured_pictures.py", lambda: _motion_belief_invariant() == (),
         "Motion beliefs retain their measured identity and reference; forecasts grow uncertainty without renewing observations."),
        ("drawing_sampling_uses_capture_cadence", "core/perception/sampling_the_drawings_pace.py",
         "tests/test_drawing_sampling_follows_its_capture_pace.py", lambda: _drawing_cadence_invariant() == (),
         "Drawing streams reduce thinning only from consecutive measured capture intervals on the same source clock."),
        ("queued_frames_keep_the_drawing_clock", "core/perception/a_clock_for_drawn_frames.py",
         "tests/test_streamed_frames_use_their_drawing_clock.py", lambda: _drawing_clock_invariant() == (),
         "A bracketed browser clock maps drawing times into a fixed stream epoch; queued delivery preserves source cadence, repeated or invalid source timestamps cannot renew a frame, and capture bounds retain their provenance."),
        ("a_reference_bridge_needs_independent_visible_support", "core/perception/a_reference_kept_by_visible_support.py",
         "tests/test_a_measured_patch_can_keep_a_point_reference.py", lambda: _reference_bridge_invariant() == (),
         "A measured origin crosses missing marks only through current matching pixels and exterior edges, with the same birth and observation time; current owned marks join in that calibrated origin."),
        ("resting_material_keeps_a_measured_origin", "core/perception/a_reference_kept_by_visible_support.py",
         "tests/test_resting_material_keeps_its_measured_origin.py", lambda: _resting_material_invariant() == (),
         "A still-visible unmatched object may retain its measured origin through current agreeing points and matching pixels; absent or contradictory support supplies no new path."),
        ("rigid_point_motion_requires_spread_consensus", "core/perception/rigid_motion_from_named_points.py",
         "tests/test_rotating_points_keep_their_reference.py", lambda: _rigid_motion_invariant() == (),
         "A rigid point transform needs a bounded consensus of continuing noncollinear correspondences; a rotating path retains its measured reference rather than a changing box centre."),
        ("control_trials_reserve_their_observation_samples", "core/agency/time_to_observe_a_control.py",
         "tests/test_control_trials_use_the_observed_cadence.py", lambda: _observation_time_invariant() == (),
         "The first measured picture interval can reserve bounded observation time after input settlement; each delivered trial keeps its deadline and time alone grants no learned response."),
        ("joined_outlines_require_current_owned_texture", "core/perception/texture_that_continues_when_outlines_join.py",
         "tests/test_continuing_texture_can_separate_a_joined_outline.py", lambda: _continuing_texture_invariant() == (),
         "A joined patch can yield a translated texture proposal only with current spread-out owned landmarks and matching visible foreground; all unassigned pixels remain."),
        ("new_point_marks_require_a_measured_origin_bridge", "core/perception/position_from_the_same_points.py",
         "tests/test_new_marks_join_only_through_a_measured_bridge.py", lambda: _point_handoff_invariant() == (),
         "Three continuing spread-out owned marks can admit a minority of new marks into the same measured origin; a replacement without that bridge cannot extend the path."),
        ("exposed_background_keeps_enclosed_body_detail", "core/perception/background_visible_beside_a_patch.py",
         "tests/test_exposed_background_can_be_a_minority.py", lambda: _exposed_background_invariant() == (),
         "Quiet agreeing opposite-side pixels can identify exposed background inside a foreground patch; enclosed matching details and unsupported boundaries remain uncorrected."),
        ("prepared_paths_need_current_feedback_and_receipts", "core/agency/following_a_prepared_path.py",
         "tests/test_a_prepared_path_requires_current_feedback.py", lambda: _prepared_path_invariant() == (),
         "A prepared input path is replayed from fresh state under the same model identity; only its own timely completed input receipt advances its cursor."),
        ("motion_forecasts_keep_released_response", "core/agency/controls_that_take_time_to_build.py",
         "tests/test_room_to_finish_a_required_motion.py", lambda: _stored_response_invariant() == (),
         "Known channel equations retain build-up and decay; a fast affine forecast agrees with physical ticks without declaring intermediate contacts safe."),
        ("arrival_preparation_includes_the_time_to_stop", "core/agency/room_to_finish_a_required_motion.py",
         "tests/test_room_to_finish_a_required_motion.py", lambda: _preparation_room_invariant() == (),
         "A constant-rate braking projection retains coupled state changes; preparation includes stopping and a bounded-speed transfer, with unknown rates unavailable."),
        ("known_transition_plans_preserve_failure_boundaries", "core/agency/solving_for_a_required_state.py",
         "tests/test_solving_for_a_required_state.py", lambda: _required_state_invariant() == (),
         "Bounded inverse planning accepts explicit known mechanics and outcomes; a failed intermediate state cannot be repaired into predicted success later in the same path."),
        ("repeated_stream_frames_keep_their_original_time", "core/perception/frames_as_they_are_drawn.py",
         "tests/test_a_frame_keeps_its_capture_time.py", lambda: _frame_time_invariant() == (),
         "A consumed stream frame cannot become a fresh observation through silence; canvas and screenshot requests retain their start time through slow completion."),
        ("control_prediction_uses_the_current_picture_age", "core/agency/time_from_picture_to_control.py",
         "tests/test_control_uses_this_pictures_age.py", lambda: _picture_age_invariant() == (),
         "Prediction refreshes the current picture age before steering and keeps measured delivery and post-acknowledgement response separate."),
        ("finished_readings_survive_only_with_their_current_capture", "core/perception/a_reading_before_more_detail.py",
         "tests/test_a_finished_reading_survives_slow_detail.py", lambda: _finished_reading_invariant() == (),
         "Optional inspection timeouts can retain completed scoped readings only after a new capture agrees; incomplete, changed or refused captures remain unavailable."),
        ("arrivals_need_measured_braking_room", "core/agency/arriving_by_her_pushes.py",
         "tests/test_a_carried_body_arrives_before_it_stops.py", lambda: _arrival_braking_invariant() == (),
         "Carried arrival plans need measured braking pushes and finite goals; terminal closing speed is weighed against room before first contact."),
        ("point_paths_keep_their_original_visible_marks", "core/perception/position_from_the_same_points.py",
         "tests/test_a_point_path_keeps_its_original_marks.py", lambda: _point_path_invariant() == (),
         "A path measured from points retains its origin through continuing admitted marks, current owned pixels and coherent displacement; an unbridged replacement ends the path."),
        ("secondary_inputs_end_independent_body_observations", "core/agency/inputs_while_measuring_control.py",
         "tests/test_inputs_while_measuring_control.py", lambda: _single_input_interval_invariant() == (),
         "Another delivered action ends the isolated control interval; measured prefixes remain intact and later mixed motion cannot extend them."),
        ("pointer_control_needs_one_continuing_body", "core/agency/what_follows_the_pointer.py",
         "tests/test_pointer_control_belongs_to_one_body.py", lambda: _pointer_body_continuity_invariant() == (),
         "Pointer-control sightings keep the same entity and birth; separate objects, capture gaps and unmeasured movement cannot share a trail."),
        ("visible_activity_does_not_need_certified_body_velocity", "core/perception/activity_in_the_picture.py",
         "tests/test_visible_activity_is_not_body_velocity.py", lambda: _picture_activity_invariant() == (),
         "Visible activity prevents a still-screen handoff without supplying body translation or control authority; capture gaps cannot establish continuous quiet."),
        ("old_control_trials_need_current_body_motion", "core/agency/which_one_answers_to_her.py",
         "tests/test_entity_press_trajectories.py", lambda: _current_body_motion_invariant() == (),
         "Historical control receipts cannot assign a new body identity while its current pixel movement is unavailable, contradicted or stale."),
        ("control_trials_start_on_delivered_input", "core/agency/playing_as_it_happens.py",
         "tests/test_control_trials_start_when_input_is_delivered.py", lambda: _unchanged_input_invariant() == (),
         "Choosing an unchanged held input preserves the physical intervention start and cannot manufacture an independent control trial."),
        ("settled_behavior_keeps_the_independent_observation", "core/agency/what_a_press_does.py",
         "tests/test_translation_needs_visible_support.py", lambda: _settled_trial_invariant() == (),
         "Settled behavior retains measured trial coverage until the delivered input interval ends; lack of response is observed rather than inferred."),
        ("plain_patch_motion_requires_visible_exterior_boundaries", "core/perception/boundary_support_for_translation.py",
         "tests/test_visible_boundary_support.py", lambda: _visible_boundary_invariant() == (),
         "Plain-patch translation requires observed corresponding exterior boundaries; an interior window on a uniform surface remains unmeasured."),
        ("control_motion_requires_agreeing_current_support", "core/perception/translation_from_visible_support.py",
         "tests/test_translation_needs_visible_support.py", lambda: _translation_support_invariant() == (),
         "Detected box movement requires agreeing current visible support before it teaches control responses or physics."),
        ("control_measurements_resume_on_new_observed_segments", "core/agency/which_one_answers_to_her.py",
         "tests/test_translation_needs_visible_support.py", lambda: _motion_segments_invariant() == (),
         "An incomplete motion reading starts a fresh measured segment of the held input; contradictory motion retires that trial."),
        ("missing_readings_preserve_measured_control_prefixes", "core/agency/which_one_answers_to_her.py",
         "tests/test_measured_control_prefix_survives_a_gap.py", lambda: _measured_prefix_invariant() == (),
         "A missing reading closes a finite measured control prefix without extending it across the gap or creating another trial; contradiction before input completion retires it."),
        ("lost_readings_are_refreshed_without_duplicate_requests", "core/perception/reading_again_after_loss.py",
         "tests/test_a_body_guess_keeps_its_observed_track.py", lambda: _reading_renewal_invariant() == (),
         "An unresolved reading can be refreshed after measured loss; unread frames, active requests and current ownership do not authorize another request."),
        ("missing_points_do_not_relax_object_identity", "core/perception/what_moves_in_the_picture.py",
         "tests/test_points_keep_their_observed_motion.py", lambda: _missing_points_invariant() == (),
         "Missing point support cannot relax measured shape and appearance continuity or lend an identity through a shared kind."),
        ("current_point_pose_keeps_restored_object_lifetime", "core/perception/what_moves_in_the_picture.py",
         "tests/test_a_continuing_point_pose_keeps_its_objects_birth.py", lambda: _point_pose_birth_invariant() == (),
         "A current point pose shares its object's restored lifetime while retaining its own angle basis; stale or unrelated poses cannot inherit that tag."),
        ("material_identity_requires_unique_current_geometry", "core/perception/continuing_material_before_outlines.py",
         "tests/test_continuing_material_survives_a_changed_outline.py", lambda: _material_identity_invariant() == (),
         "Continuing owned points can preserve observation identity through outline changes only with current rigid geometry and a unique match on both sides; no control authority is supplied."),
        ("trail_cleanup_uses_current_material_travel", "core/perception/what_moves_in_the_picture.py",
         "tests/test_material_motion_keeps_a_stopped_body_visible.py", lambda: _material_trail_invariant() == (),
         "A current material pose uses its measured path for trail cleanup; outline deformation cannot supply physical travel and an invalid material path supplies none."),
        ("point_meaning_requires_current_owned_support", "core/perception/what_the_points_on_a_thing_say.py",
         "tests/test_the_points_on_a_thing_say_how_soon_it_arrives.py", lambda: _point_meaning_invariant() == (),
         "Point-derived approach and identity require current owned pixels; absent support and ambiguous or expired owners remain unknown."),
        ("visible_corner_keeps_its_whole_connected_patch", "core/perception/the_whole_patch_around_a_visible_corner.py",
         "tests/test_a_visible_corner_keeps_the_whole_patch.py", lambda: _whole_patch_invariant() == (),
         "A foreground corner can recover its bounded connected colour patch when a measured surrounding floor distinguishes it."),
        ("point_identity_needs_current_pixels", "core/perception/points_that_keep_their_motion.py",
         "tests/test_points_keep_their_observed_motion.py", lambda: _point_identity_invariant() == (),
         "Bounded point tracks retain measured identities across consecutive frames and retire them on occlusion, gaps or resets."),
    )
