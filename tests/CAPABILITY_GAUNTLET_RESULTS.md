# Aura Capability Gauntlet Results

**Date:** 2026-05-05 11:30:30
**Status:** Execution Complete

## Execution Summary

### Closure Gauntlet
- **File:** `tests/test_audit_chain.py`
- **Status:** ✅ PASS
- **Duration:** 0.33s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 13 items

tests/test_audit_chain.py::test_canonical_json_is_deterministic PASSED   [  7%]
tests/test_audit_chain.py::test_hash_receipt_body_is_stable PASSED       [ 15%]
tests/test_audit_chain.py::test_compute_entry_hash_changes_with_any_field PASSED [ 23%]
tests/test_audit_chain.py::test_emit_appends_one_entry_per_receipt PASSED [ 30%]
tests/test_audit_chain.py::test_chain_head_advances_monotonically PASSED [ 38%]
tests/test_audit_chain.py::test_chain_persists_across_restart PASSED     [ 46%]
tests/test_audit_chain.py::test_detects_modified_receipt_body PASSED     [ 53%]
tests/test_audit_chain.py::test_detects_modified_chain_entry PASSED      [ 61%]
tests/test_audit_chain.py::test_detects_broken_link PASSED               [ 69%]
tests/test_audit_chain.py::test_detects_deleted_entry PASSED             [ 76%]
tests/test_audit_chain.py::test_detects_missing_receipt_body PASSED      [ 84%]
tests/test_audit_chain.py::test_export_produces_portable_bundle PASSED   [ 92%]
tests/test_audit_chain.py::test_exported_chain_can_be_independently_verified PASSED [100%]

============================== 13 passed in 0.17s ==============================


```

### Activation Audit
- **File:** `tests/test_causal_exclusion.py`
- **Status:** ✅ PASS
- **Duration:** 1.41s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 10 items

tests/test_causal_exclusion.py::TestCryptographicStateBinding::test_stack_state_produces_distinct_llm_params PASSED [ 10%]
tests/test_causal_exclusion.py::TestCryptographicStateBinding::test_seed_derived_states_are_informationally_distinct PASSED [ 20%]
tests/test_causal_exclusion.py::TestCryptographicStateBinding::test_narrative_changes_with_stack_state PASSED [ 30%]
tests/test_causal_exclusion.py::TestCryptographicStateBinding::test_temperature_modulation_tracks_arousal PASSED [ 40%]
tests/test_causal_exclusion.py::TestCounterfactualInjection::test_wrong_state_produces_different_params PASSED [ 50%]
tests/test_causal_exclusion.py::TestCounterfactualInjection::test_state_reversal_produces_param_reversal PASSED [ 60%]
tests/test_causal_exclusion.py::TestRLHFIsolation::test_extreme_states_produce_distinct_params_vs_human_approx PASSED [ 70%]
tests/test_causal_exclusion.py::TestRLHFIsolation::test_receptor_adaptation_makes_same_event_produce_different_params PASSED [ 80%]
tests/test_causal_exclusion.py::TestPhiCausalExclusion::test_phi_boost_changes_competition_outcome PASSED [ 90%]
tests/test_causal_exclusion.py::TestPhiCausalExclusion::test_phi_zero_provides_no_boost PASSED [100%]

============================== 10 passed in 1.05s ==============================


```

### Headless Environment Stress
- **File:** `tests/test_headless_live_boot.py`
- **Status:** ❌ FAIL
- **Duration:** 0.22s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 0 items

============================ no tests ran in 0.08s =============================


```

### Replay Learning Improvement
- **File:** `tests/test_canary_replay_real.py`
- **Status:** ✅ PASS
- **Duration:** 0.28s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 7 items

tests/test_canary_replay_real.py::TestCanaryReplayReal::test_interaction_records_available PASSED [ 14%]
tests/test_canary_replay_real.py::TestCanaryReplayReal::test_build_replay_examples PASSED [ 28%]
tests/test_canary_replay_real.py::TestCanaryReplayReal::test_canary_baseline_identity PASSED [ 42%]
tests/test_canary_replay_real.py::TestCanaryReplayReal::test_canary_patched_still_passes PASSED [ 57%]
tests/test_canary_replay_real.py::TestCanaryReplayReal::test_canary_catastrophic_regression_detected PASSED [ 71%]
tests/test_canary_replay_real.py::TestCanaryReplayReal::test_semantic_verifier_consistency PASSED [ 85%]
tests/test_canary_replay_real.py::TestCanaryReplayReal::test_canary_report_serializable PASSED [100%]

============================== 7 passed in 0.14s ===============================


```

### Abstraction Transfer
- **File:** `tests/test_grounding_and_plasticity.py`
- **Status:** ✅ PASS
- **Duration:** 0.39s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 16 items

tests/test_grounding_and_plasticity.py::test_grounded_symbol_has_concept_evidence_and_method PASSED [  6%]
tests/test_grounding_and_plasticity.py::test_confirmed_examples_raise_confidence PASSED [ 12%]
tests/test_grounding_and_plasticity.py::test_negative_feedback_weakens_link PASSED [ 18%]
tests/test_grounding_and_plasticity.py::test_predict_unknown_symbol_returns_no_match PASSED [ 25%]
tests/test_grounding_and_plasticity.py::test_grounding_predicts_correctly_after_examples PASSED [ 31%]
tests/test_grounding_and_
...[TRUNCATED]...
update_without_activity PASSED [ 43%]
tests/test_grounding_and_plasticity.py::test_plastic_layer_update_after_forward_changes_hebb PASSED [ 50%]
tests/test_grounding_and_plasticity.py::test_plastic_layer_max_delta_norm_caps_step PASSED [ 56%]
tests/test_grounding_and_plasticity.py::test_plastic_layer_reset_zeroes_state PASSED [ 62%]
tests/test_grounding_and_plasticity.py::test_adapter_changes_features_after_reward PASSED [ 68%]
tests/test_grounding_and_plasticity.py::test_governor_refuses_low_vitality PASSED [ 75%]
tests/test_grounding_and_plasticity.py::test_governor_refuses_weak_reward PASSED [ 81%]
tests/test_grounding_and_plasticity.py::test_governor_allows_with_modulation_in_unit_range PASSED [ 87%]
tests/test_grounding_and_plasticity.py::test_grounding_improves_heldout_after_confirmed_examples PASSED [ 93%]
tests/test_grounding_and_plasticity.py::test_grounding_persists_across_reopen PASSED [100%]

============================== 16 passed in 0.23s ==============================


```

### Self-Mod Rollback Drill
- **File:** `tests/test_restore_drill.py`
- **Status:** ✅ PASS
- **Duration:** 0.34s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 9 items

tests/test_restore_drill.py::test_fingerprint_tree_is_stable PASSED      [ 11%]
tests/test_restore_drill.py::test_fingerprint_changes_on_edit PASSED     [ 22%]
tests/test_restore_drill.py::test_perform_drill_full_roundtrip_succeeds PASSED [ 33%]
tests/test_restore_drill.py::test_drill_report_serializes PASSED         [ 44%]
tests/test_restore_drill.py::test_drill_uses_ephemeral_dirs_when_not_supplied PASSED [ 55%]
tests/test_restore_drill.py::test_drill_detects_post_restore_corruption PASSED [ 66%]
tests/test_restore_drill.py::test_aura_home_respects_env PASSED          [ 77%]
tests/test_restore_drill.py::test_perform_backup_honours_aura_home PASSED [ 88%]
tests/test_restore_drill.py::test_backup_restore_round_trip_via_aura_home PASSED [100%]

============================== 9 passed in 0.19s ===============================


```

### Production 32B CAA Validation
- **File:** `tests/steering/test_caa_32b.py`
- **Status:** ✅ PASS
- **Duration:** 0.37s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 6 items

tests/steering/test_caa_32b.py::TestSteeringABPipeline::test_analyze_requires_minimum_trials PASSED [ 16%]
tests/steering/test_caa_32b.py::TestSteeringABPipeline::test_analyze_requires_all_conditions PASSED [ 33%]
tests/steering/test_caa_32b.py::TestSteeringABPipeline::test_analyze_returns_report PASSED [ 50%]
tests/steering/test_caa_32b.py::TestSteeringABPipeline::test_synthetic_divergence_detectable PASSED [ 66%]
tests/steering/test_caa_32b.py::TestSteeringABPipeline::test_report_serialization PASSED [ 83%]
tests/steering/test_caa_32b.py::TestSteeringABLive::test_live_steering_divergence SKIPPED [100%]

========================= 5 passed, 1 skipped in 0.22s =========================


```

### Long-Run Stability Trace
- **File:** `tests/test_long_run_model.py`
- **Status:** ✅ PASS
- **Duration:** 1.56s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 7 items

tests/test_long_run_model.py::test_build_registry_extracts_runtime_hardening_contracts PASSED [ 14%]
tests/test_long_run_model.py::test_run_forecast_supports_all_profiles PASSED [ 28%]
tests/test_long_run_model.py::test_run_forecast_surfaces_requested_retention_cliffs PASSED [ 42%]
tests/test_long_run_model.py::test_run_forecast_restart_reentry_pressure_changes_by_restart_kind PASSED [ 57%]
tests/test_long_run_model.py::test_run_forecast_bounds_pressure_and_tracks_repairs PASSED [ 71%]
tests/test_long_run_model.py::test_run_forecast_keeps_social_budget_off_the_floor_under_stress_load PASSED [ 85%]
tests/test_long_run_model.py::test_write_report_bundle_emits_markdown_and_json PASSED [100%]

============================== 7 passed in 1.24s ===============================


```

### External Task Performance
- **File:** `tests/test_agent_workspace_integrations.py`
- **Status:** ✅ PASS
- **Duration:** 0.30s

```text
============================= test session starts ==============================
platform darwin -- Python 3.12.13, pytest-9.0.3, pluggy-1.6.0 -- /Users/bryan/.aura/live-source/.venv/bin/python
cachedir: .pytest_cache
rootdir: /Users/bryan/.aura/live-source
configfile: pyproject.toml
plugins: asyncio-1.3.0, anyio-4.13.0
asyncio: mode=Mode.AUTO, debug=False, asyncio_default_fixture_loop_scope=function, asyncio_default_test_loop_scope=function
collecting ... collected 9 items

tests/test_agent_workspace_integrations.py::test_markdown_workspace_commits_search_and_revert PASSED [ 11%]
tests/test_agent_workspace_integrations.py::test_markdown_workspace_nonblocking_conflict_merge PASSED [ 22%]
tests/test_agent_workspace_integrations.py::test_markdown_workspace_permissions PASSED [ 33%]
tests/test_agent_workspace_integrations.py::test_aura_workspace_gates_writes_and_commits_receipted_artifacts PASSED [ 44%]
tests/test_agent_workspace_integrations.py::test_aura_workspace_accepts_directory_scoped_capability_for_video_evidence PASSED [ 55%]
tests/test_agent_workspace_integrations.py::test_temporal_atlas_expands_marks_dead_and_tracks_evidence PASSED [ 66%]
tests/test_agent_workspace_integrations.py::test_simulation_well_plans_and_streams_local_records PASSED [ 77%]
tests/test_agent_workspace_integrations.py::test_agent_workspace_is_architecture_manifest_role PASSED [ 88%]
tests/test_agent_workspace_integrations.py::test_agent_workspace_activation_spec_is_required_and_autostarted PASSED [100%]

============================== 9 passed in 0.15s ===============================


```

