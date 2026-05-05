# Aura Test Commands

## Baseline Discovery

Use these first for the current runtime-breaker milestone:

```bash
python -m pytest tests/test_orchestrator_compatibility.py -q
python -m pytest tests/test_forensic_audit_regressions.py -q
python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository"
python -m pytest tests/test_runtime_stability_edges.py -q
python -m pytest tests/test_controlled_complexity_runtime.py -q
python -m pytest tests/test_effect_closure.py -q
python -m pytest tests/test_skill_surface_contracts.py -q -k "safe_execute"
```

## Runtime Breaker Slice

After the first patch set lands:

```bash
python -m pytest tests/test_orchestrator_compatibility.py -q
python -m pytest tests/test_forensic_audit_regressions.py -q
python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository"
python -m pytest tests/test_runtime_stability_edges.py -q
python -m pytest tests/test_controlled_complexity_runtime.py -q
python -m pytest tests/test_effect_closure.py -q
python -m pytest tests/test_skill_surface_contracts.py -q -k "safe_execute"
```

## Launcher / Supervisor Slice

For the watchdog / runtime-singularity / supervisor checkpoint:

```bash
python -m pytest tests/test_launcher_polish_contract.py tests/test_time_resilience.py
python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository or supervisor or actor_health_gate or reaper_manifest"
python -m pytest tests/test_orchestrator_compatibility.py -q
python -m pytest tests/test_runtime_stability_edges.py -q
python -m py_compile \
  aura_main.py \
  core/bus/actor_bus.py \
  core/bus/local_pipe_bus.py \
  core/reaper.py \
  core/resilience/sovereign_watchdog.py \
  core/supervisor/tree.py \
  scripts/one_off/launch_aura_3d.py
```

## Strict Boot / Server Ownership Slice

For the early B3 strict-runtime and websocket ownership work:

```bash
python -m pytest tests/test_resilient_boot_llm_stage.py tests/test_runtime_polish.py -k "resilient_boot or websocket_manager"
python -m pytest tests/test_forensic_audit_regressions.py -q -k "graceful_shutdown_signal"
python -m py_compile \
  core/graceful_shutdown.py \
  core/ops/resilient_boot.py \
  interface/websocket_manager.py \
  tests/test_forensic_audit_regressions.py \
  tests/test_resilient_boot_llm_stage.py \
  tests/test_runtime_polish.py
```

## State / Event / Scheduler Ownership Slice

For the later B3 infrastructure ownership pass:

```bash
python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or event_bus or scheduler or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or continuous_cognition_loop_is_task_tracked or session_guardian_monitor_loop_is_task_tracked or system_governor_health_loop_is_task_tracked or state_vault_actor_background_tasks_use_task_tracker or sensory_gate_actor_background_tasks_use_task_tracker or reaper_manifest or actor_health_gate"
python -m pytest tests/test_server_runtime_hardening.py -q -k "conversation_loop_start_is_task_tracked or conversation_loop_reflection_task_is_tracked or message_coordinator_acquire_next_message_tracks_liquid_state_update or message_coordinator_dispatch_uses_task_tracker or message_coordinator_handle_incoming_message_tracks_reply_task or metabolic_coordinator_trigger_background_reflection_is_task_tracked or metabolic_coordinator_trigger_background_learning_is_task_tracked or metabolic_coordinator_autonomous_thought_is_task_tracked or metabolic_coordinator_terminal_self_heal_is_task_tracked or metabolic_coordinator_process_cycle_tracks_bootstrap_and_drive_tasks or metabolic_coordinator_process_cycle_tracks_kernel_background_tasks or metabolic_coordinator_update_liquid_pacing_tracks_liquid_state_update or metabolic_coordinator_emit_telemetry_pulse_tracks_recovery or metabolic_coordinator_impulses_are_task_tracked or metabolic_coordinator_memory_hygiene_tracks_maintenance_tasks or metabolic_coordinator_process_world_decay_tracks_archive_and_evolution or cognitive_coordinator_voice_tts_is_task_tracked or cognitive_coordinator_surprise_learning_is_task_tracked or cognitive_coordinator_dream_liquid_state_update_is_task_tracked or lifecycle_coordinator_start_tracks_background_boot_loops or lifecycle_coordinator_handle_signal_uses_task_tracker or state_vault_actor_background_tasks_use_task_tracker or sensory_gate_actor_background_tasks_use_task_tracker or continuous_cognition_loop_is_task_tracked or session_guardian_monitor_loop_is_task_tracked or system_governor_health_loop_is_task_tracked or scheduler or event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"
python -m pytest tests/test_launcher_polish_contract.py -q
python -m pytest tests/test_orchestrator_compatibility.py -q
python -m pytest tests/test_runtime_stability_edges.py -q
python -m py_compile \
  aura_main.py \
  core/actors/sensory_gate.py \
  core/bus/actor_bus.py \
  core/bus/local_pipe_bus.py \
  core/conversation_loop.py \
  core/coordinators/cognitive_coordinator.py \
  core/coordinators/lifecycle_coordinator.py \
  core/coordinators/message_coordinator.py \
  core/coordinators/metabolic_coordinator.py \
  core/continuous_cognition.py \
  core/event_bus.py \
  core/guardians/governor.py \
  core/scheduler.py \
  core/session_guardian.py \
  core/state/state_repository.py \
  core/state/vault.py \
  tests/test_launcher_polish_contract.py \
  tests/test_server_runtime_hardening.py
```

## Import / Syntax Spot Checks

```bash
python -m py_compile \
  infrastructure/base_skill.py \
  core/skills/base_skill.py \
  core/capability_engine.py \
  core/bus/local_pipe_bus.py \
  core/graceful_shutdown.py \
  core/actors/sensory_gate.py \
  core/state/vault.py \
  core/orchestrator/mixins/incoming_logic.py \
  core/orchestrator/mixins/boot/boot_resilience.py \
  core/skills/file_operation.py
```

## Existing High-Level Suites

Documented existing suites:

```bash
./scripts/run_audit_suite.sh
./scripts/run_audit_suite.sh quick
bash scripts/run_decisive_test.sh
```

These are not the first milestone commands because they are too broad for a
checkpointed runtime-fix pass.

## Notes

- If a command cannot run because of missing optional dependencies or platform
  constraints, record the exact blocker in `docs/AURA_EXECUTION_TRACKER.md`.
- Add new commands here as each milestone introduces new focused test slices.

## General Environment Autonomy Slice

This is the focused suite for the NetHack-scale general environment kernel:

```bash
python -m pytest \
  tests/test_final_general_hardening.py \
  tests/test_environment_general_integration.py \
  tests/environment/final_blockers \
  tests/environments/terminal_grid/test_nethack_audit_comprehensive.py \
  tests/environments/terminal_grid/test_terminal_grid_live_canary.py \
  tests/environments/terminal_grid/test_nethack_adapter_preflight.py \
  tests/environments/terminal_grid/test_terminal_grid_contract.py \
  tests/architecture \
  tests/test_embodied_cognition_runtime.py \
  tests/test_runtime_stability_edges.py \
  tests/test_runtime_service_access.py \
  tests/nethack_crucible.py -q

python challenges/nethack_challenge.py --mode simulated --steps 20 \
  --trace artifacts/test_nethack_kernel_trace.jsonl --log-level ERROR
```

Current final-pass focused result: **271 passed, 1 subtests passed**.
The simulated stress canary passed and wrote **40 hash-chained trace rows** to
`artifacts/test_nethack_kernel_trace.jsonl`.

Strict-real smoke:

```bash
python challenges/nethack_challenge.py --mode strict_real --steps 40 \
  --trace /tmp/aura_strict_probe_after_threat_response.jsonl --log-level INFO
```

Current strict-real result: reached live `dlvl_1`, resolved startup modals,
opened a door, moved, and stayed alive through 40 steps. This is not an
ascension.

## Phase C-O verification slices

```
# Phase C: ServiceManifest
python -m pytest tests/test_server_runtime_hardening.py -q -k "service_manifest or aura_main_invokes_service_manifest or aura_main_strict_runtime_aborts"

# Phase D: ShutdownCoordinator
python -m pytest tests/test_server_runtime_hardening.py -q -k "shutdown_coordinator"

# Phase E: WillTransaction
python -m pytest tests/test_server_runtime_hardening.py -q -k "will_transaction"

# Phase F: AtomicWriter
python -m pytest tests/test_server_runtime_hardening.py -q -k "atomic_writer"

# Phase G: Actor supervisor proof
python -m pytest tests/test_server_runtime_hardening.py -q -k "actor_health_gate or supervision_tree_handles_actor_failure or supervision_tree_stop_all_terminates_orphans or supervision_tree_records_activity or supervision_tree_record_activity_unknown"

# Phase H: Self-repair ladder
python -m pytest tests/test_server_runtime_hardening.py -q -k "self_repair_ladder"

# Phase I: Conformance + abuse harness
python -m pytest tests/test_server_runtime_hardening.py -q -k "conformance or fault_injector or abuse_gauntlet"

# Phase J: Depth audit
python -m pytest tests/test_server_runtime_hardening.py -q -k "depth_audit"

# Phase K: Skill contracts
python -m pytest tests/test_server_runtime_hardening.py -q -k "skill_contract or skill_registry"

# Phase L: Perception runtime + movie
python -m pytest tests/test_server_runtime_hardening.py -q -k "perception_runtime or movie_session_memory or silence_policy"

# Phase M: Security / sandbox
python -m pytest tests/test_server_runtime_hardening.py -q -k "sandbox_policy"

# Phase N: Formal protocol models
python -m pytest tests/test_server_runtime_hardening.py -q -k "formal_runtime_singularity or formal_governance_receipt or formal_state_commit or formal_actor_lifecycle or formal_self_modification or formal_shutdown_ordering or formal_capability_token"

# Phase O: Release channels + runbooks
python -m pytest tests/test_server_runtime_hardening.py -q -k "release_channels or runbook_index"

# Final fuzz / SLI / gateway / turn-taking / computer-use / memory-guard
python -m pytest tests/test_server_runtime_hardening.py -q -k "fuzz_target or telemetry_sli or gateway_contracts or turn_taking or computer_use or memory_guard"

# Mixin task ownership sweep
python -m pytest tests/test_server_runtime_hardening.py -q -k "cognitive_background_reflection_uses or cognitive_background_learning_uses or message_handling_deferred_enqueue or message_handling_dispatch_uses or incoming_logic_handle_message_uses or output_formatter_eternal_snapshot or output_formatter_emit_thought_stream or autonomy_thought_uses_named"

# Full regression sweep across all touched suites
python -m pytest tests/test_server_runtime_hardening.py tests/test_orchestrator_compatibility.py tests/test_runtime_stability_edges.py tests/test_forensic_audit_regressions.py tests/test_launcher_polish_contract.py tests/test_resilient_boot_llm_stage.py tests/test_runtime_polish.py tests/test_time_resilience.py
```

## AGI / Enterprise Foundations Slice

The foundations work landed under the AGI/enterprise readiness push.
All tests are CPU-only and safe to run while LoRA training is using
the GPU.

```bash
# Tamper-evident audit chain over receipts
python -m pytest tests/test_audit_chain.py -q

# Prediction ledger with Brier + ECE scoring
python -m pytest tests/test_prediction_ledger.py -q

# Formal 11-state task lifecycle + legacy migration
python -m pytest tests/test_task_lifecycle.py -q

# Typed mutation outcomes with quarantine
python -m pytest tests/test_mutation_safety.py -q

# aura doctor --bundle: collectors, redaction, tarball
python -m pytest tests/test_diagnostics_bundle.py -q

# SLO comparator: tolerance + hard-limit semantics
python -m pytest tests/test_slo_gate.py -q

# Run the SLO gate locally (matches CI behaviour)
python -m slo.check --baseline slo/baseline.json
```

## Self-Improving Research Core Slice

The Aura-owned research substrate (Lattice model, promotion gate,
dynamic benchmarks, algorithm discovery, semantic verifier,
unknown-unknown generator, distributed abstractions, autonomous
driver). All CPU-only.

```bash
# Hybrid attention/SSM/MoE/world-head architecture
python -m pytest tests/test_lattice_model.py tests/test_lattice_trainer.py -q

# Promotion gate + dynamic benchmark + holdout vault
python -m pytest tests/test_promotion_gate.py tests/test_dynamic_benchmark.py tests/test_holdout_vault.py -q

# Algorithm discovery + AST-restricted sandbox
python -m pytest tests/test_expression_evolver.py tests/test_safe_code_evaluator.py -q

# Multi-channel semantic verifier
python -m pytest tests/test_semantic_verifier.py -q

# Unknown-unknown generation + entropy probe
python -m pytest tests/test_unknown_generator.py -q

# Distributed abstractions (compression + grad sync fallback)
python -m pytest tests/test_distributed_substrate.py -q

# Autonomous research-core driver (unit + integration)
python -m pytest tests/test_research_core.py -q

# Live end-to-end test: real cycles, real receipts, real bundle
python -m pytest tests/test_research_core_live.py -q
```
