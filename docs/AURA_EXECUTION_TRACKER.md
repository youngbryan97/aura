# Aura Execution Tracker

## Current Phase

General environment autonomy closure: NetHack is now wired through the
canonical environment kernel as a stress adapter, not through a separate
game-specific control loop. The kernel has live observation normalization,
belief/spatial memory, shared HTN policy, simulation, governance, action
gateway, command compilation, semantic outcome learning, run lifecycle,
postmortems, and trace replay.

## Current Milestone

General infrastructure hardening for arbitrary bounded environments. The
capability matrix in `core/environment/capability_matrix.py` is executable
and covers the live organs required for NetHack-scale runs without encoding
NetHack strategy in shared code.

## Files Changed (this session)

### Source

- `core/environment/environment_kernel.py` (shared HTN planner wiring,
  run lifecycle, service binding, post-action observation, semantic
  learning, resource deltas, terminal detection)
- `core/environment/belief_graph.py` (metadata-rich canonical spatial
  memory with hazard-preserving merge and legacy kind compatibility)
- `core/environment/capability_matrix.py` (new executable capability audit)
- `core/environment/generic_command_handlers.py` (generic handlers bind to
  concrete environment IDs)
- `core/environment/state_compiler.py` (legacy terminal state converts to
  canonical x/y coordinates and modal factory)
- `core/environment/outcome_attribution.py` (death and no-progress scoring)
- `core/environment/outcome/semantic_diff.py` (resource, modal, entity, and
  fatal-event diffs)
- `core/environment/policy/candidate_generator.py` (inventory, spatial,
  transition, and hazard-aware candidates)
- `core/environment/policy/action_ranker.py`
- `core/environment/simulation.py`
- `core/environment/governance_bridge.py` (authority-required effects fail
  closed when authority is unavailable)
- `core/environment/lifecycle_manager.py`
- `core/environment/strategy/goal_seeder.py` (capability-family goals instead
  of NetHack-specific milestones)
- `core/environments/terminal_grid/nethack_commands.py` (aliases for generic
  intents emitted by shared policy)
- `core/embodiment/games/nethack/state_compiler.py` (compatibility import for
  canonical compiler)
- `challenges/nethack_challenge.py` (canonical EnvironmentKernel stress loop)
- `aura_main.py` (manifest enforcement after lock_registration)
- `core/orchestrator/mixins/cognitive_background.py`
- `core/orchestrator/mixins/message_handling.py`
- `core/orchestrator/mixins/incoming_logic.py`
- `core/orchestrator/mixins/output_formatter.py`
- `core/orchestrator/mixins/autonomy.py`
- `core/runtime/service_manifest.py` (new)
- `core/runtime/shutdown_coordinator.py` (new)
- `core/runtime/will_transaction.py` (new)
- `core/runtime/atomic_writer.py` (new)
- `core/runtime/self_repair_ladder.py` (new)
- `core/runtime/fault_injection.py` (new)
- `core/runtime/conformance.py` (new)
- `core/runtime/depth_audit.py` (new)
- `core/runtime/skill_contract.py` (new)
- `core/runtime/security.py` (new)
- `core/runtime/formal_models.py` (new)
- `core/runtime/release_channels.py` (new)
- `core/runtime/fuzz_harness.py` (new)
- `core/runtime/telemetry_sli.py` (new)
- `core/runtime/gateways.py` (new)
- `core/runtime/memory_guard.py` (new)
- `core/perception/__init__.py` (new)
- `core/perception/perception_runtime.py` (new)
- `core/social/turn_taking.py` (new)
- `core/tools/computer_use.py` (new)

### Docs

- `docs/GENERAL_ENVIRONMENT_AUTONOMY.md` (new)
- `CHALLENGE.md`
- `docs/AURA_TEST_COMMANDS.md`
- `docs/AURA_EXECUTION_PLAN.md`
- `docs/AURA_EXECUTION_TRACKER.md`
- `docs/AURA_RISK_REGISTER.md`
- `docs/AURA_TEST_COMMANDS.md`
- `docs/AURA_PROMPT_COVERAGE_AUDIT.md` (new — exhaustive prompt walk)
- `docs/runbooks/` (19 files, new)

### Tests

- `tests/test_environment_general_integration.py` (new)
- `tests/test_server_runtime_hardening.py` (~1500 lines added across
  Phase B mixin sweep, Phase C-O contracts, and final-gap closures)

## Tests Added (highlight)

- `test_generic_command_handlers_bind_to_concrete_environment_id`
- `test_policy_reads_inventory_items_and_emits_generic_stair_intent`
- `test_belief_spatial_memory_keeps_metadata_and_legacy_kind_lookup`
- `test_semantic_diff_reports_resources_modal_and_new_entities`
- `test_kernel_lifecycle_records_terminal_death_and_postmortem`
- `test_environment_capability_matrix_is_executable_and_clean`
- `test_cognitive_background_reflection_uses_named_tracker`
- `test_cognitive_background_learning_uses_named_tracker`
- `test_message_handling_deferred_enqueue_uses_named_tracker`
- `test_message_handling_dispatch_uses_named_tracker`
- `test_incoming_logic_handle_message_uses_named_tracker`
- `test_output_formatter_eternal_snapshot_uses_named_tracker`
- `test_output_formatter_emit_thought_stream_uses_named_tracker`
- `test_autonomy_thought_uses_named_tracker`
- `test_service_manifest_*` (4)
- `test_aura_main_invokes_service_manifest_after_lock_registration`
- `test_aura_main_strict_runtime_aborts_on_manifest_critical_violation`
- `test_shutdown_coordinator_*` (6)
- `test_will_transaction_*` (6)
- `test_atomic_writer_*` (7)
- `test_actor_health_gate_*`, `test_supervision_tree_*` (6)
- `test_self_repair_ladder_*` (9)
- `test_conformance_*` (10)
- `test_fault_injector_*`, `test_abuse_gauntlet_*` (8)
- `test_depth_audit_*` (3)
- `test_skill_contract_*`, `test_skill_registry_*` (3)
- `test_perception_runtime_*`, `test_movie_session_memory_*`, `test_silence_policy_*` (5)
- `test_sandbox_policy_*` (6)
- `test_formal_*` (7)
- `test_release_channels_*` (3)
- `test_runbook_index_lists_every_named_scenario`
- `test_fuzz_target_*`, `test_telemetry_sli_*`, `test_gateway_contracts_*`,
  `test_turn_taking_*`, `test_computer_use_*`, `test_memory_guard_*` (13)

## Commands Run (final sweep)

```
python -m pytest tests/test_environment_general_integration.py \
  tests/environment/final_blockers tests/environments/terminal_grid \
  tests/architecture tests/test_embodied_cognition_runtime.py \
  tests/nethack_crucible.py -q

python challenges/nethack_challenge.py --mode simulated --steps 20 \
  --trace artifacts/test_nethack_kernel_trace.jsonl --log-level ERROR

python -m pytest -q

python -m pytest tests/test_server_runtime_hardening.py \
  tests/test_orchestrator_compatibility.py \
  tests/test_runtime_stability_edges.py \
  tests/test_forensic_audit_regressions.py \
  tests/test_launcher_polish_contract.py \
  tests/test_resilient_boot_llm_stage.py \
  tests/test_runtime_polish.py \
  tests/test_time_resilience.py
```

General environment result: **210 passed**. Simulated challenge emitted a
hash-chained trace at `artifacts/test_nethack_kernel_trace.jsonl`.
Full repository result: **4333 passed, 7 skipped, 7 warnings,
1 subtests passed** in 479.24s.

## Pass / Fail Results

- general environment autonomy slice: 210 passed
- simulated NetHack stress canary: passed, trace emitted
- full repository sweep: 4333 passed, 7 skipped, 7 warnings, 1 subtests passed
- mixin ownership slice: 8 passed
- service manifest slice: 6 passed
- shutdown coordinator slice: 6 passed
- will transaction slice: 6 passed
- atomic writer slice: 7 passed
- actor supervisor proof slice: 6 passed
- self repair ladder slice: 9 passed
- conformance + fault injection slice: 19 passed
- depth audit slice: 3 passed
- skill contract slice: 3 passed
- perception slice: 5 passed
- security slice: 6 passed
- formal protocol slice: 7 passed
- release channels + runbooks slice: 4 passed
- fuzz/SLI/gateway/turn-taking/computer-use/memory-guard slice: 13 passed
- broad regression sweep: 304 passed

## Unresolved Failures / Known Backlog

1. **R-001**: AGENTS.md, AURA_MASTER_SPEC.md, docs/AURA_MASTER_SPEC.md,
   docs/RUNTIME_INVARIANTS.md, docs/PRODUCTION_HARDENING_PLAN.md,
   docs/SKILL_CERTIFICATION_MATRIX.md, docs/DEPTH_AUDIT.md,
   docs/ABUSE_GAUNTLET.md, docs/FORMAL_VERIFICATION_PLAN.md never
   landed in the repo. The contracts those documents implied are now
   captured as runnable modules (service_manifest, depth_audit,
   fault_injection abuse stages, formal_models, conformance, etc.).
2. Real hardware drivers (camera/microphone/screen/subtitle) are
   contract-only. Phase L scaffolds the contract but no platform
   driver is bundled.
3. OpenTelemetry / Prometheus exporter wiring is catalog-only
   (`telemetry_sli.SLO_CATALOG`).
4. Operator CLI (`aura doctor`, `aura conformance`, etc.) is described
   in the plan but not yet implemented as a CLI surface.
5. Multimodal model router, durable-workflow engine, external red-team
   automation, day-in-the-life 24h soak runner are documented in the
   plan as Phase J/K/L follow-ons.
6. Successful NetHack ascension is not yet recorded. The architecture is now
   wired for strict-real runs, but a full autonomous win remains an empirical
   long-run target.

## Next Exact Task

After this checkpoint, the next high-leverage move is to run the strict-real
NetHack stress loop, inspect `EnvironmentKernel.run_manager.records` and
the black-box trace every 15 minutes, and fix only general architecture
failures: policy loops, modal failures, belief/spatial merge errors, action
gateway gaps, semantic diff blind spots, or outcome-learning regressions.

## Next Exact Continuation Prompt

> Continue Aura general environment autonomy from
> `docs/AURA_EXECUTION_TRACKER.md`. The canonical EnvironmentKernel is wired
> to NetHack as a stress adapter. Run
> `python challenges/nethack_challenge.py --mode strict_real --steps 5000`,
> inspect the trace and run-manager records, and patch only general
> infrastructure causes of stalls or loops.

## Exact Stopping Point

Stopped after the general environment autonomy slice was green (210/210),
the simulated stress canary emitted a trace, and docs were updated to the
current canonical-kernel state.

## Current Git Diff Summary

- Pre-existing dirty file outside the mission: `.aura/memfs/user.txt`
- Mission diff this session is on three commits:
  1. `cae9c652` — orchestrator mixin task ownership sweep
  2. `7aab3de6` — Phase C-O runtime invariants, conformance,
     capability surface
  3. (final commit, this checkpoint) — fuzz/SLI/gateway/turn-taking/
     computer-use/memory-guard contracts, prompt-coverage audit, and
     tracker doc sync.
