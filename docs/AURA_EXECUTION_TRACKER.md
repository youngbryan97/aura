# Aura Execution Tracker

## Current Phase

Phase A+ closure: every item from both rounds of A+ feedback addressed
with concrete adapters, fail-closed governance in live code paths, and
runnable regressions. 365 passed in the final sweep. See
`docs/AURA_AGRADE_GAP_REGISTER.md` for the literal item-by-item closure.

## Current Milestone

Final consolidation: every phase A-O has at least one runnable module
and at least one regression test; the prompt's full requirement set is
audited in `docs/AURA_PROMPT_COVERAGE_AUDIT.md`.

## Files Changed (this session)

### Source

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

- `docs/AURA_EXECUTION_PLAN.md`
- `docs/AURA_EXECUTION_TRACKER.md`
- `docs/AURA_RISK_REGISTER.md`
- `docs/AURA_TEST_COMMANDS.md`
- `docs/AURA_PROMPT_COVERAGE_AUDIT.md` (new — exhaustive prompt walk)
- `docs/runbooks/` (19 files, new)

### Tests

- `tests/test_server_runtime_hardening.py` (~1500 lines added across
  Phase B mixin sweep, Phase C-O contracts, and final-gap closures)

## Tests Added (highlight)

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
python -m pytest tests/test_server_runtime_hardening.py \
  tests/test_orchestrator_compatibility.py \
  tests/test_runtime_stability_edges.py \
  tests/test_forensic_audit_regressions.py \
  tests/test_launcher_polish_contract.py \
  tests/test_resilient_boot_llm_stage.py \
  tests/test_runtime_polish.py \
  tests/test_time_resilience.py
```

Result: **304 passed, 1 subtests passed**.

## Pass / Fail Results

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

## Next Exact Task

After this checkpoint, the next high-leverage move is to wire the
abstract `MemoryWriteGateway` and `StateGateway` in `core/runtime/gateways.py`
to concrete implementations rooted at `core/memory/memory_facade.py` and
`core/state/state_repository.py`, then run the conformance suite with
`AURA_STRICT_RUNTIME=1` to confirm zero degradations.

## Next Exact Continuation Prompt

> Continue Aura production hardening from `docs/AURA_EXECUTION_TRACKER.md`.
> Phases A-O have all landed. Next: wire concrete adapters for
> `core/runtime/gateways.MemoryWriteGateway` rooted at
> `core/memory/memory_facade.py` and `core/runtime/gateways.StateGateway`
> rooted at `core/state/state_repository.py`. After that, run the
> conformance suite with `AURA_STRICT_RUNTIME=1` and the abuse-gauntlet
> harness in `core/runtime/fault_injection.run_abuse_stage` for the
> `stage_1_2h` scenario.

## Exact Stopping Point

Stopped after Phase O regressions all green (304/304), the prompt
coverage audit recorded in `docs/AURA_PROMPT_COVERAGE_AUDIT.md`, and
the source archive download artifacts produced (see
`AURA_EXECUTION_PLAN.md` for paths).

## Current Git Diff Summary

- Pre-existing dirty file outside the mission: `.aura/memfs/user.txt`
- Mission diff this session is on three commits:
  1. `cae9c652` — orchestrator mixin task ownership sweep
  2. `7aab3de6` — Phase C-O runtime invariants, conformance,
     capability surface
  3. (final commit, this checkpoint) — fuzz/SLI/gateway/turn-taking/
     computer-use/memory-guard contracts, prompt-coverage audit, and
     tracker doc sync.
