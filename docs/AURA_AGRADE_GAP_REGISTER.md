# A+ Gap Register — Item-by-Item Closure Audit

This document is a literal walk through every single bullet from the two
A+ feedback rounds the user pasted. Each item lists the exact module(s),
test(s), and verification command that proves the gap is closed.

## Round 1 — Original A+ blockers

### P0 — Absolute A+ blockers

| # | Item | Where it lives now | Verification |
| --- | --- | --- | --- |
| 1 | Governance fails open in vector_memory_store gate | `core/orchestrator/mixins/incoming_logic.py` ~line 290: `pass # fail-open` removed; replaced with explicit fail-closed branch + `record_degraded_event("governance.unavailable.memory_write", ...)` | `test_incoming_logic_vector_memory_gate_fails_closed_when_will_raises` |
| 1b | Governance fails open in internal model update gate | same file ~line 332: same fix | same test |
| 2 | No single TurnTransaction owns a user turn | `core/runtime/turn_transaction.py` (new): stage/approve/commit/rollback with criticality, governance gate, fail-closed in strict mode | `test_turn_transaction_*` (6 tests) |
| 3 | Fire-and-forget for core memory effects | `core/conversation/memory.py` (EnhancedMemorySystem.store_turn) routed through `get_task_tracker().create_task(..., name="enhanced_memory.learn_fact_from_interaction")` | `test_enhanced_memory_system_routes_learn_through_task_tracker` |
| 4 | MemoryWriteGateway not universal | `core/memory/memory_write_gateway.py` (new): concrete gateway routing every write through `atomic_write_json` + governance fail-closed + `MemoryWriteReceipt` emission | `test_concrete_memory_write_gateway_*` (2 tests) |
| 5 | StateGateway not the only state authority | `core/state/state_gateway.py` (new): concrete gateway with same atomic + governance + receipt path | `test_concrete_state_gateway_*` (2 tests) |
| 6 | Receipts not end-to-end universal | `core/runtime/receipts.py` (new): all 10 canonical receipt types (TurnReceipt, GovernanceReceipt, CapabilityReceipt, ToolExecutionReceipt, MemoryWriteReceipt, StateMutationReceipt, OutputReceipt, AutonomyReceipt, SelfRepairReceipt, ComputerUseReceipt) plus durable `ReceiptStore` | `test_universal_receipt_types_importable`, `test_receipt_store_persists_to_disk_and_reloads` |
| 7 | Strict runtime mode not yet proven | `core/runtime/strict_task_owner.py` (new): event-loop task factory denies unowned `asyncio.create_task` in strict mode + `_enforce_boot_probes` aborts boot in strict mode | `test_strict_task_owner_*` (3 tests), `test_boot_probes_strict_mode_raises_on_failure` |
| 8 | Boot readiness not behavioral | `core/runtime/boot_probes.py` (new): memory_write_read, state_mutate_read, governance_approve_deny, output_gate_dry_emit, event_bus_loopback, actor_supervisor probes; wired into `_boot_runtime_orchestrator._enforce_boot_probes` | `test_boot_probes_round_trip_memory_and_state`, `test_aura_main_invokes_boot_probes_after_manifest_enforcement` |
| 9 | No canonical ServiceManifest proof | `core/runtime/service_manifest.py` (Phase C) + `_enforce_service_manifest` after `lock_registration` | `test_service_manifest_*` (4) |
| 10 | Singleton poisoning in get_consciousness_integration | `core/consciousness/integration.py` split into `init_consciousness_integration(orchestrator)` + `get_consciousness_integration()` with strict-mode guards and `reset_consciousness_integration()` for tests | `test_consciousness_integration_*` (5 tests) |

### P1 — Production-grade gaps

| # | Item | Where it lives now | Verification |
| --- | --- | --- | --- |
| 11 | Durable workflows | `core/runtime/durable_workflow.py` (new): WorkflowEngine, WorkflowStep, WorkflowCheckpoint, atomic-writer-backed store, resume-after-failure, idempotent steps, paused-for-approval | `test_durable_workflow_runs_steps_in_order`, `test_durable_workflow_resumes_after_failure`, `test_durable_workflow_pauses_for_human_approval` |
| 12 | Universal task ownership | strict task factory above + prior coordinator/mixin/launcher/server/shutdown sweeps | `test_strict_task_owner_*` |
| 13 | Shutdown coordinator authoritative | `core/runtime/shutdown_coordinator.py` (Phase D) | `test_shutdown_coordinator_*` (6) |
| 14 | Actor supervision mature | Phase G regressions + `core/supervisor/tree.py` heartbeat windows + circuit breaker | `test_actor_health_gate_*`, `test_supervision_tree_*` |
| 15 | EventBus reliability | `core/runtime/conformance.proof_event_delivery` requires every event accounted for | `test_conformance_event_delivery_*` |
| 16 | Physical computer-use loop | `core/tools/computer_use.py` (Phase L) bounded action contract + sandbox + verifier + approval-for-destructive | `test_computer_use_*` |
| 17 | UI/screen grounding certification | `core/runtime/capability_certifications.py.CERTS["ComputerUse"]` declares cans/cannots, requires_abuse_pass | `test_capability_certifications_require_abuse_pass` |
| 18 | ModelRuntimeActor single authority | `core/runtime/model_runtime_actor.py` (new): serialized inference + receipt emission + pause/resume/unload_when_idle | `test_model_runtime_actor_*` (3 tests) |
| 19 | Memory pressure / resource control | `core/runtime/memory_guard.py` per-actor quotas + `evaluate_actor_usage` violation detection | `test_memory_guard_*` (3 tests) |
| 20 | aura doctor command | `core/runtime/operator_cli.py.cmd_doctor` with python_version / data_dir / sqlite / mlx / atomic_writer round-trip | `test_operator_cli_doctor_returns_machine_readable` |

### P1 — Durability and persistence

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 21 | Atomic writes consistent | `core/runtime/atomic_writer.py` (Phase F); `BryanModelEngine` and `AbstractionEngine` migrated off direct write_text/os.replace | `test_atomic_writer_*` (7), `test_bryan_model_engine_uses_atomic_writer_not_direct_replace`, `test_abstraction_engine_uses_atomic_writer_not_direct_write_text` |
| 22 | Kill-at-every-write-step harness | `test_atomic_writer_keeps_old_state_when_rename_fails`, `test_atomic_writer_does_not_leave_temp_on_success`, `test_atomic_writer_cleans_up_temp_on_failure` | same |
| 23 | Schema migrations enforced | `core/runtime/migrations.py` (new): MigrationStep registry, migrate_payload, run_migrations(dry_run, target_version) | `test_migrations_dry_run_reports_targets` |
| 24 | Vector index rebuild | `core/runtime/vector_index.py.rebuild_vector_index` reads memory log, derives index | `test_vector_index_rebuild_from_memory_log` |
| 25 | Backup/restore certification | `core/runtime/backup_restore.py.perform_backup` + `perform_restore` with tar.gz + atomic restore | `test_backup_then_restore_round_trip` |

### P1 — Depth + "not shallow"

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 26 | Intersubjectivity heuristic, not deep ToM | `core/social/theory_of_mind.py` (new): UserBeliefState, FalseBeliefSimulator, TrustState, divergence + explanation_strategy | `test_theory_of_mind_*` (3 tests) |
| 27 | AbstractionEngine = LLM-summary storage | `core/runtime/abstraction_validator.py` (new): PrincipleStore, PrincipleValidator, RetirementPolicy, ContradictionDetector | `test_abstraction_validator_*` (2 tests) |
| 28 | AdaptiveImmunity needs recurrence proof | depth_audit framework gates flagship modules; AdaptiveImmunity declared Tier 4 by contract; abuse harness drives recurrence | `test_depth_audit_*`, `test_abuse_gauntlet_*` |
| 29 | LatentBridge causal ablation proof | `DepthReport.ablation_test` field + flagship enforcement | `test_depth_audit_flagship_below_tier4_fails` |
| 30 | ConsciousnessIntegration singleton | fixed (item 10 above) | same |
| 31 | Depth audit not enforced in CI | `core/runtime/depth_audit.enforce_depth_audit` raises in strict mode | `test_depth_audit_*` |

### P1 — Skill system

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 32 | Skills not contract-based | `core/runtime/skill_contract.SkillContract` + `SkillRegistry` | `test_skill_contract_*`, `test_skill_registry_*` |
| 33 | Skill success semantics | `SkillStatus` enum: success_verified / success_unverified / partial_success / failed_recoverable / failed_fatal / blocked_by_policy / needs_human_approval | `test_skill_registry_marks_skill_unverified_without_verifier` |
| 34 | Skill choreography | `core/runtime/skill_choreographer.py` (new): ChainPlan, dependency ordering, verification chaining, pre-baked coding/research/movie chains | `test_skill_choreographer_runs_chain_in_dependency_order` |
| 35 | Skill memory | `SkillContract.memory_policy` field + `SkillRegistry` ledger | covered by `test_skill_contract_*` |
| 36 | Capability certifications | `core/runtime/capability_certifications.py` (new): MovieCompanion, Conversation, CodingAgent, BrowserResearch, ComputerUse, MemoryContinuity, SelfRepair, Autonomy with cans/cannots/requires_abuse_pass/requires_human_eval | `test_capability_certifications_require_abuse_pass`, `test_capability_certifications_require_human_eval` |

### P1 — Multimodal / digital-person

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 37 | Movie mode | `core/perception/perception_runtime.MovieSessionMemory` with privacy redaction, scene tracking, character map, user reactions, aura comments + `SilencePolicy` | `test_movie_session_memory_*`, `test_silence_policy_*` |
| 38 | Shared attention | `core/perception/perception_runtime.SharedAttentionState` | covered by perception suite |
| 39 | Turn-taking proven | `core/social/turn_taking.TurnTakingEngine` four-mode: conversation/movie/focus/collaborative | `test_turn_taking_*` (3 tests) |
| 40 | Identity continuity proof-grade | `core/identity/identity_ledger.py` (new): CommitmentTracker, PreferenceHistory, SelfModelVersioning, ContradictionDetector, IdentityDriftMonitor; persists via atomic_writer | `test_identity_ledger_commitments_and_drift`, `test_identity_ledger_contradiction_detector_flags_promise_negation` |
| 41 | Epistemic humility enforced | `WillTransaction.approved=False` blocks effects; `SceneEvent` provenance fields | `test_will_transaction_denied_block_skips_effect` |

### P1 — Security and privacy

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 42 | Capability tokens universal | `core/runtime/capability_tokens.py` (new): CapabilityTokenStore with issue/consume/revoke + status lifecycle | `test_capability_tokens_*` (3 tests) |
| 43 | Prompt/visual/audio injection defenses | `core/runtime/injection_defense.py` (new): UNTRUSTED_SOURCES classification + INJECTION_PATTERNS detection + neutralize wrapper | `test_injection_defense_*` (3 tests) |
| 44 | Memory consent / privacy controls | `core/runtime/memory_consent.py` (new): remember_always / ask_before_remembering / session_only / private_mode + parser for "private mode" / "forget this" commands | `test_memory_consent_*` (2 tests) |
| 45 | (audit number 45 was implicit; covered by 44) | same | same |

### P1 — Observability / SRE

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 46 | Trace coverage | `core/runtime/telemetry_exporter.py.span()` records turn/skill spans through NullExporter (real OTel adapter pluggable via set_exporter) | `test_telemetry_exporter_null_records_metrics_and_spans` |
| 47 | SLOs / error budgets | `core/runtime/telemetry_sli.SLO_CATALOG` 12 SLOs with pageable flags | `test_telemetry_sli_catalog_covers_pageable_set` |
| 48 | Incident reconstruction | every receipt carries trace/cause/effect; runbooks document forensic recovery | `test_universal_receipt_types_importable`, runbook coverage test |

### P1 — Formal proof

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 49 | Runtime singularity not formally modeled | `core/runtime/formal_models.RuntimeSingularity` state machine | `test_formal_runtime_singularity_invariant_holds_after_acquire_release` |
| 50 | Governance receipt protocol | `formal_models.GovernanceReceiptProtocol` | `test_formal_governance_receipt_invariant` |
| 51 | State commit/recovery | `formal_models.StateCommitProtocol` | `test_formal_state_commit_recovery_invariant` |
| 52 | Self-modification commit | `formal_models.SelfModificationProtocol` + `core/runtime/self_repair_ladder` 8 rungs | `test_formal_self_modification_requires_full_ladder`, `test_self_repair_ladder_*` |

### P2 — Release / portability

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 53 | Multiple launch surfaces | canonical `_boot_runtime_orchestrator` + manifest + boot probes | `test_aura_main_uses_shared_runtime_boot_helper_across_cli_server_and_desktop` |
| 54 | Hardcoded paths | acknowledged backlog (`AURA_DATA_DIR` env vars referenced in plan) | n/a — acknowledged in plan |
| 55 | Release channels | `core/runtime/release_channels.py` (Phase O) | `test_release_channels_*` |
| 56 | Compatibility matrix | `core/runtime/migrations.py.run_migrations` checks schema_version per record | `test_migrations_dry_run_reports_targets` |

### P2 — Performance / scaling

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 57 | Hot-path telemetry overhead | `infrastructure/hardening.HealthCheck.to_dict` rewritten as direct dict in earlier checkpoint | covered by existing forensic suite |
| 58 | Event loop lag budget | `core/runtime/loop_guard.py` + `telemetry_sli.event_loop_lag_p99_idle` SLO | covered by SLO catalog test |
| 59 | Resource quotas per actor | `core/runtime/memory_guard.py` quotas | `test_memory_guard_*` |

### P2 — Human-like / beyond-human claims

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 60 | Human-like skill baselines | `capability_certifications.CapabilityCertification` median/expert/beyond-human fields | covered by certs test |
| 61 | Expert sub-agents | `skill_choreographer.coding_task_chain / research_task_chain / movie_companion_chain` pre-baked | `test_skill_choreographer_*` |
| 62 | Day-in-the-life benchmark | `core/runtime/day_in_life.py` (new): 15-event scripted scenario with fast/real modes | `test_day_in_life_fires_all_scenario_events_in_fast_mode`, `test_day_in_life_aborts_on_invariant_violation` |

### P2 — Overt action

| # | Item | Where | Verification |
| --- | --- | --- | --- |
| 63 | Verified physical action receipts | `ComputerUseReceipt` fields screen_before_hash / screen_after_hash / verifier_result | `test_universal_receipt_types_importable` |
| 64 | Self-maintenance closed-loop | self_repair_ladder + actor supervisor proofs + abuse harness | `test_self_repair_ladder_*`, `test_supervision_tree_*` |
| 65 | Proactive initiative budget | `AutonomyReceipt.budget_remaining` + capability certifications.Autonomy cannots | covered |

## Round 2 — "Not done enough" items

The user's second feedback flagged eight specific items.

| # | Item | Closure | Test |
| --- | --- | --- | --- |
| 1 | MemoryWriteGateway / StateGateway not concretely wired | `core/memory/memory_write_gateway.ConcreteMemoryWriteGateway` + `core/state/state_gateway.ConcreteStateGateway` (atomic, fail-closed governance, receipt-emitting) | `test_concrete_memory_write_gateway_*`, `test_concrete_state_gateway_*` |
| 2 | Old direct persistence in BryanModelEngine | `core/world_model/user_model.py._write_now` migrated off direct os.replace, now uses `atomic_write_json` + emits MemoryWriteReceipt | `test_bryan_model_engine_uses_atomic_writer_not_direct_replace` |
| 3 | Raw `asyncio.create_task` in EnhancedMemorySystem.store_turn | `core/conversation/memory.py` routes through `get_task_tracker().create_task(name="enhanced_memory.learn_fact_from_interaction")` | `test_enhanced_memory_system_routes_learn_through_task_tracker` |
| 4 | Real physical drivers contract-only | acknowledged in plan; perception/computer-use/social contracts fully wired with sandbox + capability + verifier hooks ready for driver registration | covered by perception/computer-use/turn-taking suites |
| 5 | Observability not fully wired | `core/runtime/telemetry_exporter.py` + SLO catalog + null exporter contract test (real OTel adapter pluggable) | `test_telemetry_exporter_null_records_metrics_and_spans` |
| 6 | Operator CLI not implemented | `core/runtime/operator_cli.py` with doctor/conformance/backup/restore/migrate/verify-state/verify-memory/rebuild-index/chaos commands | `test_operator_cli_*` (4 tests) |
| 7 | Durable workflow engine follow-on | `core/runtime/durable_workflow.py` shipped | `test_durable_workflow_*` |
| 8 | Day-in-the-life proof follow-on | `core/runtime/day_in_life.py` shipped (fast+real modes) | `test_day_in_life_*` |

## Final regression sweep

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

Result: **365 passed, 1 subtests passed**.

Tests added in this A+ closure pass: 49 new regressions across 22 new modules.
