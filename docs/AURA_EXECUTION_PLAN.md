# Aura Execution Plan

## Mission

Production-harden Aura in checkpointed milestones, in priority order, with
regression tests for every bug fixed and a continuously updated execution
tracker.

## Source Documents

Requested by mission prompt:

- `AGENTS.md`
- `AURA_MASTER_SPEC.md`
- `docs/AURA_MASTER_SPEC.md`
- `docs/RUNTIME_INVARIANTS.md`
- `docs/PRODUCTION_HARDENING_PLAN.md`
- `docs/SKILL_CERTIFICATION_MATRIX.md`
- `docs/DEPTH_AUDIT.md`
- `docs/ABUSE_GAUNTLET.md`
- `docs/FORMAL_VERIFICATION_PLAN.md`
- `ARCHITECTURE.md`
- `HOW_IT_WORKS.md`
- `TESTING.md`

Currently present in repo:

- `ARCHITECTURE.md`
- `HOW_IT_WORKS.md`
- `TESTING.md`
- `specs/ARCHITECTURE.md`
- `specs/QUALITY_GATES.md`
- `specs/PERSONALITY.md`
- `audit/aura_critical_source_compilation.txt`

Missing from repo at time of execution start:

- `AGENTS.md`
- `AURA_MASTER_SPEC.md`
- `docs/AURA_MASTER_SPEC.md`
- `docs/RUNTIME_INVARIANTS.md`
- `docs/PRODUCTION_HARDENING_PLAN.md`
- `docs/SKILL_CERTIFICATION_MATRIX.md`
- `docs/DEPTH_AUDIT.md`
- `docs/ABUSE_GAUNTLET.md`
- `docs/FORMAL_VERIFICATION_PLAN.md`

The missing docs are treated as part of the hardening gap and are recorded in
the tracker/risk register rather than silently ignored.

## Execution Order

1. Phase A: Baseline / doctor / import / test discovery
2. Phase B: Critical runtime breakers
3. Phase C: Runtime singularity / boot / service manifest
4. Phase D: TaskSupervisor / ShutdownCoordinator
5. Phase E: Governance receipts / WillTransaction
6. Phase F: MemoryWriteGateway / StateGateway / atomic persistence
7. Phase G: EventBus / ActorSupervisor
8. Phase H: Self-repair validation ladder
9. Phase I: Conformance and abuse tests
10. Phase J: Depth audit and Tier 4/5 subsystem upgrades
11. Phase K: Skill contracts / verifiers / certification
12. Phase L: Multimodal / movie / digital-person capability layer
13. Phase M: Security / privacy / sandboxing
14. Phase N: Formal protocol models
15. Phase O: Release engineering and runbooks

## Milestone Strategy

Each milestone must:

- Leave the repo coherent.
- Update tests for the bugs fixed.
- Run the smallest relevant verification slice.
- Update `docs/AURA_EXECUTION_TRACKER.md` before moving on.

## Current Milestones

### Milestone A1

Baseline discovery and execution control plane.

Scope:

- Verify available spec/docs surface.
- Create execution plan/tracker/risk/test command docs.
- Map priority runtime-breaker files and related tests.
- Run narrow baseline commands for audited fault lines.

Exit criteria:

- Docs created.
- Baseline commands/results captured.
- Next patch set is scoped to the smallest critical runtime breaker cluster.

### Milestone B1

Critical runtime breaker patch set.

Initial target surface:

- `BaseSkill.safe_execute` false `ok: true` and sync blocking paths
- capability-engine false `ok: true`
- filesystem existence bypass in incoming logic
- `LocalPipeBus` loop/bootstrap and connection ownership hazards
- `SensoryGateActor` lifecycle/cleanup/path hijack hazards
- `GracefulShutdown` `sys.exit(0)` in async path
- `StateVault` degraded boot in strict mode
- production `agency_test` exception in file tool

Exit criteria:

- Targeted regressions added and green.
- Narrow import/boot/test slice green or exact blockers documented.

Status:

- Complete in this checkpoint.

### Milestone B2

Watchdog / launch-surface / supervisor hardening.

Target surface:

- watchdog duplicate-runtime ownership in `aura_main.py`
- launch-surface runtime singularity around reaper/port ownership
- supervisor heartbeat semantics in `core/supervisor/tree.py`
- stable reaper manifest ownership across launcher/process contexts
- 3D launcher stale timestamp runtime detection
- launcher/watchdog background task ownership in hot bootstrap paths

Exit criteria:

- watchdog does not boot a full Aura runtime itself
- launcher-only owners are the only processes allowed to reap ports / spawn reaper
- targeted actor/supervisor/launcher regressions are green

Status:

- Complete in this checkpoint.

### Milestone B3

Remaining critical runtime-breaker cleanup before Phase C.

Target surface:

- last legacy shared-connection `LocalPipeBus(connection=conn)` compatibility paths
- broader codebase `create_task` ownership sweep outside launcher/watchdog slices
- stricter canonical boot/service-manifest ownership across CLI/server/desktop surfaces
- additional strict-mode fail-closed probes for critical services beyond `StateVault`

Exit criteria:

- legacy shared-connection IPC fallback is removed or isolated behind explicit compatibility tests
- remaining high-risk background task surfaces are lifecycle-owned
- next runtime-singularity work can move into Phase C without reopening B-class regressions

Status:

- In progress in this checkpoint.
- Landed so far:
  - strict fail-closed boot behavior for critical `ResilientBoot` stages
  - websocket disconnect task ownership via the server task spawner
  - graceful shutdown signal bridge task ownership via the task tracker
  - canonical shared runtime boot helper across CLI/server/desktop
  - explicit task ownership for `ActorBus` telemetry and `LocalPipeBus` reader/dispatcher loops
  - explicit task ownership for `StateRepository` consumer startup/repair and the `AuraEventBus` Redis listener
  - lazy runtime initialization for `Scheduler` async primitives plus tracked scheduler main/task loops
  - tracked long-lived service loops for `ContinuousCognitionLoop`, `SessionGuardian`, and `SystemGovernor`
  - tracked actor-owned background loops for `StateVaultActor` and `SensoryGateActor`
  - tracked coordinator-side task ownership across `AutonomousConversationLoop`,
    `MessageCoordinator`, `MetabolicCoordinator`, and key
    `CognitiveCoordinator` background paths

## Deferred Backlog Additions

Requested by user and recorded for later priority phases, not ahead of runtime
invariants:

- Chrome Polish Phase:
  `core/runtime/chaos_engine.py`, fuzzing harness, telemetry/SLI dashboard,
  policy framework, memory safety guardrails
- PerceptionRuntime / governed capability layer
- social intelligence / movie mode / turn-taking layer
- computer-use realism, OCR/window detection, and governed desktop actions
- formal verification skeletons, property-based proofs, and TLA+/PlusCal models
