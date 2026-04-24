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

Actor/runtime lifecycle follow-through.

Initial target surface:

- remaining actor/supervisor lifecycle hazards in `core/supervisor/tree.py`
- watchdog / duplicate-runtime ownership checks
- stricter runtime singularity and canonical boot ownership
- remaining unowned background task hotspots outside the B1 slice
- direct shared-connection legacy IPC callers that still bypass split pipe pairs

Exit criteria:

- supervised actor IPC paths use explicit owned transports end-to-end
- strict runtime ownership failures fail closed
- targeted actor/supervisor regressions are green
