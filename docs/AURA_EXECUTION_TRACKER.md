# Aura Execution Tracker

## Current Phase

Phase B: Critical runtime breakers

## Current Milestone

Milestone B1: patch the first critical runtime-breaker cluster and leave the
repo in a coherent, tested state.

## Files Changed

- `docs/AURA_EXECUTION_PLAN.md`
- `docs/AURA_EXECUTION_TRACKER.md`
- `docs/AURA_RISK_REGISTER.md`
- `docs/AURA_TEST_COMMANDS.md`
- `infrastructure/base_skill.py`
- `core/skills/base_skill.py`
- `core/capability_engine.py`
- `core/orchestrator/mixins/incoming_logic.py`
- `core/graceful_shutdown.py`
- `core/skills/file_operation.py`
- `core/orchestrator/mixins/boot/boot_resilience.py`
- `core/bus/local_pipe_bus.py`
- `core/bus/actor_bus.py`
- `core/supervisor/tree.py`
- `core/actors/sensory_gate.py`
- `core/state/vault.py`
- `core/orchestrator/main.py`
- `tests/test_forensic_audit_regressions.py`
- `tests/test_orchestrator_compatibility.py`
- `tests/test_server_runtime_hardening.py`

## Tests Added

- `test_core_base_skill_preserves_error_dict_without_forcing_ok_true`
- `test_legacy_base_skill_uses_to_thread_for_sync_execute_and_preserves_errors`
- `test_filesystem_reality_shortcut_is_disabled_for_user_facing_requests`
- `test_graceful_shutdown_signal_path_does_not_raise_system_exit`
- `test_file_operation_no_longer_allows_desktop_agency_test_escape`
- `test_sensory_gate_run_always_closes_browser_and_bus`
- `test_local_pipe_bus_start_requires_running_event_loop`
- `test_local_pipe_bus_stop_closes_shared_connection_once`
- `test_local_pipe_bus_stop_closes_connection_pairs_independently`
- `test_actor_bus_rejects_none_transport_without_registering_actor`
- `test_start_state_vault_actor_strict_runtime_fails_when_handshake_never_succeeds`
- `test_start_state_vault_actor_fallback_ping_supports_split_pipe_pairs`

## Commands Run

1. `pwd && rg --files -g 'AGENTS.md' -g 'AURA_MASTER_SPEC.md' -g 'ARCHITECTURE.md' -g 'HOW_IT_WORKS.md' -g 'TESTING.md' -g 'docs/**'`
2. `git status --short`
3. `ls -la`
4. `rg --files | rg '(^|/)(AGENTS\\.md|AURA_MASTER_SPEC\\.md|RUNTIME_INVARIANTS\\.md|PRODUCTION_HARDENING_PLAN\\.md|SKILL_CERTIFICATION_MATRIX\\.md|DEPTH_AUDIT\\.md|ABUSE_GAUNTLET\\.md|FORMAL_VERIFICATION_PLAN\\.md|AURA_EXECUTION_PLAN\\.md|AURA_EXECUTION_TRACKER\\.md|AURA_RISK_REGISTER\\.md|AURA_TEST_COMMANDS\\.md)$'`
5. `find .. -maxdepth 3 \\( -name 'AGENTS.md' -o -name 'AURA_MASTER_SPEC.md' -o -name 'RUNTIME_INVARIANTS.md' -o -name 'PRODUCTION_HARDENING_PLAN.md' -o -name 'SKILL_CERTIFICATION_MATRIX.md' -o -name 'DEPTH_AUDIT.md' -o -name 'ABUSE_GAUNTLET.md' -o -name 'FORMAL_VERIFICATION_PLAN.md' \\)`
6. `find specs -maxdepth 2 -type f | sort`
7. `find audit -maxdepth 3 -type f | sort`
8. `sed -n '1,220p' ARCHITECTURE.md`
9. `sed -n '1,240p' HOW_IT_WORKS.md`
10. `sed -n '1,260p' TESTING.md`
11. `sed -n '1,240p' specs/QUALITY_GATES.md`
12. Targeted source/test inspection commands for:
`infrastructure/base_skill.py`, `core/skills/base_skill.py`,
`core/capability_engine.py`, `core/bus/local_pipe_bus.py`,
`core/state/vault.py`, `core/actors/sensory_gate.py`,
`core/graceful_shutdown.py`, `core/orchestrator/mixins/incoming_logic.py`,
`core/skills/file_operation.py`, `core/orchestrator/mixins/boot/boot_resilience.py`,
and related tests.
13. `python -m pytest tests/test_orchestrator_compatibility.py -q`
14. `python -m pytest tests/test_forensic_audit_regressions.py -q`
15. `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus"`
16. `python -m pytest tests/test_runtime_stability_edges.py -q`
17. `python -m pytest tests/test_controlled_complexity_runtime.py -q`
18. `python -m pytest tests/test_effect_closure.py -q`
19. `python -m pytest tests/test_skill_surface_contracts.py -q -k "safe_execute"`
20. `python -m py_compile infrastructure/base_skill.py core/skills/base_skill.py core/capability_engine.py core/orchestrator/mixins/incoming_logic.py core/graceful_shutdown.py core/skills/file_operation.py core/orchestrator/mixins/boot/boot_resilience.py core/bus/local_pipe_bus.py core/bus/actor_bus.py core/actors/sensory_gate.py core/state/vault.py core/supervisor/tree.py core/orchestrator/main.py`
21. `git status --short`
22. `git diff --stat`

## Pass / Fail Results

- `pwd` and repo discovery: pass
- Requested master/docs spec lookup: fail, files are missing from repo
- Source/read mapping for current milestone: pass
- Baseline runtime fault surfaces identified in live code: pass
- `tests/test_orchestrator_compatibility.py -q`: pass (`8 passed`)
- `tests/test_forensic_audit_regressions.py -q`: pass (`38 passed`)
- `tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus"`:
  pass (`6 passed, 60 deselected`)
- `tests/test_runtime_stability_edges.py -q`: pass (`19 passed, 1 subtests passed`)
- `tests/test_controlled_complexity_runtime.py -q`: pass (`11 passed`)
- `tests/test_effect_closure.py -q`: pass (`9 passed`)
- `tests/test_skill_surface_contracts.py -q -k "safe_execute"`:
  pass (`56 passed, 9 deselected`)
- `python -m py_compile ...`: pass

## Unresolved Failures

1. Required mission docs are missing from the repository.
2. Remaining runtime/lifecycle audit items still open outside this milestone:
   watchdog duplicate-runtime ownership, 3D launcher stale lock heuristic,
   broader supervisor invariants, remaining unowned background tasks, and
   full runtime singularity across launch surfaces.
3. Legacy single-connection `LocalPipeBus(connection=conn)` compatibility still
   exists for non-upgraded callers; supervised actor paths were moved to split
   pipe pairs, but the legacy fallback surface still needs elimination.

## Next Exact Task

Start Milestone B2: continue the actor/boot/runtime-singularity cluster by
auditing watchdog ownership, launcher duplication, supervisor health semantics,
and the remaining unowned background task surfaces.

## Next Exact Continuation Prompt

Continue Aura production hardening from `docs/AURA_EXECUTION_TRACKER.md`.
Milestone B1 is complete. Begin Milestone B2 and focus on the remaining
actor/runtime lifecycle audit items: watchdog duplicate runtime ownership,
launcher/runtime singularity, supervisor health semantics, remaining
`asyncio.create_task` ownership gaps, and removing the last legacy shared-conn
`LocalPipeBus` compatibility paths. Keep the tracker updated before any stop.

## Exact Stopping Point

Stopped after completing Milestone B1 and rerunning the focused runtime slices.
The repo is coherent, the new regressions are green, and the next work item is
the B2 actor/runtime lifecycle cluster.

## Current Known Failures

- Missing requested spec docs listed in mission prompt.
- Remaining launch/runtime singularity issues are not yet addressed.
- Remaining watchdog and actor-supervision proof harnesses are not yet added.
- Remaining codebase-wide `create_task` ownership sweep is not yet complete.

## Current Git Diff Summary

- Pre-existing dirty file not owned by this mission:
  `.aura/memfs/user.txt`
- Mission diff now includes docs plus the B1 runtime-breaker patch set across
  skill execution, governance bypass removal, shutdown semantics, file tool
  policy, actor IPC/runtime ownership, and focused regressions.
