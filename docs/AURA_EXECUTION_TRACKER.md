# Aura Execution Tracker

## Current Phase

Phase B: Critical runtime breakers

## Current Milestone

Milestone B3: remaining IPC / task-ownership / strict boot cleanup before
Phase C.

## Files Changed

- `docs/AURA_EXECUTION_PLAN.md`
- `docs/AURA_EXECUTION_TRACKER.md`
- `docs/AURA_RISK_REGISTER.md`
- `docs/AURA_TEST_COMMANDS.md`
- `aura_main.py`
- `core/actors/sensory_gate.py`
- `core/bus/actor_bus.py`
- `core/bus/local_pipe_bus.py`
- `core/continuous_cognition.py`
- `core/event_bus.py`
- `core/graceful_shutdown.py`
- `core/guardians/governor.py`
- `core/ops/resilient_boot.py`
- `core/reaper.py`
- `core/resilience/sovereign_watchdog.py`
- `core/scheduler.py`
- `core/session_guardian.py`
- `core/state/state_repository.py`
- `core/state/vault.py`
- `core/supervisor/tree.py`
- `interface/websocket_manager.py`
- `scripts/one_off/launch_aura_3d.py`
- `tests/test_launcher_polish_contract.py`
- `tests/test_server_runtime_hardening.py`
- `tests/test_forensic_audit_regressions.py`
- `tests/test_resilient_boot_llm_stage.py`
- `tests/test_runtime_polish.py`
- `tests/test_time_resilience.py`

## Tests Added

- `test_reaper_manifest_uses_shared_env_override`
- `test_actor_health_gate_counts_only_distinct_miss_windows`
- `test_watchdog_start_uses_task_tracker_ownership`
- `test_watchdog_mode_remains_supervision_only`
- `test_aura_main_routes_bootstrap_background_tasks_through_task_tracker`
- `test_aura_main_uses_shared_runtime_boot_helper_across_cli_server_and_desktop`
- `test_3d_launcher_uses_runtime_lock_instead_of_stale_state_timestamp`
- `test_local_pipe_bus_rejects_legacy_shared_single_connection`
- `test_actor_bus_rejects_legacy_single_connection_transport`
- `test_local_pipe_bus_reader_tasks_are_task_tracked`
- `test_actor_bus_telemetry_loop_is_task_tracked`
- `test_event_bus_redis_listener_is_task_tracked`
- `test_state_repository_initialize_tracks_owner_consumer_task`
- `test_state_vault_actor_background_tasks_use_task_tracker`
- `test_sensory_gate_actor_background_tasks_use_task_tracker`
- `test_resilient_boot_strict_runtime_fails_closed_on_llm_stage_error`
- `test_resilient_boot_non_strict_runtime_degrades_on_llm_stage_error`
- `test_websocket_manager_uses_task_spawner_for_disconnect_on_overflow`
- `test_graceful_shutdown_signal_handlers_are_task_tracked`
- `test_scheduler_import_defers_asyncio_primitives_until_runtime`
- `test_scheduler_tracks_main_loop_and_registered_tasks`
- `test_continuous_cognition_loop_is_task_tracked`
- `test_session_guardian_monitor_loop_is_task_tracked`
- `test_system_governor_health_loop_is_task_tracked`

## Commands Run

1. Tracker/plan/risk/test-command reads for the current checkpoint
2. `test -f AGENTS.md && sed -n '1,220p' AGENTS.md || echo 'AGENTS.md missing'`
3. `git status --short`
4. `git diff --stat`
5. Targeted source inspection commands for:
   `aura_main.py`, `core/bus/local_pipe_bus.py`, `core/bus/actor_bus.py`,
   `core/event_bus.py`, `core/state/state_repository.py`,
   `core/scheduler.py`, `core/continuous_cognition.py`,
   `core/session_guardian.py`, `core/guardians/governor.py`,
   `core/supervisor/tree.py`, `core/reaper.py`,
   `core/ops/resilient_boot.py`, `interface/websocket_manager.py`,
   `scripts/one_off/launch_aura_3d.py`,
   `core/resilience/sovereign_watchdog.py`,
   `tests/test_launcher_polish_contract.py`,
   `tests/test_server_runtime_hardening.py`
6. `python -m pytest tests/test_launcher_polish_contract.py tests/test_time_resilience.py`
7. `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository or supervisor or actor_health_gate or reaper_manifest"`
8. `python -m pytest tests/test_orchestrator_compatibility.py -q`
9. `python -m pytest tests/test_runtime_stability_edges.py -q`
10. `python -m py_compile aura_main.py core/reaper.py core/supervisor/tree.py core/resilience/sovereign_watchdog.py scripts/one_off/launch_aura_3d.py tests/test_server_runtime_hardening.py tests/test_launcher_polish_contract.py tests/test_time_resilience.py`
11. `python -m py_compile core/bus/local_pipe_bus.py core/bus/actor_bus.py tests/test_server_runtime_hardening.py`
12. `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or actor_health_gate or reaper_manifest"`
13. `python -m py_compile core/ops/resilient_boot.py interface/websocket_manager.py tests/test_resilient_boot_llm_stage.py tests/test_runtime_polish.py`
14. `python -m pytest tests/test_resilient_boot_llm_stage.py tests/test_runtime_polish.py -k "resilient_boot or websocket_manager"`
15. `python -m py_compile core/graceful_shutdown.py tests/test_forensic_audit_regressions.py`
16. `python -m pytest tests/test_forensic_audit_regressions.py -q -k "graceful_shutdown_signal"`
17. `python -m py_compile core/state/state_repository.py core/event_bus.py tests/test_server_runtime_hardening.py`
18. `python -m pytest tests/test_server_runtime_hardening.py -q -k "event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`
19. `python -m py_compile core/scheduler.py tests/test_server_runtime_hardening.py`
20. `python -m pytest tests/test_server_runtime_hardening.py -q -k "scheduler or event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`
21. `python -m py_compile core/continuous_cognition.py core/session_guardian.py core/guardians/governor.py tests/test_server_runtime_hardening.py`
22. `python -m pytest tests/test_server_runtime_hardening.py -q -k "continuous_cognition_loop_is_task_tracked or session_guardian_monitor_loop_is_task_tracked or system_governor_health_loop_is_task_tracked or scheduler or event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`
23. `python -m pytest tests/test_launcher_polish_contract.py -q`
24. `python -m pytest tests/test_orchestrator_compatibility.py -q`
25. `python -m pytest tests/test_runtime_stability_edges.py -q`
26. `git status --short`
27. `git diff --stat`
28. `python -m py_compile core/state/vault.py core/actors/sensory_gate.py tests/test_server_runtime_hardening.py`
29. `python -m pytest tests/test_server_runtime_hardening.py -q -k "state_vault_actor_background_tasks_use_task_tracker or sensory_gate_actor_background_tasks_use_task_tracker or continuous_cognition_loop_is_task_tracked or session_guardian_monitor_loop_is_task_tracked or system_governor_health_loop_is_task_tracked or scheduler or event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`
30. `git status --short`
31. `git diff --stat`

## Pass / Fail Results

- `AGENTS.md` lookup: fail (`AGENTS.md` is missing from repo)
- current git diff verification before edits: pass
- focused runtime/launcher/supervisor slice:
  - `python -m pytest tests/test_launcher_polish_contract.py tests/test_time_resilience.py`:
    pass (`13 passed`)
  - `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository or supervisor or actor_health_gate or reaper_manifest"`:
    pass (`22 passed, 47 deselected`)
  - `python -m pytest tests/test_orchestrator_compatibility.py -q`:
    pass (`8 passed`)
  - `python -m pytest tests/test_runtime_stability_edges.py -q`:
    pass (`19 passed, 1 subtests passed`)
- focused B3 strict-boot/server-ownership slice:
  - `python -m pytest tests/test_resilient_boot_llm_stage.py tests/test_runtime_polish.py -k "resilient_boot or websocket_manager"`:
    pass (`6 passed, 8 deselected`)
- focused shutdown-ownership slice:
  - `python -m pytest tests/test_forensic_audit_regressions.py -q -k "graceful_shutdown_signal"`:
    pass (`2 passed, 37 deselected`)
- focused B3 shared-boot/bus/state/event slice:
  - `python -m pytest tests/test_launcher_polish_contract.py -q`:
    pass (`11 passed`)
  - `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or actor_health_gate or reaper_manifest"`:
    pass (`11 passed, 60 deselected`)
  - `python -m pytest tests/test_server_runtime_hardening.py -q -k "event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`:
    pass (`14 passed, 59 deselected`)
- focused B3 scheduler/service-loop slice:
  - `python -m pytest tests/test_server_runtime_hardening.py -q -k "scheduler or event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`:
    pass (`16 passed, 59 deselected`)
  - `python -m pytest tests/test_server_runtime_hardening.py -q -k "continuous_cognition_loop_is_task_tracked or session_guardian_monitor_loop_is_task_tracked or system_governor_health_loop_is_task_tracked or scheduler or event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`:
    pass (`19 passed, 59 deselected`)
  - `python -m pytest tests/test_server_runtime_hardening.py -q -k "state_vault_actor_background_tasks_use_task_tracker or sensory_gate_actor_background_tasks_use_task_tracker or continuous_cognition_loop_is_task_tracked or session_guardian_monitor_loop_is_task_tracked or system_governor_health_loop_is_task_tracked or scheduler or event_bus or state_repository_repair_runtime or state_repository_initialize_tracks_owner_consumer_task or local_pipe_bus or actor_bus or reaper_manifest or actor_health_gate"`:
    pass (`21 passed, 59 deselected`)
  - `python -m pytest tests/test_orchestrator_compatibility.py -q`:
    pass (`8 passed`)
  - `python -m pytest tests/test_runtime_stability_edges.py -q`:
    pass (`19 passed, 1 subtests passed`)
- `python -m py_compile ...` for all touched launcher/supervisor/reaper/bus/boot/server files:
  pass
- `python -m py_compile ...` for all touched state/event/scheduler/service-loop files:
  pass

## Unresolved Failures

1. Required mission docs are still missing from the repository:
   `AGENTS.md`, `AURA_MASTER_SPEC.md`, `docs/AURA_MASTER_SPEC.md`,
   `docs/RUNTIME_INVARIANTS.md`, `docs/PRODUCTION_HARDENING_PLAN.md`,
   `docs/SKILL_CERTIFICATION_MATRIX.md`, `docs/DEPTH_AUDIT.md`,
   `docs/ABUSE_GAUNTLET.md`, `docs/FORMAL_VERIFICATION_PLAN.md`.
2. Runtime singularity is improved but not yet fully canonicalized across every
   boot surface; CLI/server/desktop now share a canonical boot helper, but
   remaining launch/service-manifest ownership still needs tightening.
3. Codebase-wide background-task ownership is still incomplete outside the
   launcher/watchdog/server/shutdown/bus/state/event/scheduler/service-loop
   and actor-local slices touched so far.
4. The newly requested Chrome-polish / perception / social / formal-verification
   modules are recorded in the plan, but correctly deferred until earlier
   runtime-invariant phases are complete.

## Next Exact Task

Continue Milestone B3 by hardening the next coordinator/runtime loop cluster:
audit `core/conversation_loop.py`, `core/coordinators/message_coordinator.py`,
and adjacent orchestrator-side background reply/reflection task spawning so
those paths move under explicit lifecycle ownership.

## Next Exact Continuation Prompt

Continue Aura production hardening from `docs/AURA_EXECUTION_TRACKER.md`.
Milestone B3 is in progress. Continue with the next runtime-breaker slice:
harden the coordinator/runtime loop cluster in `core/conversation_loop.py`,
`core/coordinators/message_coordinator.py`, and adjacent orchestrator-side
background task spawning so those reply/reflection paths are lifecycle-owned.
Keep the tracker updated before any stop.

## Exact Stopping Point

Stopped after:

1. consolidating CLI/server/desktop boot through a shared runtime boot helper,
2. routing `LocalPipeBus` reader/dispatcher and `ActorBus` telemetry loops
   through the task tracker,
3. routing `StateRepository` consumer startup/repair and `AuraEventBus` Redis
   listener through the task tracker,
4. fixing `Scheduler` so async primitives are created lazily at runtime and its
   loops are task-tracked, and
5. routing long-lived `ContinuousCognitionLoop`, `SessionGuardian`, and
   `SystemGovernor` service loops through explicit task ownership.
6. routing actor-local `StateVaultActor` and `SensoryGateActor` background
   loops through tracked task ownership helpers with named tasks.

## Current Known Failures

- Missing requested mission docs listed in the prompt.
- Remaining coordinator/runtime ownership sweep is not yet complete.
- Broader launch-surface/service-manifest canonicalization is not yet complete.
- Additional strict-mode readiness probes for critical services beyond the
  current `ResilientBoot` critical-stage set are not yet implemented.

## Current Git Diff Summary

- Pre-existing dirty file not owned by this mission:
  `.aura/memfs/user.txt`
- Mission diff now includes:
  - canonical shared runtime boot helper across CLI/server/desktop
  - watchdog supervision-only mode
  - launcher-only reaper/port ownership
  - canonical reaper manifest path resolution
  - task-tracker ownership for launcher/watchdog hot-path tasks
  - task-tracker ownership for graceful-shutdown signal scheduling
  - supervisor missed-heartbeat window semantics and locking
  - lock-based 3D launcher runtime detection
  - removal of legacy shared single-connection `LocalPipeBus` compatibility
  - task-tracker ownership for `LocalPipeBus` reader/dispatcher loops
  - task-tracker ownership for `ActorBus` telemetry loop
  - task-tracker ownership for `StateRepository` consumer startup/repair
  - task-tracker ownership for `AuraEventBus` Redis listener
  - lazy runtime initialization plus task-tracked loops in `Scheduler`
  - task-tracker ownership for `ContinuousCognitionLoop`, `SessionGuardian`,
    and `SystemGovernor`
  - task-tracker ownership for actor-local `StateVaultActor` and
    `SensoryGateActor` background loops
  - strict fail-closed boot behavior for critical `ResilientBoot` stages
  - websocket disconnect task ownership via the server task spawner
  - graceful shutdown signal bridge ownership via the task tracker
  - focused launcher/supervisor/runtime regressions plus updated execution docs
