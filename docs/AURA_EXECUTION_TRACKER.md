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
- `core/bus/actor_bus.py`
- `core/bus/local_pipe_bus.py`
- `core/graceful_shutdown.py`
- `core/ops/resilient_boot.py`
- `core/reaper.py`
- `core/resilience/sovereign_watchdog.py`
- `core/supervisor/tree.py`
- `interface/websocket_manager.py`
- `scripts/one_off/launch_aura_3d.py`
- `tests/test_launcher_polish_contract.py`
- `tests/test_resilient_boot_llm_stage.py`
- `tests/test_runtime_polish.py`
- `tests/test_server_runtime_hardening.py`
- `tests/test_forensic_audit_regressions.py`
- `tests/test_time_resilience.py`

## Tests Added

- `test_reaper_manifest_uses_shared_env_override`
- `test_actor_health_gate_counts_only_distinct_miss_windows`
- `test_watchdog_start_uses_task_tracker_ownership`
- `test_watchdog_mode_remains_supervision_only`
- `test_aura_main_routes_bootstrap_background_tasks_through_task_tracker`
- `test_3d_launcher_uses_runtime_lock_instead_of_stale_state_timestamp`
- `test_local_pipe_bus_rejects_legacy_shared_single_connection`
- `test_actor_bus_rejects_legacy_single_connection_transport`
- `test_resilient_boot_strict_runtime_fails_closed_on_llm_stage_error`
- `test_resilient_boot_non_strict_runtime_degrades_on_llm_stage_error`
- `test_websocket_manager_uses_task_spawner_for_disconnect_on_overflow`
- `test_graceful_shutdown_signal_handlers_are_task_tracked`

## Commands Run

1. Tracker/plan/risk/test-command reads for the current checkpoint
2. `test -f AGENTS.md && sed -n '1,220p' AGENTS.md || echo 'AGENTS.md missing'`
3. Targeted source inspection commands for:
   `aura_main.py`, `core/supervisor/tree.py`, `core/utils/task_tracker.py`,
   `core/reaper.py`, `core/ops/resilient_boot.py`,
   `core/bus/local_pipe_bus.py`, `core/bus/actor_bus.py`,
   `interface/websocket_manager.py`,
   `scripts/one_off/launch_aura_3d.py`,
   `core/resilience/sovereign_watchdog.py`
4. `git status --short && git diff --stat`
5. `python -m pytest tests/test_launcher_polish_contract.py tests/test_time_resilience.py`
6. `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository or supervisor or actor_health_gate or reaper_manifest"`
7. `python -m pytest tests/test_orchestrator_compatibility.py -q`
8. `python -m pytest tests/test_runtime_stability_edges.py -q`
9. `python -m py_compile aura_main.py core/reaper.py core/supervisor/tree.py core/resilience/sovereign_watchdog.py scripts/one_off/launch_aura_3d.py tests/test_server_runtime_hardening.py tests/test_launcher_polish_contract.py tests/test_time_resilience.py`
10. `python -m py_compile core/bus/local_pipe_bus.py core/bus/actor_bus.py tests/test_server_runtime_hardening.py`
11. `python -m py_compile core/ops/resilient_boot.py interface/websocket_manager.py tests/test_resilient_boot_llm_stage.py tests/test_runtime_polish.py`
12. `python -m pytest tests/test_resilient_boot_llm_stage.py tests/test_runtime_polish.py -k "resilient_boot or websocket_manager"`
13. `python -m py_compile core/graceful_shutdown.py tests/test_forensic_audit_regressions.py`
14. `python -m pytest tests/test_forensic_audit_regressions.py -q -k "graceful_shutdown_signal"`
15. `git status --short`
16. `git diff --stat`

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
- `python -m py_compile ...` for all touched launcher/supervisor/reaper/bus/boot/server files:
  pass

## Unresolved Failures

1. Required mission docs are still missing from the repository:
   `AGENTS.md`, `AURA_MASTER_SPEC.md`, `docs/AURA_MASTER_SPEC.md`,
   `docs/RUNTIME_INVARIANTS.md`, `docs/PRODUCTION_HARDENING_PLAN.md`,
   `docs/SKILL_CERTIFICATION_MATRIX.md`, `docs/DEPTH_AUDIT.md`,
   `docs/ABUSE_GAUNTLET.md`, `docs/FORMAL_VERIFICATION_PLAN.md`.
2. Runtime singularity is improved but not yet fully canonicalized across every
   boot surface; CLI/server/desktop ownership still needs a stricter shared
   boot/service-manifest contract.
3. Codebase-wide background-task ownership is still incomplete outside the
   launcher/watchdog/server/shutdown slices touched so far.
4. The newly requested Chrome-polish / perception / social / formal-verification
   modules are recorded in the plan, but correctly deferred until earlier
   runtime-invariant phases are complete.

## Next Exact Task

Continue Milestone B3 by auditing the next high-risk `asyncio.create_task`
cluster outside the launcher/server surfaces and by tightening canonical boot
ownership between CLI/server/desktop paths so readiness and service manifests
are derived from one shared authority.

## Next Exact Continuation Prompt

Continue Aura production hardening from `docs/AURA_EXECUTION_TRACKER.md`.
Milestone B3 is in progress. Continue with the next runtime-breaker slice:
audit the next high-risk raw `asyncio.create_task` cluster outside the
launcher/server surfaces, and tighten canonical boot/service-manifest ownership
across CLI/server/desktop paths. Keep the tracker updated before any stop.

## Exact Stopping Point

Stopped after:

1. completing the watchdog / launcher / supervisor hardening slice,
2. removing the last legacy shared single-connection `LocalPipeBus`
   compatibility path, and
3. landing the first B3 strict-runtime slice:
   strict fail-closed `ResilientBoot` handling for critical stages plus
   websocket disconnect task ownership through the server task spawner.
4. extending lifecycle ownership to the graceful-shutdown signal bridge.

## Current Known Failures

- Missing requested mission docs listed in the prompt.
- Broader launch-surface canonicalization is not yet complete.
- Broader codebase `create_task` ownership sweep is not yet complete.
- Additional strict-mode readiness probes for critical services beyond the
  current `ResilientBoot` critical-stage set are not yet implemented.

## Current Git Diff Summary

- Pre-existing dirty file not owned by this mission:
  `.aura/memfs/user.txt`
- Mission diff now includes:
  - watchdog supervision-only mode
  - launcher-only reaper/port ownership
  - canonical reaper manifest path resolution
- task-tracker ownership for launcher/watchdog hot-path tasks
- task-tracker ownership for graceful-shutdown signal scheduling
  - supervisor missed-heartbeat window semantics and locking
  - lock-based 3D launcher runtime detection
  - removal of legacy shared single-connection `LocalPipeBus` compatibility
  - strict fail-closed boot behavior for critical `ResilientBoot` stages
  - websocket disconnect task ownership via the server task spawner
  - graceful shutdown signal bridge ownership via the task tracker
  - focused launcher/supervisor/runtime regressions plus updated execution docs
