# Aura Execution Tracker

## Current Phase

Phase B: Critical runtime breakers

## Current Milestone

Milestone B2: watchdog / launch-surface / supervisor hardening.

## Files Changed

- `docs/AURA_EXECUTION_PLAN.md`
- `docs/AURA_EXECUTION_TRACKER.md`
- `docs/AURA_RISK_REGISTER.md`
- `docs/AURA_TEST_COMMANDS.md`
- `aura_main.py`
- `core/bus/actor_bus.py`
- `core/bus/local_pipe_bus.py`
- `core/reaper.py`
- `core/resilience/sovereign_watchdog.py`
- `core/supervisor/tree.py`
- `scripts/one_off/launch_aura_3d.py`
- `tests/test_launcher_polish_contract.py`
- `tests/test_server_runtime_hardening.py`
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

## Commands Run

1. `sed -n '1,220p' docs/AURA_EXECUTION_TRACKER.md`
2. `sed -n '1,220p' docs/AURA_EXECUTION_PLAN.md`
3. `sed -n '1,220p' docs/AURA_RISK_REGISTER.md`
4. `sed -n '1,220p' docs/AURA_TEST_COMMANDS.md`
5. `test -f AGENTS.md && sed -n '1,220p' AGENTS.md || echo 'AGENTS.md missing'`
6. Targeted source inspection commands for:
   `aura_main.py`, `core/supervisor/tree.py`, `core/utils/task_tracker.py`,
   `core/reaper.py`, `scripts/one_off/launch_aura_3d.py`,
   `core/resilience/sovereign_watchdog.py`,
   `core/bus/local_pipe_bus.py`, and `core/bus/actor_bus.py`
7. `git status --short && git diff --stat`
8. `python -m pytest tests/test_orchestrator_compatibility.py -q tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository or supervisor"`
9. `python -m pytest tests/test_launcher_polish_contract.py -q tests/test_time_resilience.py -q`
10. `python -m pytest tests/test_launcher_polish_contract.py tests/test_time_resilience.py`
11. `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository or supervisor or actor_health_gate or reaper_manifest"`
12. `python -m pytest tests/test_orchestrator_compatibility.py -q`
13. `python -m pytest tests/test_runtime_stability_edges.py -q`
14. `python -m py_compile aura_main.py core/reaper.py core/supervisor/tree.py core/resilience/sovereign_watchdog.py scripts/one_off/launch_aura_3d.py tests/test_server_runtime_hardening.py tests/test_launcher_polish_contract.py tests/test_time_resilience.py`
15. `python -m py_compile core/bus/local_pipe_bus.py core/bus/actor_bus.py tests/test_server_runtime_hardening.py`
16. `git status --short`
17. `git diff --stat`

## Pass / Fail Results

- `AGENTS.md` lookup: fail (`AGENTS.md` is missing from repo)
- current git diff verification: pass (only pre-existing non-mission dirty file before edits was `.aura/memfs/user.txt`)
- current failing-test check for the B2 slice: pass (no failures observed in the focused runtime/launcher slices)
- `python -m pytest tests/test_launcher_polish_contract.py tests/test_time_resilience.py`:
  pass (`13 passed`)
- `python -m pytest tests/test_server_runtime_hardening.py -q -k "local_pipe_bus or actor_bus or state_repository or supervisor or actor_health_gate or reaper_manifest"`:
  pass (`22 passed, 47 deselected`)
- `python -m pytest tests/test_orchestrator_compatibility.py -q`:
  pass (`8 passed`)
- `python -m pytest tests/test_runtime_stability_edges.py -q`:
  pass (`19 passed, 1 subtests passed`)
- `python -m py_compile ...` for touched launcher/supervisor/reaper/bus files:
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
   launcher/watchdog slice.
4. The newly requested Chrome-polish / perception / social / formal-verification
   modules are recorded in the plan, but correctly deferred until earlier
   runtime-invariant phases are complete.

## Next Exact Task

Start Milestone B3: remove remaining high-risk runtime breaker surfaces before
Phase C by auditing the broader `asyncio.create_task` spread, tightening
canonical boot/service-manifest ownership across CLI/server/desktop paths, and
extending strict fail-closed readiness checks beyond `StateVault`.

## Next Exact Continuation Prompt

Continue Aura production hardening from `docs/AURA_EXECUTION_TRACKER.md`.
Milestone B2 is complete. Begin Milestone B3 and focus on the remaining
critical runtime-breaker surfaces: the broader `asyncio.create_task` ownership
sweep outside launcher/watchdog, canonical boot/service-manifest ownership
across CLI/server/desktop surfaces, and additional strict fail-closed readiness
checks for critical services beyond `StateVault`. Keep the tracker updated
before any stop.

## Exact Stopping Point

Stopped after completing the watchdog / launcher / supervisor hardening slice
and an immediate B3 follow-up that removes the last legacy shared single-
connection `LocalPipeBus` compatibility path. The repo is coherent and the
focused launcher/supervisor/runtime regression slices are green.

## Current Known Failures

- Missing requested mission docs listed in the prompt.
- Broader launch-surface canonicalization is not yet complete.
- Broader codebase `create_task` ownership sweep is not yet complete.
- Additional strict-mode readiness probes for critical services beyond
  `StateVault` are not yet implemented.

## Current Git Diff Summary

- Pre-existing dirty file not owned by this mission:
  `.aura/memfs/user.txt`
- Mission diff now includes:
  - watchdog supervision-only mode
  - launcher-only reaper/port ownership
  - canonical reaper manifest path resolution
  - task-tracker ownership for launcher/watchdog hot-path tasks
  - supervisor missed-heartbeat window semantics and locking
  - lock-based 3D launcher runtime detection
  - removal of legacy shared single-connection `LocalPipeBus` compatibility
  - focused launcher/supervisor/runtime regressions plus updated execution docs
