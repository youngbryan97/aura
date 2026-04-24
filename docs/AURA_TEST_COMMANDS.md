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
python -m pytest tests/test_launcher_polish_contract.py -q
python -m pytest tests/test_orchestrator_compatibility.py -q
python -m pytest tests/test_runtime_stability_edges.py -q
python -m py_compile \
  aura_main.py \
  core/actors/sensory_gate.py \
  core/bus/actor_bus.py \
  core/bus/local_pipe_bus.py \
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
