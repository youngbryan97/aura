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
