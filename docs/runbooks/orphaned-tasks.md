# Runbook: Background task orphaning (F08)

**Fault:** F08 — a task's creator dies or forgets it, leaving background
work running with no owner (wasted resources, stale state writes).

## Symptoms

- Rising task counts in the reaper's periodic census log lines.
- CPU/GPU busy while the system is nominally idle.
- Degradation records from `reaper` or task-tracker subsystems.

## Automated mitigation

`core/reaper.py` runs an orphan census and cancels tasks whose owners are
gone. Target MTTR: 60s.

## Manual diagnosis

1. `aura doctor --bundle` → `tasks.json` names every tracked task with its
   done and cancelled state, and `health.json` carries the degradation
   records that say which subsystem is spawning them.
2. Check the reaper log lines for which task names it keeps cancelling —
   a task that gets re-orphaned every cycle means its spawner loops.
3. For GPU-holding orphans, confirm the inference queue drained; a stuck
   generation is a worker problem (see `worker-crash.md`), not a task
   problem.

## Escalation

Recurring orphans from the same call site are a code defect: the spawner
must own its task handle and cancel on teardown. File it against the
spawning module — do not widen the reaper's kill list as a workaround.
