# Runbook: Background task orphaning (F08)

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

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
