# Runbook: Shutdown hangs (F04)

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

**Fault:** F04 — graceful shutdown does not complete (blocked async task,
hung worker, deadlocked service).

## Symptoms

- The process outlives the 12s graceful-shutdown budget after quit.
- `data/error_logs/stalls/` gains a loop-wedge report from the stall
  watchdog during the shutdown window.
- Launchd/desktop wrapper eventually escalates to SIGKILL.

## Automated mitigation

`core/ops/graceful_shutdown.py` bounds the whole sequence (12s budget) and
`core/resilience/stall_watchdog.py` forces exit if the loop wedges.
Target MTTR: 15s.

## Manual diagnosis

1. Find what refused to die: the shutdown sequence logs each phase; the
   last phase logged before silence is the culprit.
2. Cross-check `data/error_logs/crash/` for a faulthandler dump — the
   thread stacks show exactly which await/lock never returned.
3. Usual suspects: an on-loop sync write (the async-write-lane ratchet
   exists because one fsync froze the loop for 20 minutes), a worker
   process not acknowledging terminate, or a third-party client without
   a close timeout.

## Escalation

If SIGKILL was needed, treat the next boot as a dirty-shutdown recovery
(see `dirty-shutdown-recovery.md`) and verify SQLite integrity on boot
passed in the launch log.
