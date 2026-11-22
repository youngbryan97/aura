# Runbook: Shutdown hangs (F04)

<!-- reader-navigation:start -->
> Aura checks whether parts are making progress, keeps state that can be restored, and has procedures for recovery.
>
> A mistaken restart can interrupt work or leave two large processes competing for memory. If a reply takes a long time, first check whether its worker is still making progress. A stalled worker needs a different response.
>
> [Plain-language guide](../PLAIN_ENGLISH_GUIDE.md#reliability) · [Terms and examples](../VARIABLES_EXPLAINED.md#reliability) · [Evidence and related work](../DEMOS_AND_RELATED_WORK.md#runtime)
> [All docs](../DOCUMENTATION_INDEX.md) · [Glossary](../GLOSSARY.md) · [Worked formulas](../FORMULAS.md)
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
