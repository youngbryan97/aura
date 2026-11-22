# Runbook: lock contention and deadlock

Covers [F12](../../KNOWN_FAILURE_MODES.md).

Lockdep finds ABBA deadlocks *without the deadlock happening*. It only sees
locks it wraps. That second sentence is the one that matters during an
incident: a stall in un-instrumented code is invisible to the detector.

## Symptoms
- A request stalls until a watchdog releases it.
- Stall traces in `data/error_logs/stalls/`.
- Lockdep splats in `runtime_health_report()["integrity"]`.

## Diagnosis

1. Check the integrity block. `runtime_health_report()["integrity"]` carries
   lockdep splats and PSI alongside the taint register.
2. A **splat** means a lock-order violation was observed — declared ranks did
   not match observed acquisition order. That names the pair for you.
3. **No splat but a real stall** means the offending lock is not instrumented.
   Find the blocking frame in the stall trace, then wrap it.

Ranks live in `LockRank` (`core/runtime/lockdep.py`). Acquiring downward
through ranks is legal; acquiring upward is the violation.

## Safe mitigation
- The watchdog releases stale locks past threshold; most stalls clear alone.
- Reduce concurrency on the contended subsystem while you diagnose.

## Unsafe mitigation (last resort)
- Do not raise the watchdog threshold to stop the alarm. That converts a
  visible stall into an invisible one.

## Rollback
Not applicable. The fix is either a corrected lock order or an added rank.

## Verification
- `python -m pytest tests/ -k lockdep -q`
- No new splats in the integrity block under the load that triggered it.

## Postmortem checklist
- Use `checked_lock` / `checked_async_lock` rather than raw `threading.Lock`
  or `asyncio.Lock`. Adopt an existing lock with `instrument(name)`.
- An un-instrumented lock is a blind spot, not a safe lock. If this incident
  involved one, wrapping it *is* the fix.
- Three locks are sanctioned as blocking-by-design. If yours is genuinely one
  of those, say so explicitly rather than leaving it unranked.
