# Runbook: failure-lockdown escalation from expected backpressure

Covers [F17](../../KNOWN_FAILURE_MODES.md). **Fixed.**

A background job timed out because the foreground was using the model. That
is the system working. It got recorded as a critical service failure, and
the lockdown counter walked to 1.00.

Expected backpressure is not a degradation. There's a difference, and this
bug was what happens when the code doesn't encode it.

## Symptoms
- `unified_failure_lockdown_1.00` in the log.
- `Executive REJECTED` on memory writes, tool execution, self-modification.
- Existential-threat spikes.
- Crucially: **no actual subsystem is broken.** Everything that got blocked
  was blocked by the lockdown, not by its own failure.

## Diagnosis

A bounded background generation — memory consolidation, the dialectical
crucible — timed out while the foreground lane held the model. The timeout
was recorded as a degradation on a *fail-closed* subsystem. Modules on the
fail-closed list escalate warning-and-above to CRITICAL, so a plain
`TimeoutError` became a CRITICAL SERVICE FAILURE and drove the lockdown.

To confirm:

1. Find the originating record. If it is a `TimeoutError` from a background
   lane while the foreground was generating, this is F17.
2. Check whether the "failed" subsystem answers when you call it directly.
   Under F17 it will.

## Safe mitigation
- Restart clears the lockdown.
- Reduce concurrent background load so the foreground is not competing.

## Unsafe mitigation (last resort)
- **Do not remove a subsystem from the fail-closed list to stop the alarm.**
  The list is doing its job; the classification of the input was wrong. You
  would be disabling a real safety property to silence a symptom.

## Rollback
The fix is `core/runtime/backpressure.py`: expected backpressure is recorded
on a non-fail-closed channel with the policy disabled, and foreground yields
precede background generation. A recurrence means one of those regressed, or
a new call site is recording a load-shed timeout as a degradation.

## Verification
- Sustained load with background jobs running keeps `unified_failure_lockdown`
  at baseline.
- Timeouts under contention appear at info level, not as CRITICAL.

## Postmortem checklist
- New code that can time out under load must decide, explicitly, whether the
  timeout is backpressure or failure. Per `CLAUDE.md`: log expected
  backpressure at info, and only record a degradation when the condition is
  persistent or total.
- Add a regression test that drives foreground and background contention
  together and asserts the lockdown does not move.
