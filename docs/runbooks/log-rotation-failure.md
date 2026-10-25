# Runbook: log rotation failure

<!-- reader-navigation:start -->
> **In plain English:** Observability records what the system did, how long each stage took, and where an assumption failed.
>
> **Why it matters:** A visible error or delay needs a traceable cause; otherwise a repair can change the wrong thing while hiding the real failure.
>
> **In everyday use:** When a reply arrives late, separate time spent waiting for input, reasoning, synthesis and playback before deciding which stage to optimise.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#observability) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#observability) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

Covers [F13](../../KNOWN_FAILURE_MODES.md). Low severity — **no data loss**.
Logs stop writing; the runtime keeps running.

## Symptoms
- Log write errors; log files stop growing.
- Disk space monitor alerting.

## Diagnosis
1. `df -h` on the volume holding `~/.aura/logs/`.
2. Check permissions on the log directory.
3. Confirm which sink is failing. `AURA_LOG_DIR` redirects the sink — if it
   points somewhere unwritable, that is the whole bug.

## Safe mitigation
- Free disk space (see [disk-full.md](disk-full.md)).
- Fix directory permissions, then restart log rotation.

## Unsafe mitigation (last resort)
- Do not delete the live log the process holds open. Rotate or truncate it.

## Rollback
Not applicable.

## Verification
- New entries appearing in the current log file.
- Rotation produces a new file at the threshold.

## Postmortem checklist
- Anything test-like must set `AURA_LOG_DIR` so it never writes into the
  live instance's logs. A test that fills the live log directory shows up
  here as a rotation failure.
