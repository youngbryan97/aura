# Runbook: log rotation failure

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
