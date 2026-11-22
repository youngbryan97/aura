# Runbook: telemetry emission failure

Covers [F14](../../KNOWN_FAILURE_MODES.md). Low severity — observability
gap, not a runtime fault.

A silent channel reads STALE, not nominal. That distinction is the reason
this is survivable: a missing metric cannot be mistaken for a good one.

## Symptoms
- Gaps in the dashboard.
- Telemetry health check failing.
- Channels reporting STALE.

## Diagnosis
1. Telemetry health check first — it names the failing channel.
2. Look up the channel in `core/fsw/telemetry_dictionary.py`. Every declared
   channel has an id, a unit, and limits.
3. Check the integrity block for telemetry limit violations, which are
   reported as transitions rather than repeated alarms.

## Safe mitigation
- Restart telemetry emission. Expect a data gap for the outage window; that
  gap is honest and should not be backfilled.

## Unsafe mitigation (last resort)
- Do not synthesize values to close the gap. A fabricated reading is worse
  than a missing one, and STALE exists precisely so you don't have to.

## Rollback
Not applicable.

## Verification
- The channel reports fresh values with its declared unit.
- No STALE entries for it in the next health report.

## Postmortem checklist
- New telemetry is a declared channel with an id, a unit, and limits. Ids are
  a contract — never reuse one.
