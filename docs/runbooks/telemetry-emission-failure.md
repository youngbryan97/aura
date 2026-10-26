# Runbook: telemetry emission failure

<!-- reader-navigation:start -->
> **In plain English:** Observability records what the system did, how long each stage took, and where an assumption failed.
>
> **Why it matters:** A visible error or delay needs a traceable cause; otherwise a repair can change the wrong thing while hiding the real failure.
>
> **In everyday use:** When a reply arrives late, separate time spent waiting for input, reasoning, synthesis and playback before deciding which stage to optimise.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#observability) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#observability) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

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
