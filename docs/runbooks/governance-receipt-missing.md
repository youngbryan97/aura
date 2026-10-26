# Runbook: Governance Receipt Missing

<!-- reader-navigation:start -->
> **In plain English:** Governance checks whether a proposed action is allowed and records the decision and relevant evidence.
>
> **Why it matters:** The user needs meaningful control and a way to find out what was authorised, attempted and observed.
>
> **In everyday use:** Before editing a file or sending information outside the machine, the action must be checked against the applicable scope and recorded honestly.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#governance) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#governance) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

**Diagnostic scope:** bundle members and field names below are examples to
locate in the actual output, not a guaranteed schema on every revision.
Collectors can report missing services or errors. A separate CLI process may
observe its own container and PID rather than the resident runtime. Match
source/process identity and retained live logs before attributing a reading.
See [diagnostic collection](../../core/runtime/diagnostics_bundle.py).

## Symptoms
- `receipts.json[counts][governance]` has not advanced after consequential actions (tool executions, memory writes, state mutations).
- `receipts.json[recent.tool_execution]` entries have `governance_receipt_id=None`.
- `audit_chain/info.json[verify].ok` is true but `length` lags the count of effects in `receipts.json[counts]`.
- `logs/` shows `record_degradation('governance', ...)` entries.
- `gateway.json[registered]` may show `unified_will.ready=false`.

## Diagnosis
- Confirm AURA_STRICT_RUNTIME mode (env: AURA_STRICT_RUNTIME)
- Confirm the deployed revision and last rollback point: `make release-status`
- Identify the failing source owner and choose its named regression in an
  isolated checkout. A placeholder selector is not a test receipt; report the
  selected and passing counts. Set a separate `AURA_LOG_DIR`.
- Check ServiceManifest results in boot logs (`_enforce_service_manifest`)

## Safe mitigation
- Preserve the failure evidence. If rollback is justified, identify a verified
  source/state pair and follow [the recovery procedure](disaster-recovery.md)
  during an authorised maintenance window.
- If restart is necessary, obtain an authorised maintenance window, identify
  the resident/campaign owner and use the supported graceful lifecycle. The
  internal `_boot_runtime_orchestrator` helper is not a shell recovery command.
  Confirm one replacement process; do not load a second Cortex.
- Re-run conformance suite to confirm invariants hold post-mitigation

## Unsafe mitigation (last resort)
- Do not kill a PID merely because a lock file names it. Verify process
  identity and ownership first; preserve diagnostics and prefer graceful stop.
- Deleting the state vault destroys evidence and data. It is not ordinary
  recovery; use a verified backup or a scratch-copy repair first.

## Rollback
- Reverse to previous release channel (use `evaluate_release` outputs to compare gates)
- If self-repair patch caused regression, run `validate_patch` on the prior known-good source

## Verification
- `aura doctor --bundle` and inspect `bundle_manifest.json` plus the fields named in Symptoms above
- Conformance suite: `python -m pytest tests/test_server_runtime_hardening.py -q -k "conformance"`
- Atomic-write proof: `python -m pytest tests/test_server_runtime_hardening.py -q -k "atomic_writer"`

## Postmortem checklist
- Add regression test under tests/
- Update docs/AURA_RISK_REGISTER.md
- Update docs/AURA_EXECUTION_TRACKER.md if invariant gaps were discovered
