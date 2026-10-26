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

## Symptoms
- `receipts.json[counts][governance]` has not advanced after consequential actions (tool executions, memory writes, state mutations).
- `receipts.json[recent.tool_execution]` entries have `governance_receipt_id=None`.
- `audit_chain/info.json[verify].ok` is true but `length` lags the count of effects in `receipts.json[counts]`.
- `logs/` shows `record_degradation('governance', ...)` entries.
- `gateway.json[registered]` may show `unified_will.ready=false`.

## Diagnosis
- Confirm AURA_STRICT_RUNTIME mode (env: AURA_STRICT_RUNTIME)
- Confirm the deployed revision and last rollback point: `make release-status`
- Run `python -m pytest tests/test_server_runtime_hardening.py -q -k "<scenario>"`
- Check ServiceManifest results in boot logs (`_enforce_service_manifest`)

## Safe mitigation
- Roll back to last known-good checkpoint (per checkpoint-restore-failed runbook if needed)
- Restart through canonical `_boot_runtime_orchestrator` path
- Re-run conformance suite to confirm invariants hold post-mitigation

## Unsafe mitigation (last resort)
- Manual process kill via PID file at `~/.aura/locks/orchestrator.lock`
- Wipe state vault: only after explicit user confirmation. State will be lost.

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
