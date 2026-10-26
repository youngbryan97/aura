# Runbook: Self Repair Failed

<!-- reader-navigation:start -->
> **In plain English:** Reliability work detects failed or stalled components, preserves useful state and recovers through a controlled lifecycle.
>
> **Why it matters:** A user should not lose work because a slow component was mistaken for a dead one, or because a restart created another large process.
>
> **In everyday use:** If generation is progressing slowly, the system needs a different response from the one appropriate to a worker that has stopped making progress.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#reliability) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#reliability) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

## Symptoms
- `receipts.json[recent.self_repair]` shows entries whose `rungs_passed` list is shorter than expected (e.g. compile_fail or assertion_fail in the typed mutation evaluator).
- `~/.aura/data/mutation_quarantine/` has new entries: each contains `source.py`, `result.json` with `outcome != passed`, plus `stdout.log` and `stderr.log`.
- `audit_chain/info.json[verify]` may show a self_repair receipt with `rolled_back=true` immediately after a failed mutation.
- `logs/` contains `MutationOutcome.timeout` or `MutationOutcome.runtime_exception` traces.
- The same target file is being repeatedly proposed for repair and quarantined.

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
