# Runbook: Self Repair Failed

## Symptoms
- `receipts.json[recent.self_repair]` shows entries whose `rungs_passed` list is shorter than expected (e.g. compile_fail or assertion_fail in the typed mutation evaluator).
- `~/.aura/data/mutation_quarantine/` has new entries: each contains `source.py`, `result.json` with `outcome != passed`, plus `stdout.log` and `stderr.log`.
- `audit_chain/info.json[verify]` may show a self_repair receipt with `rolled_back=true` immediately after a failed mutation.
- `logs/` contains `MutationOutcome.timeout` or `MutationOutcome.runtime_exception` traces.
- The same target file is being repeatedly proposed for repair and quarantined.

## Diagnosis
- Confirm AURA_STRICT_RUNTIME mode (env: AURA_STRICT_RUNTIME)
- Confirm release channel (`aura status` once available)
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
