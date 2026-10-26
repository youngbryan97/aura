# Runbook: Memory Corruption

<!-- reader-navigation:start -->
> **In plain English:** Resource management keeps model inference, background work and the operating system within the machine’s real capacity.
>
> **Why it matters:** A useful assistant must avoid exhausting the host or making the conversation unusable while doing background work.
>
> **In everyday use:** A second large model load can disrupt a working session even if each model fits alone. Scheduling and a measured memory budget matter.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#resources) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#resources) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#runtime) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

## Symptoms
- `audit_chain/info.json[verify].ok` is `false` with `content_hash mismatch` or `entry_hash mismatch` reasons.
- `receipts.json[counts]` numbers do not match the count of files on disk under `~/.aura/receipts/<kind>/`.
- `logs/` shows `read_json_envelope: schema_version` errors or `JSONDecodeError` on receipt reload.
- `memory.json[stats]` reports a mismatch between expected and actual record counts.
- New `aura doctor --bundle` runs report the same chain mismatches deterministically (not transient).

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
