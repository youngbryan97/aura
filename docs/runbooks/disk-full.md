# Runbook: Disk Full

## Symptoms
- TODO: list visible signals (logs, metrics, UX) for this scenario

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
- aura doctor (when CLI ships)
- Conformance suite: `python -m pytest tests/test_server_runtime_hardening.py -q -k "conformance"`
- Atomic-write proof: `python -m pytest tests/test_server_runtime_hardening.py -q -k "atomic_writer"`

## Postmortem checklist
- Add regression test under tests/
- Update docs/AURA_RISK_REGISTER.md
- Update docs/AURA_EXECUTION_TRACKER.md if invariant gaps were discovered
