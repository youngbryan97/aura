# Runbook: Aura Will Not Boot

## Symptoms
- `aura doctor` (without `--bundle`) returns `ok=false` with one or more failed pre-boot checks (`python_version`, `data_dir_writable`, `sqlite_available`, `atomic_writer_round_trip`).
- `~/.aura/locks/orchestrator.lock` exists and points to a PID that is no longer running.
- No `health.json` is produced; `aura doctor --bundle` exits with the bundle still containing collector `_error.txt` files for `health` and `gateway`.
- `logs/` contains a fatal traceback near the top of the most recent log.
- The orchestrator process is absent from `ps aux | grep aura_main`.

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
