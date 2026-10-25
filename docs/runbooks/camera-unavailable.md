# Runbook: Camera Unavailable

<!-- reader-navigation:start -->
> **In plain English:** Visual control connects what is seen with an identified object, its motion and actions that change what happens next.
>
> **Why it matters:** A system needs to keep tracking the same object and account for delay, rather than react to whichever pixels look convenient in one frame.
>
> **In everyday use:** In a moving game, a key can arrive after the picture it was chosen from. The controller must learn how its actions affect the identified body.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#vision) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#vision) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#agents) · [Document map](../DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

## Symptoms
- `models.json[sensors][camera].status` is `unavailable` or `permission_denied`.
- `health.json[services]` shows the vision pipeline degraded.
- `logs/` contains `AVCaptureDevice`, `v4l2`, or `media-permission` errors.
- Recent `receipts.json[recent.tool_execution]` shows vision tools returning `verifier_result=false` with `output_digest=""`.
- macOS only: `tccutil reset Camera com.aura.app` has not yet been run after a system update.

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
