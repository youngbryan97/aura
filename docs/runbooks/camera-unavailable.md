# Runbook: Camera Unavailable

<!-- reader-navigation:start -->
> **In plain English:** Visual control connects what is seen with an identified object, its motion and actions that change what happens next.
>
> **Why it matters:** A system needs to keep tracking the same object and account for delay, rather than react to whichever pixels look convenient in one frame.
>
> **In everyday use:** In a moving game, a key can arrive after the picture it was chosen from. The controller must learn how its actions affect the identified body.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#vision) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#vision) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#agents) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

**Diagnostic scope:** bundle members and field names below are examples to
locate in the actual output, not a guaranteed schema on every revision.
Collectors can report missing services or errors. A separate CLI process may
observe its own container and PID rather than the resident runtime. Match
source/process identity and retained live logs before attributing a reading.
See [diagnostic collection](../../core/runtime/diagnostics_bundle.py).

## Symptoms
- `models.json[sensors][camera].status` is `unavailable` or `permission_denied`.
- `health.json[services]` shows the vision pipeline degraded.
- `logs/` contains `AVCaptureDevice`, `v4l2`, or `media-permission` errors.
- Recent `receipts.json[recent.tool_execution]` shows vision tools returning `verifier_result=false` with `output_digest=""`.
- macOS only: `tccutil reset Camera com.aura.app` has not yet been run after a system update.

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
