# Runbook: Movie Mode Broken

<!-- reader-navigation:start -->
> **In plain English:** Conversation support turns a request into a reply and, when enabled, into speech. Streaming lets delivery begin before every later stage finishes.
>
> **Why it matters:** The person should receive a timely, coherent answer to the current request without hearing a stale or unapproved fragment.
>
> **In everyday use:** When you interrupt a spoken reply, the old reply should stop owning the conversation and the new turn should take priority.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#conversation) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#conversation) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#interaction) · [Document map](../DOCUMENTATION_INDEX.md)
> [Glossary](../GLOSSARY.md) · [Formulas with worked examples](../FORMULAS.md)
<!-- reader-navigation:end -->

**Diagnostic scope:** bundle members and field names below are examples to
locate in the actual output, not a guaranteed schema on every revision.
Collectors can report missing services or errors. A separate CLI process may
observe its own container and PID rather than the resident runtime. Match
source/process identity and retained live logs before attributing a reading.
See [diagnostic collection](../../core/runtime/diagnostics_bundle.py).

## Symptoms
- `receipts.json[recent.computer_use]` shows `verifier_result=false` for screen-driven actions.
- `models.json[screen_vision].status` is degraded or `failed`.
- `logs/` contains frame-capture errors (`screencapture`, `CGDisplay`, or `Quartz` complaints).
- `health.json[services]` shows the perception pipeline degraded but the orchestrator is otherwise healthy.
- `tasks.json` lists movie-mode loops that complete instantly (no frames captured).

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
