# Runbook: Microphone Unavailable

<!-- reader-navigation:start -->
> **In plain English:** Conversation support turns a request into a reply and, when enabled, into speech. Streaming lets delivery begin before every later stage finishes.
>
> **Why it matters:** The person should receive a timely, coherent answer to the current request without hearing a stale or unapproved fragment.
>
> **In everyday use:** When you interrupt a spoken reply, the old reply should stop owning the conversation and the new turn should take priority.
>
> [Example and metaphor](../PLAIN_ENGLISH_GUIDE.md#conversation) · [Variables and practical meanings](../VARIABLES_EXPLAINED.md#conversation) · [Demos and related work](../DEMOS_AND_RELATED_WORK.md#interaction) · [Document map](../DOCUMENTATION_INDEX.md)
<!-- reader-navigation:end -->

## Symptoms
- `models.json[sensors][microphone].status` is `unavailable` or `permission_denied`.
- `logs/` contains `PortAudio`, `ALSA`, or `coreaudio` errors.
- `health.json[services]` shows the audio capture pipeline degraded.
- macOS only: microphone permission was revoked from System Settings → Privacy → Microphone, and the Aura process is no longer listed.
- Recent `receipts.json[recent.tool_execution]` for audio tools returns `output_digest=""`.

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
