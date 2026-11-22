# Runbook: launch-provenance `ready:false` on source drift

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

Covers [F18](../../KNOWN_FAILURE_MODES.md). **Expected in development.**

Read this one first: nothing is broken. A signed `Aura.app` pins the exact
commit and workspace hash it was built for. You changed the code. The check
noticed. That is tamper detection succeeding, not a fault.

## Symptoms
- `ready:false` with a `launch_provenance` blocker.
- `boot_phase: launch_provenance_failed`.
- Issues named `commit_sha_mismatch` or `workspace_state_sha256_mismatch`.
- She is fully conversational the whole time, via the `degraded_ready` path.

## Diagnosis

If you have edited the checkout since the app was built and signed, you are
done — this is F18. Every launch of an actively developed checkout does it.

The one case worth a second look: `ready:false` on a checkout you have **not**
touched. Then the drift is real and you want to know what changed. Compare
the pinned commit against `git rev-parse HEAD` and check for stray writes
into the workspace.

## Safe mitigation

Pick whichever fits what you're doing:

- **Actively developing** — launch via `launch_aura.sh`, which does not
  require provenance.
- **Want the signed path back** — rebuild and re-sign the app to re-pin it
  to the current commit and workspace hash.

## Unsafe mitigation (last resort)
- **Do not disable the provenance check to make the blocker go away.** It is
  the only thing that distinguishes "I edited this" from "something else
  edited this." Turning it off costs you the signal permanently and buys a
  green flag that means nothing.

## Rollback
Not applicable — no failure to roll back from.

## Verification
- After a rebuild and re-sign: `ready:true`, no `launch_provenance` blocker.
- Via `launch_aura.sh`: boots and serves without the provenance phase.

## Postmortem checklist
- Only escalate if provenance fails on an untouched checkout. That is a real
  integrity question and belongs with the source-body proprioception
  machinery, which reports boot-over-boot diffs of her own code.
