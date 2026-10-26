# Runbook: MLX worker-kill cold-lane cascade

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

Covers [F16](../../KNOWN_FAILURE_MODES.md). **Partially mitigated. The
architectural cause is open.** This is the honest daily-stability edge — do
not expect a clean fix below.

MLX cannot soft-cancel a running generation. So freeing a busy worker means
force-killing it, and killing it unloads ~18 GB of model. The kill *is* the
recovery. That is the whole problem.

## Symptoms
- A cluster of turns returns 503 or fails closed, then recovers on its own.
- Log: `Cortex generation exceeded inference-gate timeout … aborting`,
  followed by repeated `Loading model:`.
- Worker RSS drops to roughly zero, then climbs back to ~21 GB.

## Diagnosis

A foreground deep generation exceeds its budget, so the worker is killed. On
a host without headroom the reload races the next turn's timeout — which
kills the *reloading* worker and starts the load again. That is the cascade.

**Check RSS behaviour first, because it tells you which failure you have:**

| RSS pattern | Failure |
|---|---|
| Cycles 21 GB → ~1 GB → reload, bounded | **F16.** This runbook. |
| Grows monotonically without release | [F07 resource exhaustion](resource-exhaustion.md). Different problem. |

Then check free memory. Below roughly 25 GB free — other apps open — the
cascade becomes likely rather than possible.

## Safe mitigation
- **Close other memory-heavy apps.** Headroom is the actual lever here.
- Let it finish. It self-recovers and RSS stays bounded.
- Avoid issuing more turns during the reload; each one is another timeout
  that can restart the cycle.

## Unsafe mitigation (last resort)
- Restarting mid-load makes it worse: you pay the 20 GB load again from cold.
- Do not lower the inference-gate timeout to "fail faster." That increases
  the kill rate, which is the thing causing the cascade.

## Rollback
Three mitigations are in place; a recurrence means one regressed:

1. Background timeouts no longer kill the shared worker.
2. Respawn waits for memory reclaim.
3. Mid-load workers are not torn down.

## Verification
- A long deep generation completes without a `Loading model:` storm behind it.
- Worker RSS holds steady instead of cycling.

## Open architectural work

Stated plainly because the mitigations do not close it:

- A soft-cancel path into the MLX worker, **or** a persistent model server.
- More host RAM headroom removes the cascade entirely.

Until one of those lands, this failure mode is bounded and survivable, not
gone. Anyone quoting daily-stability numbers should say so.
