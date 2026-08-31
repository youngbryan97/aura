# Live Φ measurement — state, cause, and the remaining wiring

Written 2026-08-04 so this can be picked up cold. Everything below is measured,
not inferred; commit hashes are on `main`.

## The claim being repaired

Prior Φ values — `φ_s mean 0.253` among them — are **retracted** as quantitative
evidence of integration. The old estimator assigned substantial φ to a system
constructed to be memoryless, and at reachable history lengths could rank it
ABOVE a genuinely coupled ring:

| n transitions | memoryless φ |
|---|---|
| 100 | 0.19 |
| 600 | 0.60 |
| 2000 | 0.91 |
| 6000 | 1.00 |

The history buffer holds 2000. There is no sample size this runtime reaches at
which raw φ over a 256-state TPM means what it appears to mean.

## What is fixed (commits c570beb29, e87d7356e, 9ba35e1a8)

1. **A null.** `PhiCore.measure_phi_null` permutes the timeline of ONE SIDE of
   the MIP — destroying cross-partition dependence while preserving each side's
   internal dynamics. A global shuffle was tried first and is not enough: two
   independently predictable halves still clear it.
   Synthetic separation: memoryless `0.000`, independent halves `0.049`,
   coupled ring `0.563`. `INTEGRATION_FRACTION_FLOOR = 0.10` is derived from
   that middle number, not chosen.
2. **Grounding-aware selection.** `compute_full_kernel_selection` picks by
   grounding tier first and magnitude second, and carries every candidate that
   could not run WITH its reason. `max(phi_s)` across four estimators over four
   different substrates was a category error.
3. **The live path runs it.** `compute_phi` (called by `closed_loop`) used to
   end at the 16-node spectral result or the affective 8-node fallback — both
   STATE_SUMMARY, neither null-corrected. It now returns the selection winner.
4. **The residual sample is no longer gated on steering firing.** It sat inside
   `if composite is not None:` in the steering hook, so any stand-down starved
   the activation-grounded complex.

## Why there is still no live number — the actual blocker

`_maybe_record_phi_residual` (`core/consciousness/affective_steering.py`, line
1481) resolves PhiCore with an **in-process** lookup:

```python
if not ServiceContainer.has("phi_core"):
    return
```

Generation does not run in that process. `core/brain/llm/mlx_client.py:12608`:

```python
p = ctx.Process(
    target=_mlx_worker_loop,
    args=(self.model_path, self._req_q, self._res_q, self.device,
          self._substrate_mem, self._steering_active, ...),
    daemon=True,
)
```

The steering hook lives in the **MLX worker subprocess**; PhiCore is registered
in the **main runtime**. They have never shared a process, so the lookup returns
False on every token and `_grassmann_state_history` stays empty.

Observed on a boot with three hooks installed and seven real cortex generations:

```
PhiCore is reporting a state_summary measurement because better-grounded
estimators could not run: residual_stream_grassmann
(insufficient_history:0/50 grassmann transitions)
```

Zero. Not "not enough yet" — none, ever. **This is the whole reason no
activation-grounded live Φ has ever existed.**

## The remaining fix, and why this shape

Do **not** thread samples through the generation response payload — that edits
the hot path and serialises per token.

The precedent is already there: `_substrate_mem` is shared memory passed
parent→worker carrying substrate state for steering. Mirror it in reverse.

- The Grassmann encoder reduces a ~5120-d residual vector to an **8-bit state
  integer**. One byte per sample. A fixed-size shared ring plus a monotonic
  write counter is sufficient — single writer (worker), single reader (parent),
  no lock needed if the reader tolerates missing a wrapped-past entry.
- Worker side: run `GrassmannResidualComplex.observe(vec)` where the activations
  already are, and write the resulting int.
- Parent side: `PhiCore` drains entries newer than its last read index into
  `_grassmann_state_history` / `_grassmann_state_visits`.
- Sampling stays bounded by `AURA_PHI_RESIDUAL_SAMPLE_EVERY` (default 32) and
  single-token decode only — prefill is not a thought moment and materialising
  a whole sequence off the GPU is what caused the 58–82s first-token stalls.

Entry points: `_maybe_record_phi_residual` (worker side),
`PhiCore._record_grassmann_residual` (parent side), and the `ctx.Process(...)`
arg list above for the channel itself.

## How to verify it worked

1. Boot headless, drive ~6 substantive turns.
2. `grep "grassmann transitions" <log>` should show a rising count, not `0/50`.
3. Past 50, `compute_phi` returns an `activation_geometry` winner with
   `is_best_grounded=True`, and `_maybe_attach_null` fills
   `integration_fraction` / `null_p_value` on its 300s interval.
4. Publish with `PhiResult.provenance()` — grounding, estimator identity, node
   count, population, and the null — never the bare scalar.

A live value is only citable as evidence when `integration_is_significant` is
true AND `null_surrogates >= 2`. Below that the honest report is *unmeasured*,
which is where the system stands today.

## Related

- `core/consciousness/phi_grounding.py` — tiers and selection
- `tests/test_phi_satisfies_the_iit_axioms.py` — the axiom battery and the bias
  measurement that forced all of this
- `core/organism/model_validation.py` — `Evidence` on `Claim`; this claim is
  `RETRACTED` until the above lands
