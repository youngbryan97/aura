# Keep repaired faces in the restoration problem

The retained nonlinear fitter corrected only the faces violated by its latest
proposal. A correction could cross a face repaired one iteration earlier.
Those alternating corrections repeatedly exhausted restoration and expanded
the outer working set without an accepted update.

Restoration now accumulates encountered faces and recomputes their gradients
at each proposed point. It solves their requirements jointly. The original
floors and the full stored-precision acceptance check remain unchanged.
The affine storage solver supplies measured rounding reserves; restoration no
longer adds a fixed reserve that moves an exact affine optimum.

## Paired numerical replay

The immutable problem has SHA-256
`7303b5b8361583662baa309ea26bf992e064af7feb0039323d659d191a3ba8ae`.
It contains 25,038 comparisons, 286,731 parameters and 20 wrong comparisons.
The source capture used 25 mining examples and 764 source-retention examples.
No validation or test labels were used for fitting.

- Previous implementation: stopped after four unaccepted working-set
  expansions in 77.1 seconds. The last tiny trial violated 541 retained faces.
- Holding the operation-pointer block fixed: one accepted step, 20 wrong
  comparisons reduced to 10, no retained-floor regression, 16.9 seconds.
- Accumulated restoration, all blocks trainable: one accepted step, 20 wrong
  comparisons reduced to zero, no retained-floor regression, 23.3 seconds.
  Two restoration iterations, no outer working-set expansion. Loss decreased
  from 0.0002655025757400773 to 0.0000000977793089551145.

These times include loading the archived problem. The old replay was bounded
at four expansions; its termination does not prove infeasibility. The result
archive is `semantic-replay-capture-20260920/accumulated-no-interior-result.npz`
under `~/.aura/rlc-evidence/`.

Ninety-seven focused tests pass, including four regressions for intersecting
restoration faces, stored margins, analytic minimum displacement and replay.
The first accumulated version passed the large replay but moved an affine
optimum unnecessarily; the existing tests caught that and the fixed reserve
was removed before acceptance.

Another 58 subspace, bilinear-geometry, joint-training and operation-pointer
tests pass. Smoke: 164 passed, one skipped. Lint, compile, governance,
layering and writing gates pass.

This closes the reproduced restoration-cycle defect. It does not close G03:
the frozen comparisons must still be tested by fresh decoding and new
competitor search. No candidate has been promoted to serving.
