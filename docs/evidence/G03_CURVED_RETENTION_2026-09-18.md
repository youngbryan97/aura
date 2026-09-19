# Curved retention boundaries and competing feasible steps

Two numerical counterexamples use the actual graph-score implementation.
Neither depends on a benchmark answer or a language-specific rule.

## Curved boundary

Let retained margin be `q.d - 0.9`, with initial `q = d = (1, 0)`.
The wrong comparison is `q1*d2 - q2*d1 - 0.5`. Its improving tangent moves
`q = (1, -t)` and `d = (1, t)`, so the retained margin becomes `0.1 - t^2`.
Every nonzero straight tangent step violates the floor. The old fitter
accepted no updates, although `q = (1.05, -0.3)` and `d = (1.05, 0.3)`
satisfy both required margins.

The fitter now tries bounded local corrections to violated curved faces.
Every corrected proposal is converted to float32, independently reevaluated
against every retained floor, and accepted only if the objective decreases.
This is a numerical search, not an infeasibility proof.

## Nearby boundary

A second example retains `x >= 0.1`, starting at `x = 0.100001`, and needs
`y - x - 0.5 >= 0.1`. The unprojected descent direction consumes the tiny
slack in `x`, forcing tiny updates in `y`. A projected alternative can keep
`x` unchanged while increasing `y`. Previously, accepting the first tiny
proposal prevented testing that alternative.

The fitter retains the first feasible proposal while following newly discovered
blocking-face projections. Each refinement adds at least one previously absent
face, so the finite constraint inventory bounds this search. It chooses the
lower-loss feasible result. Keeping
both proposals matters: blindly treating every slack face as binding was
itself a previously reproduced defect. Scalar and batched tests now repair
both examples, while incompatible-objective and slack-motion tests still pass.
The numerical policy is `working_face_deficit_v6`, with corresponding
logistic and fixed-step identities. Checkpoint identity includes the changed
algorithm source.

## Source evidence

The correction-only intermediate trial is retained at
`~/.aura/rlc-evidence/semantic-nonlinear-restoration-trial-20260918.json`.
Receipt: `8e6035bc0dc3dce1f462f0991ece39234e5cfe8adc55d3c39bfc025cfe11a9a5`.
Training stayed 7/8, validation stayed 6/8, and 125 retained comparisons were
still wrong or tied after 64 updates. This negative result exposed the
nearby-boundary problem; it is not a successful learning campaign.

The one-alternative intermediate trial is retained at
`~/.aura/rlc-evidence/semantic-competing-faces-trial-20260918.json`.
Receipt: `96709df3d8e59e0347cb6bcf633e9f6a2a0652376f09df9638fcab6a09a1ec4e`.
It repaired the training error (7/8 to 8/8) and reduced wrong retained
comparisons to four. Validation remained 6/8, including a decode refusal.
A three-variable test then reproduced a second blocking face revealed by the
first projection; the search now follows that face instead of stopping there.

The working-face trial completed with all 2,504 retained margins at or above
0.1 after 14 updates. Training improved 7/8 to 8/8, validation stayed 6/8,
and neither cohort had a refusal or a previously equivalent answer regress.
Its remaining blocker is incomplete operation-retention search. Receipt
`a5ab32339e201c1d01be63327e7f0818ea324ddabe798f14a19aa993540b195a` is retained at
`~/.aura/rlc-evidence/semantic-working-faces-trial-20260918.json`.

Eighty-one focused tests pass. Neither these numerical tests nor the source
trial establishes G03 closure, fresh transfer, or serving authority.
