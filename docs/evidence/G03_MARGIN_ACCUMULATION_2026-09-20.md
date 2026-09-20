# Complete-graph margin accumulation

The full-source development fit exposed repeated rejection at retained
boundaries. A separate numerical reproduction found a defect in margin replay:
the gradient and batched evaluators added the fixed margin before cancelling
shared graph scores. With a fixed margin of 0.1 and identical +100 and -100
terms, both returned 0.09999999999999432. The scalar evaluator returned 0.1.
The exact supplied expression is 0.1, and its parameter gradient is zero.

All three evaluators now sum their complete signed score terms with `math.fsum`.
The retained floors, required margin, storage precision and strict acceptance
checks are unchanged. The change does not turn a near miss into a pass by
adding a tolerance. It avoids losing the fixed term during cancellation.

## Checks

The focused graph batch, constraint, minimum-change and bilinear geometry
suites pass 75 tests. New cases cover both term orders, positive and negative
scores at 100 and 1e16, exact zero gradients, and an independent repair beside
a constant cancelling witness. Both batched and scalar constrained fits reach
their original required margins. An invariant records the shared-score check.

The frozen source-expansion run at `997efa864` was stopped after its first
update repeatedly failed retention. Its last report reached 14 constraint
refinements and 506 projected faces, without an accepted update. The full
incumbent scan remains 755/764; no candidate or paired evaluation was produced.
The interruption landed inside the affine projection's QR decomposition.
The frozen source tree and incumbent checkpoint are unchanged.

The numerical reproduction does not establish that cancellation explains every
rejected update, nor does it close G03 or authorize a candidate. A separate
12-constraint curved-boundary reproduction also fails the existing sequential
restoration, despite independent feasible corrections. That repair follows.
