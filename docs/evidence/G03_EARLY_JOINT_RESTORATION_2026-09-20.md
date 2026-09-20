# Restore omitted curved faces before expanding the working set

The minimum-change fitter previously deferred nonlinear restoration until all
violated faces entered its affine working set. The full-source diagnostic at
740f291e5 reached fourteen cut rounds without accepting its first update;
restoration remained unused despite measured floor violations.

The minimum-change path now attempts joint restoration on its first trial.
Remaining blockers are collected after restoration. Strict retained floors,
stored float32 replay and lower-loss acceptance remain unchanged.

A constructed twelve-face case previously needed twelve cut rounds. It now
needs zero, reaching identical stored margins: twelve retained margins near
0.1008623022 and the repaired comparison near 0.0943851321. That last margin
does not meet the requested 0.1 target and is not reported as doing so.

Applying early restoration to the older working-face optimizer caused four
regression failures. The change is restricted to minimum-change proposals;
the existing working-face tests remain in the regression suite. Seventy-four
tests pass across constraints, minimum-change fitting, checkpoints and
bilinear geometry. This is an optimizer reproduction, not a measured transfer
gain. G03 remains open pending candidate evaluation.
