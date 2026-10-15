# Each local point fit finishes independently

The CPU profile on 32 generated pictures spent 0.92913 seconds of
1.23365 seconds in source-point observation, including 0.52459 seconds in
the local rigid patch fit. This was a generated rotating and translating
textured body on a contended host. It is not an original-game measurement.

The patch solver stopped only when every patch in its batch converged.
A difficult or mismatched patch kept converged patches in the same twelve
iterations. The new active set retires each patch at the existing
translation and rotation stopping tolerances. Singular fits leave the
active set as soon as their support is rejected. Every patch still gets
its final colour residual, and forward/backward checks retain their limits.
No identity or control threshold changes.

The new work-count test failed on the prior implementation: every sample
batch still had all three patches. It now observes smaller batches while
the difficult patch continues. Tests also compare a fit alone with that
same fit among unrelated patches, and keep the numerical claim registered.

## Paired generated-pixel measurement

`tools/measure_point_fit_cost.py` takes only the earlier `_refine` function
from a local Git revision. The remaining observer, input frames and point
budget stay current. Alternating runs compare the baseline at 6fc5ce05b
with this repair. Each run has forty generated source pictures.

| Order | Version | Median observation seconds | Total seconds | Retained point observations |
| --- | --- | ---: | ---: | ---: |
| 1 | Before | 0.035088479 | 1.389716960 | 882 |
| 2 | After | 0.027928500 | 1.106885127 | 895 |
| 3 | After | 0.027817313 | 1.101120957 | 895 |
| 4 | Before | 0.034611541 | 1.345776628 | 882 |
| 5 | Before | 0.037300812 | 1.474094412 | 882 |
| 6 | After | 0.027597042 | 1.116491127 | 895 |

The median of the three per-run medians fell from 35.088479 milliseconds
to 27.817313 milliseconds, a 20.7 percent reduction in this generated scene.
The retained-point counts differ, so the two algorithms are not claimed
bit-for-bit equivalent. Existing tests independently check known geometry,
colour replacement rejection, missing pixels and retained material.
These counts alone establish no increase in correct correspondences.

The focused perception and correction set has 255 passes in 41.09 seconds.
Smoke has 165 passes and one skip in 67.58 seconds. Ruff, changed-file
compilation, layering, dependency consistency, changed-method size and
writing checks passed. No GPU, browser or language model was used.

New raw CPU evidence is under this worktree's
`artifacts/flash-campaign/2026-10-15-perception-cost/`. It survives temporary
directory cleanup. The first exploratory profile attempted to save outside
the test runtime's owned state and was refused; no guard was disabled.
The repeated profile and paired measurement use checkout-owned paths.

The campaign plan retained at 6fc5ce05b has all 56 identities: 45 pending,
11 excluded (9 and the ten unopened holdouts), no task passes and no
presentation qualification. It was made with `--plan-only` and read no
original program. G09 and the resident runtime were not changed. Ordinary
live performance and original-game success remain unproved.
