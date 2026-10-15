# Bounded cursor inspection, 15 October 2026

The original case 3 run at `a73fc355f` ended in 82.09 seconds after three
moves, with no win. The final screenshot shows the first room. The report's
`last_seen` still contains collectible instructions, so it cannot describe
that final picture. The raw report, final image and ordinary page-input
records are under
`artifacts/flash-campaign/2026-10-15-case3-current-corners/`.

Current-corner queries did find local cursor hits in the room. Their log
entries appear at 13:25:52.258, 13:25:53.551 and 13:25:55.263. Earlier in the
same run, one read exhausted eight seconds during extra inspection and used
its retained, pixel-checked reading. None of those facts proves the task was
completed or that corners improved the whole run.

The existing finer sweep checked its time before each group, then awaited
all of that group's queries. A physical fallback could therefore run past
the reading deadline and lose the measurements already completed. The new
outer bound applies to all cursor inspection, including finer passes. Its
two-second deadline cancels the remaining inspection. Completed physical
queries remain available as coordinate proposals if they include both a hit
and a negative inside the requested region. Synthetic batch answers cannot
fill unread positions. Changed page addresses or viewport geometry discard
the partial result. External cancellation propagates.

A partial result is not a census of controls. It retains actual sampled
coordinates, and normal delivery still reacquires the cursor affordance.
No body identity, task role or completion follows from these samples.

The focused run passed 60 tests, including interrupted physical sampling,
uniform cursors, changed coordinate frames, cancellation and unchecked batch
answers. Smoke passed 165 with one skipped. The initial pure canary used
exact float equality for a normalized midpoint; it now checks the numerical
result within `1e-12`. The measured hit coordinates remain unchanged.

Ruff, syntax compilation, layering, dependency consistency and writing also
passed. No baseline was raised. This is CPU component evidence; it does not
establish an original-game win or native-app readiness.
