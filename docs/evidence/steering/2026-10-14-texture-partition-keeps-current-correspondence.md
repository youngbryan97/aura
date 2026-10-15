# Texture partition keeps current correspondence — 14 October 2026

The preceding continuous-trial run preserved 117 ordinary observations
inside its first 24 physical presses but assigned no body. Its material
split is recorded in `2026-10-14-continuous-trials-expose-a-material-split.md`.
The first joined outline followed a 0.268130064-second capture gap.

Point tracking and material matching already accept current named
correspondences through 0.35 seconds. The texture partition rejected the
same correspondence after 0.25 seconds, before testing its current pixels.
That prevented a measured small part from being separated before the
joined outline replaced its template.

The strong texture path now uses the point tracker's named 0.35-second
limit. Exact preceding/current observation times, birth checks, unique
point names, full-rank geometry, matching RGB and support shares remain
required. Weak colour partitioning retains its shorter interval. No
causal identity threshold or game-specific rule changed.

## Reproduction and tests

The direct joined-mask regression passed at 0.249 and 0.25 seconds before
the repair and failed at 0.268130064 and 0.35 seconds. All four pass after
the repair. A correspondence beyond 0.35 seconds remains rejected.
Stale, reborn, collinear, duplicate, unsupported and changed-texture cases
are checked again at the longer measured interval.

An actual-pixel fixture introduces a fixed vertical painted line and a
textured moving patch after the initial blank view, then moves the patch
across and away from the line. The ordinary tracker retains one entity
and its material position at both 0.1 and 0.268130064-second cadence.
A bounded ablation restoring only the old 0.25-second eligibility fails
at x=84: the moving patch becomes a tall joined component under a new
number. This ablation changes no production file.

The separate control-attribution fixture uses the actual pixel path,
independent contrasting controls and two texture seeds at both cadences.
It requires a confirmed body throughout the final four control intervals,
one entity number and the existing repeated-trial, effect and statistic
requirements. It supplies no exact state to the observer.

The targeted texture, correspondence, rotation, point and attribution set
passed 112 tests in 18.55 seconds. Ruff, writing, changed-method size,
layering, dependency consistency, compilation, lint and smoke passed;
smoke reports 165 passed and one skipped in 45.95 seconds. The existing
two size failures and 123 governance regressions remain unchanged.

The source method/bytecode work remains separate and uncommitted at this
checkpoint. These fixture results do not establish a current original-game
body, a win, native-app prompt behavior or presentation readiness.
