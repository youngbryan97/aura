# Point resolution and calibration coverage, October 14, 2026

The preceding browser run stored no qualified direction samples despite
212 current pose observations. Its binding replay produced nine direction
pairs; all exceeded the existing 0.12-radian estimated-error limit. The
point tracker measures a bounded source grid and publishes coordinates in
working pixels. Its resolution term had remained a quarter of a working
pixel, independent of that mapping.

`PointMotion` now records the tracked grid's cell size in working units.
The default source tracker derives it from the actual bounded tracking grid,
using the larger axis scale. It does not borrow the original image's finer
resolution after reducing that image to its pixel budget. `PointPose` keeps
that size, and angle and position estimates map the quarter-cell term with
the coordinates. Retained and bridged poses keep the larger size when two
receipts differ. Unknown resolution retains one working cell; invalid
resolution supplies no pose. Optional trackers retain that conservative
default here.

This corrects units in an RMS estimate. The estimate still does not bound
systematic correspondence error or drift. No heading threshold changed.
Tests compare the same geometry in different coordinate units, exercise
actual source tracking, check its bounded-grid mapping, and reject invalid
resolution. A registered canary protects the source mapping.

The first combined selection passed 189 tests and failed the existing
texture-831 stopped-turn test. Recorded pairs agreed on their direction
offset; the last 16 samples spanned 0.511 radians, below the required pi/6.
Nearby new readings had displaced the earlier observations that supplied
the needed angular range. This was an observation-memory failure, not an
instruction to lower the angular requirement.

Direction calibration now retains measured angular coverage within its
existing 16-sample limit. On replacement it always retains the newest pair,
including a contradiction. Among removals that preserve the most angular
coverage it removes the older, less precise redundant reading. The agreement
test still checks every retained pair. A new body or reference resets the
memory. Tests cover repeated nearby readings, an unrelated spinning
animation, and a latest contradiction that revokes the relation.

The coverage repair restored the stopped-turn test. An initial validation
registration used the wrong function scope; that was fixed before commit.
The final focused selection passed 195 tests in 24.88 seconds. It includes
the existing coupled and uncoupled pixel tests and the material, lifetime,
rotation, source-grid and calibration suites. The 16 new registration and
precision tests also passed separately; these overlap the final selection.

- Falsifying pair history: `/private/tmp/codex-steer/pixel-resolution-falsification-1014/direction-samples.json`
- Binding replay: `/private/tmp/codex-steer/material-and-trail-original9-1014/binding-replay.json`

Repository gates and another browser measurement remain required. This
change has no demonstrated Flash-game win or ordinary native-app proof.
Presentation readiness remains open.

The final gate run passed Ruff, writing, changed-method size, layering,
dependency consistency, compile, lint and smoke. Global size debt remains
194 lines above the aggregate budget and the existing response-generation
class remains at 22 methods against 20. Governance reports 123 existing
regressions. No changed module is a listed governance offender and no
baseline changed. Log: `/private/tmp/codex-steer/point-unit-gates-1014.log`.
