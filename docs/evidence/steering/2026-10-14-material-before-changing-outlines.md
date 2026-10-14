# Material correspondence before changing outlines, October 14, 2026

The source-eye diagnostic captured 21 material-name transfers in assigned
game 9 at `42e1db888`. Sixteen had a rigid fit from consecutive points.
Thirteen of those had an infinite outline-match cost. Examples changed
between a small moving patch and a tall joined outline. The same material
could continue while the outline's centre or aspect rejected its match.
These observations identify a correspondence defect; they do not identify
the controlled actor.

The motion eye now associates current material before its ordinary outline
matches. A previous measured point pose must belong to the object's current
lifetime and last capture. Continuing marks must still belong to that old
support and current foreground support, fit a spread-out proper rigid
transform, and retain the required share. Both the prior object and current
proposal need exactly one eligible match. Missing, stale, deformed,
unsupported and ambiguous evidence cannot take this path. The original
object and its material reference remain in place. No trial history is
copied to another number and no control identity is established here.

The focused suites passed 151 tests in 18.84 seconds. Two textures were
followed from actual pixels through disappearance of an attached strip.
The ordinary outline match rejected their changed aspect; the material
association kept the original object and measured reference displacement.
Other tests exercise stale clocks, lifetime changes, duplicate point names,
deformation, collinear marks, insufficient support and ambiguous matches.
A registered canary protects distinguishing material geometry.

A diagnostic replay of the recorded old objects, point receipts and current
observed supports retained 9 of the 21 transfers. That replay uses observed
objects rather than the complete original proposal set, so it is a partial
counterexample check, not an app-path performance measurement or a win.

The first source-audit attempt produced no source records because the eye
opened in the main process and the diagnostic hook targeted only the child.
Its owned process group was stopped after 61.684 seconds. The corrected
hook only records measurements in either process; it changes no choosing
or input. That run completed in 105.017 seconds with an unread result and
no win. Its instrumentation cost prevents a timing comparison.

- Source records: `/private/tmp/codex-steer/source-eye-original9-retry-1014/source-eye.jsonl`
- Clock and geometry readout: `/private/tmp/codex-steer/source-eye-original9-retry-1014/transfer-geometries.json`
- Partial replay: `/private/tmp/codex-steer/source-eye-original9-retry-1014/material-match-replay.json`

Browser and ordinary native-app proof remain open. Presentation readiness
has not been established.

## Falsification and the final combined repair

A broader pixel world applies acknowledged independent inputs to its own
renderer, supplies only pixels to `WhatMoves`, and asks `WhichIsHers` to find
the responder. An attached strip appears and disappears while the textured
body moves and stops. The baseline attributed the controlled body in 67 of
120 observations and ended at experiment epoch zero.

The first material-priority implementation attributed 56 observations and
reset three times. Moving the material association behind ordinary matches
still produced that regression. A trial requiring prior material displacement
was also rejected: it attributed an autonomous distractor in 7 observations
and never established the true responder. That extra guard was removed.
These results are retained; none was promoted as an improvement.

Source records locate the loss: at the stopped body's 8.5-second picture,
only the strip was proposed. The still-present texture had entered the
background. Trail cleanup had used the changing outline centre as travel.
The material path showed a different displacement. Cleanup now uses that
current measured material path when its pose, lifetime and observation clock
agree. Invalid or absent current material history supplies no trail measure.
Ordinary matching runs first; the new geometric association recovers only
unmatched objects and proposals before the existing leap check.

With both repairs, the pixel world again attributed 67 observations, kept one
body number and finished at epoch zero. Its repeated-input receipt selected
the actual renderer-controlled body. This removes the regression; the count
of attributed observations equals the baseline. The permanent tests exercise
the ordinary pixel and control paths with two textures and no supplied body
identity. The full focused selection, including paint and trail suites,
passed 176 tests in 34.28 seconds.

- Baseline: `/private/tmp/codex-steer/pixel-identity-before-1014/records.json`
- Rejected priority: `/private/tmp/codex-steer/pixel-identity-after-1014/records.json`
- Rejected displacement guard: `/private/tmp/codex-steer/pixel-identity-resolved-1014/records.json`
- Source diagnosis: `/private/tmp/codex-steer/pixel-identity-diagnostic-1014/source-decisions.json`
- Corrected path: `/private/tmp/codex-steer/pixel-identity-physical-path-1014/records.json`

The earlier scoped gate run passed writing, method size, layering, dependency
consistency, compile and lint. After import ordering was corrected, Ruff and
smoke passed; smoke had 165 passes and one skip in 44.98 seconds. Global
module-size debt remained 194 lines above its aggregate budget, with the
existing `ResponseGenerationPhase` at 22 methods against 20. Governance
reported 123 regressions, with no changed module listed. No baseline changed.
Final gates and a fresh browser measurement are still required for the
combined repair described in this section.

The final combined gate run passed Ruff, writing, changed-method size,
layering, dependency consistency, compile, lint and smoke. Smoke passed 165
tests with one skip in 42.32 seconds. The global size gate still reports the
same 194-line aggregate debt and `ResponseGenerationPhase` method debt;
neither owned module exceeds its cap. Log:
`/private/tmp/codex-steer/material-match-gates-1014.log`.
