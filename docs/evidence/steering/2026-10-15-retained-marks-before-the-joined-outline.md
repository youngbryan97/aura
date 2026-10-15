# Retained marks before the joined outline

Bryan supplied an analysis of five earlier failures. It describes the
older 0.25-second texture limit, blocked motion, queued exposure, menu
occurrences and attempted placements. The local tree already contains
several later repairs. The report is a set of hypotheses and prototype
tests; it is not a measured result for all 56 original games.

## What is already connected

- Texture separation uses `NAMED_POINT_GAP_S`, currently 0.35 seconds.
  `test_continuing_texture_can_separate_a_joined_outline.py` exercises
  the ordinary `_blobs` path at 0.249, 0.25, 0.268130064 and 0.35 seconds.
  The two appearance-only paths retain their shorter window. Extending
  their prediction window would not supply missing material evidence.
- `_captured` requires the selected capture start, and the collector
  takes its final neutral baseline after ordinary observation finishes.
  A queued frame cannot replace the selected physical exposure.
- Navigation already checks current word pixels before more body trials.
  Its occurrence budget requires two ordered captures showing that the
  previous label disappeared. The whole-label grammar includes Next Level.
  Empty OCR or a skipped callback cannot invent that disappearance.
- Commit 05c75b75e is present. A tried carry alone does not establish a
  creation workflow or a placed object. Existing transfer and completion
  checks still require the observed work.

`motion_is_resolved` already uses the largest measured excursion over
either coordinate. A freely moving direction is not erased by a later
clamped direction on that same entity. The suggested resolver additionally
needs independent evidence that a boundary or bounce occurred. Its input
booleans do not provide that evidence, and movement by an unrelated object
cannot establish control. No boundary label or game variable was added to
production control attribution.

## A remaining failure reproduced from pixels

The material path had a different majority problem. Once a body's outline
included a large stationary textured area, `FollowingPoints.motion_of`
collected every point in that enlarged support and chose their largest
motion consensus. The body's own points could keep moving while hundreds
of surrounding points supplied zero motion. An initial reproducer reported
zero displacement from 236 points although the body had moved four pixels.

The retained regression generates RGB pictures of a textured moving body
and stationary texture. The real trackers extract the correspondences.
The test deliberately supplies a joined outline, then exercises ordinary
`_match` and `_moved` calls before separating the outline again. Both the
coarse and source-resolution trackers reproduce the failure on the parent
checkpoint 7fa671d25. Both pass with the repair. This tests point extraction,
material motion and association; it does not test foreground segmentation,
physical input, an original game or the native Aura prompt path.

The source tracker lost six of its thirteen initial marks on the first
displacement in the first fixture. Seven marks do not meet the existing
60-percent retention requirement. That failure remains unavailable. The
retained fixture first establishes a moving material path through ordinary
observations, then tests the joined-outline failure. No retention threshold
was lowered to make the source tracker pass.

## Change

A current `PointPose` now supplies the first motion hypothesis. At least
three of its retained named marks, and at least 60 percent of its anchors,
must agree with their previous observed coordinates and a rigid fit. New
corners can join only when their own measured displacement agrees with
that fit. The final fit must still retain the original anchor fraction.
The receipt has at most 256 points and keeps retained names before new
corners. The existing 0.75-pixel rigid residual limit remains unchanged.

An absent, stale, replaced, duplicated or deformed retained set supplies
no motion receipt. It cannot yield to a larger surrounding consensus.
This preserves measured material motion; it grants no control identity,
stationary turn model or game outcome. The repair needs no canonical-area
guess, character label, named key or game-specific condition.

The final focused perception set passed 249 tests in 35.35 seconds. The
new regressions include rotation, stationary and moving surroundings,
retained-mark loss, changed birth, stale pose, coordinate disagreement,
duplicate names and the bounded receipt. The invariant and measured claim
are registered beside the implementation and its tests.

Claude's G09 process remains running and untouched. No browser, model or
desktop runtime was launched for this repair. Fresh original-game evidence,
ordinary native-app prompt proof and presentation readiness remain open.

Smoke passed 165 tests with one skip in 72.57 seconds while G09 was
running. Ruff, compile, layering, dependency graph, writing and changed-method
checks passed. The size gate reports an aggregate excess of 218 lines and
the existing `ResponseGenerationPhase` method excess. The clean parent
worktree reports the same two size failures. Governance reports the existing
123 regressions. No baseline was relaxed. The governance output is kept in
`~/.aura/flash-campaign/evidence/2026-10-15-owned-material-gates/governance.log`.
