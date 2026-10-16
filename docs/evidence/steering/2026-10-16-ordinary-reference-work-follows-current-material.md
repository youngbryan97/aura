# Ordinary reference work follows current material

The detached-reference checkpoint preserved the right observation owner,
but normal play did not launch that work. This change connects registered
source artwork to `PlayObservation`. It does not establish a game win.

## Runtime connection

`visual_references_for_current_sources` acquires local registered exports
for the complete source tables from the current program read. Each catalog
acquisition checks the renderer receipt, original Flash bytes, full source
declaration, reference PNG, SVG and independent coordinate raster. It
remeasures raster alignment before accepting the stored source coordinates.
The reference pixels then belong to the catalog and are read-only.

The local index points to an existing export bundle. Moving or changing that
bundle can make the next acquisition unavailable. `--publish` on the export
tool registers a source-coordinate export under the selected runtime state
root. Tests and replay use the private CPU state root, not native state.
There is no automatic renderer invocation for a newly visited source yet.

A new program acquisition replaces the guide's current visual source scope.
A failed acquisition clears it. Historical source tables cannot retain
artwork authority when the current program is different or unavailable.
This changes the previous union behavior; the declaration test now requires
replacement. Saved guide memory still contains no source declarations.

`SourceReferencesInPlay` launches optional work from the ordinary observation
wrapper. There is one global reference-operation permit and at most one
pending operation per owner. Catalog I/O and matching run on the executor.
The main loop copies only a consumed, bounded material frame after admission.
New observations never queue behind a reference operation. A running job
keeps its permit after source replacement or shutdown until it actually ends.
Its old generation cannot publish into a new source scope.

An acquired source origin still requires qualified pixel correspondence,
unique current material ownership and the existing named-point basis.
Delivery transports that origin through the original observer's continuing
material. Stale captures, changed tracks and replaced bases refuse it.
The worker publishes current source geometry without changing the motion
origin, causal body identity, contact rules or task role.

## Reacquisition

The initial worker searched progressively different orientations even after
finding a reference. The recorded replay exposed lost reacquisitions after
point-basis replacement. The worker now retains measured appearance scale
and orientation as search proposals. Two of three subsequent catalog passes
search nearby poses; the third keeps broad discovery available. A proposal
must pass the same fresh pixel and material checks. A failed appearance
search cannot discard an already continuing measured binding.

Several qualified poses can describe the same tracked material. Their
current origin estimates are coalesced only if one enclosing error circle
fits within the existing one-working-pixel budget. Its radius includes the
distance to every candidate plus that candidate's estimated error. These
correlated hypotheses do not increase confidence. Disagreement beyond the
budget refuses publication. Different tracked copies remain distinct.

## Measurements

The source-paced replay uses 100 previously recorded original-case-18 RGB
captures, through the ordinary asynchronous observation wrapper. It starts
no model and sends no game input. Its guide reads the supplied source bytes;
this is not a fresh native-browser source acquisition.

The initial worker published current points on 27 frames in 18.763430 seconds.
Its largest event-loop pulse gap was 0.015647 seconds. After measured-view
search, the same recording produced points on 57 frames in 18.766834 seconds,
with a 0.018073-second maximum pulse gap. Scheduling is real and these counts
are measurements from one run each, not deterministic success guarantees.

That second run retained multiple agreeing hypotheses on some frames. Fresh
zero-angle correspondence supplied 39 independent point checks, with largest
difference 0.630021 working pixels. Another 47 point checks had no unique
fresh appearance and remain unmeasured. Those counts are point checks, not
independent trials or distinct frames. The final coalesced run is recorded
separately under the same private artifact directory.

Artifacts: `artifacts/flash-campaign/2026-10-16-reference-worker/` contains the
bounded test receipts, initial and measured-view replays, fresh pixel checks
and the final coalesced replay. Missing checks do not pass. The replay sets
`task_passed=None` and `presentation_qualified=False`.

Focused checks: 219 passed. They cover source/declaration changes, damaged
reference bundles, failed current reads, delayed delivery, replaced bases,
two current copies, source generations, pose disagreement and retirement.
A blocked matcher test verifies that an input lease releases while the
matcher is still running and another ordinary observation is published.
The test hands implement the ordinary input protocol; this is not native
keyboard-delivery evidence.

## Open boundaries

Native deployment and ordinary live play have not been tested here. The
exclusive corpus/model job remained active, and no resident runtime was
started, stopped or restarted. G09 and the recorded holdouts were not used.

The local reference catalog still needs automatic source-asset acquisition.
Destination occurrences and source-defined contact predicates still need
their own verified connection to control. A recognized export cannot stand
in for semantic goal binding, source collision geometry or a controlled body.
Neither the 56-game campaign nor independent native demo readiness passes
on this evidence.

Final coalesced replay: 57 of 100 frames published a point, in 18.783248
seconds. The largest event-loop pulse gap was 0.017494 seconds; the largest
ordinary frame call was 0.130080 seconds. Thirty-one fresh independent pixel
checks were measured, all within one working pixel, with largest difference
0.630021. Twenty-six checks remain unmeasured. The 57 points comprise 39
single-hypothesis estimates, seven two-hypothesis estimates and eleven
three-hypothesis estimates. Agreement does not create independent trials.

The global module-size gate still reports the existing `phantom_browser`
ceiling and two response-phase method regressions, plus total oversize of
197 lines. None of the changed modules adds a size regression; the new
catalog and worker contain 206 and 276 lines respectively. Its baseline
was not changed.

The first governance check counted two new publication effects in the
perception module. Publication is an operator/build operation; normal play
only acquires an existing catalog. The publisher now lives in
`tools/publish_program_visual_reference.py`, also used by the export and
replay tools. Runtime acquisition has no writer method. Its test refuses
any writer request and verifies that acquisition leaves the index unchanged.
The catalog is now 181 lines. Post-move focused checks: 57 passed.
Governance lint reports 131 existing regressions, with none in these new
modules. The baseline was not expanded and existing debt remains open.

Final gates after the publication move: smoke 165 passed, one skipped;
lint, compile, layering, generated DEPS consistency and writing passed.
The full suite was not run beside the exclusive model job. The corpus job
was still active as PID 53965 at this checkpoint. No native process was
started or restarted, and no gameplay success or demo qualification is
inferred from the replay or these gates.
