# Control samples after a missing picture

## Browser result before this change

The CPU-only original case 18 run at `68f5b8f65` completed in 85.95 seconds.
It entered play from the title through physical clicks. The recorded active
stretch lasted 14.6 seconds and ended in a loss. The independently viewed
last active capture showed zero landings, one crash and zero percent success.
No native-app demo or original-game win was established.

The clock repair reached the actual actor. Its 67 control observations
contained 33 frames with fresh source geometry and 23 with source points
attached to observed objects. Fresh matches included 32 frames of export 57
and one of export 52. Independent repeated input trials selected body 2 in
31 observations. There were zero recorded goal-directed play choices.
Geometry and body identification therefore reached ordinary browser play;
successful control did not follow.

The recording closed without dropped events: 177 images and 32 input-port
events. The browser reported SwiftShader. The process-local classical import
boundary stayed active, reported no loaded model backends and refused four
MLX imports. G09 and the holdout games were excluded.

Artifacts are under
`artifacts/flash-campaign/2026-10-16-case18-clock-fixed/`.
The pixel-only replay completed in 5.32 seconds. It supplies perception
diagnostics on the recorded inputs, not an outcome under a changed policy.
An earlier control-replay request refused a stream without owned input
snapshots. That refusal supplies no control result.

## A measurement gate found in the code

`_Speeds.unseen` retained a two-sample continuous prefix, sealed its input
receipt, and refused all later samples for that held input. Acceleration
estimation requires four samples spanning at least 0.15 seconds. A later
clear interval could never satisfy it after that first short prefix.

The change keeps later continuous intervals separately. One selected
continuous interval can supply acceleration for each physical press. Missing
stretches are never joined, and several intervals from one press never
become independent repeated presses. The first prefix used for pace
statistics stays unchanged. An actual input end prevents further collection;
a contradiction before that end retires the input's acceleration evidence.

The acquisition gate reads the same selected interval as the acceleration
estimator. Its existing four-sample requirement, independent-repeat rule,
hard physical input deadline, control allowance and identification thresholds
are unchanged. Later storage is bounded to 80 receipts and 128 samples per
interval. A timestamp high-water mark also survives a gap.

The actor's opt-in recording now includes its selected owned control
intervals and each object's observed velocity and extent. These are actor
measurements, not independently observed game state.

## Checks so far

Both targeted regression checks failed before the production change: no
acceleration estimate and no acquisition completion from the later interval.
The focused group then passed 174 checks. It covers the ordinary
`WhichIsHers.saw` path, repeated acceleration identification, finite storage,
input completion, contradictions, pinned or unsettled samples, timestamp
ordering across gaps, existing prefix custody, actual trial acquisition and
input leases. The claim is registered beside its invariant.

The first broad command named a nonexistent test file and collected nothing.
The next command exposed three nested-approximation assertions in the new
test; tuple comparisons corrected those assertions. The subsequent groups
passed 173 and 174 checks. All attempts remain under
`artifacts/flash-campaign/2026-10-16-later-control-intervals/`.

This establishes the repaired measurement path in regression tests. It does
not establish that missing intervals caused every failure in the preceding
browser run. A fresh browser attempt must measure whether it changes play.

The final focused group passed 176 checks after a recording-selection test
was added. That test distinguishes an original interval from a later one
when both exist. The preceding gate group passed smoke (165 passed, one
skipped), lint, compilation, layering, dependency consistency and writing.
Governance reported the same 131 existing regressions; no baseline changed.

The final gate group again passed smoke (165 passed, one skipped), lint,
compilation, layering, dependency consistency and writing. Module-size
reported the same four existing regressions: `phantom_browser.py` at 2008
lines, two response-generation class method counts, and the aggregate
oversize budget at 197 lines above baseline. No changed module crossed a
size limit. The full suite remains deferred beside the exclusive model job.
