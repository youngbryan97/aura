# Optional geometry shares observation time

Date: 2026-10-18. Worktree: `codex-steering-1012`.

## Measured failure

The saved ordinary CPU case-18 run at `1e277a2c7` selected its controlled
vehicle but learned no lateral control and ended with zero landings and
one crash. The 90-picture prefix from its first stream was replayed with
the recorded actual input ownership snapshots. Pixel hashes were checked.
The asynchronous reference worker, changed decisions and new outcomes are
outside this replay.

The default classical tracker selected a body on 41 pictures, as did the
original CPU continuity tracker. Neither learned a lateral key. Their measured
replay durations were 9.745608 and 12.589926 seconds. The alternate tracker
is not promoted. The detailed paired receipt is under
`artifacts/flash-campaign/2026-10-18-lateral-paired-replay/`.

A second, instrumented replay kept copies of each sample list at observation
time; the first diagnostic had retained mutable lists for its raw-sample
details. Its complete replay and point results were unaffected. The second
receipt is `2026-10-18-lateral-material-replay/replay.json` under the same
artifact root. It reproduces the first lateral trial losing its material
origin at a 0.350355-second gap. The next thrust and right trials encounter
outline/point disagreement before their first usable acceleration interval.
Later measured pictures do not rescue those contradicted physical trials.

The saved ordinary stage trace locates the first gap. Mandatory motion
observation took 0.148598 seconds, state copying 0.005082 seconds, then
optional fresh reference geometry took another 0.105878 seconds before
the already measured state reached control. Complete observation took
0.260533 seconds. The next capture arrived too late for the unchanged
0.35-second named-material interval. This identifies a critical-path cost;
it does not establish that removing that cost alone will win the game.

A serial cProfile replay measured 6.815 seconds inside the 90 ordinary
motion observations. Source-point observation accounted for 4.761 seconds,
including 1.659 seconds of local patch refinement. That profile is preserved
in `2026-10-18-lateral-replay-profile/`. Its instrumented timings cannot be
read as ordinary application latency.

## Retained mechanism

`OptionalWorkShare` is a general, worker-owned accounting mechanism. It earns
credit from observed essential work and charges actual optional elapsed time,
including failed operations. Credit is capped at 0.1 seconds; a new operation
needs 0.025 seconds. The default optional fraction is one quarter of combined
accounted work. One indivisible operation may overdraw its allowance, but
that debt must be repaid before another starts. There is no wait or queue.

These are scheduling policy constants, not measured optimal parameters.
This mechanism does not enforce a per-frame deadline, preempt a running
operation or measure the host's CPU share. A long optional operation can
still delay its own frame. The next costly operation must yield intervening
pictures to essential work.

The ordinary observation worker measures motion and state-copy time, then
uses this account for optional current-pixel reference refitting. A deferred
operation returns no fresh geometry and records
`fresh_reference_work_deferred`. Direct correspondence qualification,
source scope, material ownership, control thresholds and physical leases
are unchanged. The existing asynchronous catalog/matcher remains separate.
No unsearched alternative becomes an identity or task role.

## Validation in progress

Five new tests failed before the budget API existed. The first integration
group passed 71 checks and failed three instant-observer fixtures that
expected a refit without any essential-work interval. Those fixtures now
provide a measured synthetic duration; budget deferral is tested separately.
Original-game carryover, native-app proof and presentation qualification
remain open.

No protected game, holdout, native lifecycle or exclusive model job was used.

The corrected integration group passed 89 checks in 20.81 seconds (22.56
seconds under supervision). It covers current-pixel qualification, reference
scope, ordinary observer publication, held-input release while optional work
blocks, delayed reference transport and measured budget admission. Three
variable-work simulations check that earlier overspend is repaid before
another operation and that only the final indivisible operation can exceed
earned credit. Smoke and the repository code gates are running next.

Smoke passed 165 checks with one skipped. Lint, compilation, layering,
dependency consistency and writing passed under a 143.79-second supervisor.
The writing baselines were unchanged. This checkpoint is ready for the
ordinary CPU-browser follow-up; it does not qualify the native demo.
