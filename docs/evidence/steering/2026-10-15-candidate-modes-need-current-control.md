# A candidate mode needs current control, 15 October 2026

Original case 3 at `911ee48d0` reached the room but did not demonstrate a win.
The run was manually interrupted after 148.17 seconds, following an observed
mode error. Only owned probe PID 4824 received SIGINT after its exact command
and trace directory were checked. The browser closed and the supervisor reaped
it with rc -2. The recording is explicitly interrupted and incomplete: nine
pictures, three requested/completed normal input pairs, zero dropped events.
There is no completed game result to promote.

Raw evidence: `artifacts/flash-campaign/2026-10-15-case3-cursor-material/`.

The pointer material path produced current correspondences, then refused
clicks whose material it could not find after moving. Resampling a hit after
the full sweep also lost many hand observations. Pairing capture with the
original physical hit remains the next acquisition change.

## The mode error

The screenshot at 126 seconds still shows the house. At 15:58:34.086 the
specialist declared an 8 by 7 well. At 15:58:58.377 it returned 39 pieces and
198 clears after 24.3 seconds, while also returning that nothing pressed moved
the piece across. These were interpretations of scene motion. The outer
actor already rejected that final reason, so this is not evidence of a false
whole-game win. The specialist's counters, learned clear rules and narration
were still unsupported.

The provisional deadline sat below continuation branches for changing or
resting cells. Those branches could postpone admission failure. Saved key
names also counted as known controls without a response in the current run.

Every provisional branch now spends one admission budget. Separate current
horizontal responses must support distinct left and right controls. Recalled
names are candidates to recheck, and a different visible piece cannot supply
the missing response. Before admission, reconstructed cells update provisional
state but cannot produce played-piece counts, clear counts or new saved rules.
The actor also rejects a specialist result without confirmed controls.

Previously verified learning is copied before an attempt. A failed candidate
cannot mutate it through a shared rule, appearance or narration object, and
failure preserves the earlier hypotheses. New saved learning carries its
control-verification flag. Legacy key names can still propose revalidation;
legacy unqualified clear-rule objects do not become verified learning.

A separate observed crash came from an offscreen candidate well producing an
empty array for connected-component extraction. Invalid geometry and empty
crops now return no measured piece. Saved nonfinite values, fractional counts,
invalid color bytes and overflowing extents are refused.

## Checks

The well and new admission suites passed 37 tests in 26.22 seconds before the
last overflow-boundary case was added. The existing pixel worlds still learn
nonstandard controls and clear rows or token pairs. The integration suites
passed 67 checks in 8.90 seconds; smoke passed 165, with one skipped, in 43.50
seconds. Lint, compilation, layering, dependency and writing gates passed
before the final formatting and boundary edits. Final focused checks follow.

This repairs candidate-mode custody. It does not prove a fully causal model of
every grid, successful point acquisition, a game win or native-app readiness.
G09, original game 9, the reserved holdouts and the resident runtime remain
untouched. The default native source checkout has not been deployed from this
worktree.

Final boundary suite: 26 passed in 2.71 seconds. Final smoke: 165 passed,
one skipped, in 45.52 seconds. Final scoped Ruff and compilation passed.
