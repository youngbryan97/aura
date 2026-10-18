# One position per optional reference visit

Date: 2026-10-18. Follows the failed ordinary run recorded in
`2026-10-18-optional-geometry-shares-observation-time.md`.

## Remaining critical-path work

Amortized accounting deferred 26 optional fits in that run. One admitted
operation still took 0.193575 seconds. A previous reference pose entered
up to three local and three whole-picture candidate positions, each with
forward and reverse patch refinement. One pose was not one fine operation.
No lateral response or original win qualified in the follow-up.

## Implementation

The reference matcher now accepts an explicit position budget for one
pose. A position selects one ranked local or whole-picture candidate.
At most one candidate enters the existing fine solver. Invalid budgets
and a multi-pose budgeted request are refused before fitting. The complete
offline search API remains available when no position is requested.

Ordinary budgeted observation requests this one-position path. Failed or
imprecise visits rotate over all three local and all three wider peaks
for each pose. The existing pose and reference rotation remains in place.
A precision-qualified fit returns that pose to its first local peak. A
fine match that fails target precision cannot restart the local search.
The cursor storage is bounded by the 32-reference, four-pose catalog and
clears on source or clock scope replacement.

Untried positions are recorded as
`fresh_reference_positions_unsearched`. Each published result still needs
the current consumed pixels, current source, redundant material support
and the unchanged precision test. A selected search position supplies no
temporal identity, controlled body, role or task outcome. No control
threshold or physical lease changes.

## Checks

Fourteen new regressions failed before the position API existed. The
implemented integration group passed 105 checks in 36.76 seconds (38.81
seconds under supervision). It includes a bounded fine-call count for
all six positions, invalid and multi-pose refusals, real current-pixel
recovery outside the local window, absent-material refusal, all 24
pose/position visits, and the ordinary worker's budgeted-path request.
The earlier reference, input-deadline and delayed-material tests also pass.

Repository gates and original carryover remain to be checked. These
component results do not qualify the native demo or 56-game campaign.

Smoke passed 165 checks with one skipped. Lint, compilation, layering,
dependency consistency and writing passed in 195.05 supervised seconds.
The next original run will retain timing and input/pixel receipts. A longer
bounded play allocation gives recovery after an early loss time to show;
it does not supply private state or change Aura's policy.

Governance lint completed in 12.80 seconds and reports 131 regressions
against its unchanged baseline. The preceding checkpoint had 130. The
current findings include the newly merged `core/interiority/what_a_turn_did.py`
directory write, outside this repair's files. Do not report aggregate quality
as green or refresh the debt baseline to hide this result.

The same scoped AST scanner compared this repair's five production files
with `b96671a9b`, before either scheduling change. Each file has zero
recognized consequential effects both before and after; the scoped counts
are identical. The retained receipt is
`2026-10-18-single-position-refit/effect-comparison.json` under the campaign
artifact root. The aggregate gate still fails and its baseline is untouched.
