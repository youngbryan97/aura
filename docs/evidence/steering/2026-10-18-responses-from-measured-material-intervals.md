# Responses from measured material intervals

This follows the counterfactual in
`2026-10-18-information-in-observed-intervals.md`. It changes general
measurement and control code. G09, the recorded holdouts and the resident
runtime are excluded. The work uses classical CPU perception.

## Measurement path

The point observer now publishes an immutable local displacement receipt
when current visible names continue an existing material origin. It refits
the local transform and estimates registration precision at that origin.
Shared reference translation error cancels; rotation transports the shared
error through the difference between the linear transform and identity.
The estimate also includes disagreement between that fit and the continued
origin. The working-pixel resolution floor remains one pixel. These are
estimated errors, not guarantees against correlated point-tracking errors.
Missing correspondence supplies no displacement receipt.

Control attribution consumes only receipts for its current entity, birth,
origin and measured position. Both capture receipts must have been consumed,
keep their source, surface and epoch, and fall wholly within one delivered
input after its response delay. A velocity is dated at the midpoint of its
actual displacement interval. Its error includes local displacement error
and the widths of both capture intervals.

Acceleration fits use continuous intervals over at least 0.15 seconds.
Gaps split fits; multiple fits from one physical hold remain one trial.
An active contradiction retires that hold's estimate. The comparison adds
estimated measurement variance to within-input variation and subtracts estimated
error from the effect. The existing independent-trial gate, resolved-motion
gate, exclusions and statistic/effect thresholds of 20/20 remain. The
older four-picture estimator remains available for observations without
this precision channel. Selecting a body copies its new response history
without sharing mutable sample lists.

## First checks

The initial numerical helper passed 23 tests. The integrated focused group
passed 230 tests in 44.61 seconds (47.05 seconds supervised), including the
existing causal identification, point-origin, visible-support, inertial-axis
and independent-trial tests. Receipts are under the private artifact root
`artifacts/flash-campaign/2026-10-18-interval-model/`.

A hash-verified replay of the first 44 recorded pictures from
`2026-10-18-ordinary-single-position-refit` completed in 6.63 seconds. It
selected track 1 on 15 observations, starting at 302428.5934510231. The
first selection's statistic/effect were 33.567061/21.331932; the older replay
selected no body. Motion statuses remained 70 unavailable and 370 measured.
The independently viewed `frame-0000098.png` shows the selected box on the
visible vehicle. Its material origin differs from its current outline's
centre, as expected for a retained origin through animation.

This replay uses recorded actor input ownership. It does not replay new
decisions, asynchronous source-reference work or changed outcomes. It
establishes that the new measurement reaches attribution on those saved
pixels. It establishes no original-game win, landing, 56-game pass or
native-app presentation readiness.

## Capture uncertainty on the fit's time axis

Capture widths also perturb each velocity interval's midpoint. The fitter
now propagates that time-axis error, solves its feedback term in the slope
error estimate, and refuses a time axis whose uncertainty consumes all the
measured separation. The focused follow-up passed 97 tests in 20.87 seconds
(23.35 seconds supervised). This also checks that the local displacement
receipt survives the ordinary detached worker snapshot and serialization.

Three serial 44-picture replays completed in 18.72 seconds. With time-axis
uncertainty included, the ordinary code still selected the vehicle on 15
observations, with first statistic/effect 27.434242/20.325563. Disabling the
interval estimator selected zero bodies. Removing capture-clock metadata
also selected zero. All three had the same motion status counts. These
ablations isolate the new measurement path on the saved inputs; they do not
establish a new gameplay result. No threshold was changed to retain the
positive result.

## Retirement before the first precise interval

Review found that an active contradiction with no precise samples yet left
no retirement marker in the new estimator. A later clear interval could then
enter that same physical hold. Physical completion had the same empty-history
gap. Both now leave bounded markers even when no samples were measured.
Copying history also respects retirement. The regression group passed 153
tests in 28.40 seconds (30.59 seconds supervised). A further original-pixel
replay still selected track 1 on 15 of 44 pictures, in 7.64 seconds supervised.

Smoke passed 165 tests with one skip. Lint, compile, layering, generated DEPS
and writing passed before this small retirement refinement. The module-size
aggregate failed on existing files outside this change:
`phantom_browser.py`, `response_generation_unitary.py` and
`response_generation.py`, plus the total size budget. Their working-tree
diffs are empty. Its receipt is `final-code-gates.json`; no baseline changed.
A scoped AST review confirms all six changed production modules remain
within 2,000 lines and 30 methods per class, and their recognized effect
inventory is zero before and after. See `scope-review.json`.

## October 20 continuation

The interrupted checkpoint gate finished after the session ended. Its
supervisor records exit 2 in 140.57 seconds for smoke, lint, compile and
governance. A new governance-only run reproduces 131 existing effect-ownership
regressions in 5.84 seconds. None occurs in the six production modules above;
their scoped comparison remains zero before and after. Baselines are unchanged.
The artifact is `2026-10-20-interval-checkpoint/governance.json` under the
private campaign root. This aggregate gate is not reported as passing.
