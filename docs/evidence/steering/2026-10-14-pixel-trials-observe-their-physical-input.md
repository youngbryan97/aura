# Pixel trials observe their physical input — 14 October 2026

## Measured break

The preceding run's first experiment qualified a responsive region and
applied its point-sampling hint. It did not establish a body. The separate
audit, `2026-10-14-where-the-motion-layer-chain-breaks.md`, corrects the
earlier claim that neither experiment found a region.

During its 24 physical presses, the ordinary observer received no frames.
The independent pixel extractor used the captures, but `WhichIsHers.saw`
could not associate a continuing visible entity with those inputs. The
accepted sampling hint could not recover the lost observations afterward.

## Connected observation

The collector can now acquire several current frames during one unchanged
physical input. Its separate `after_capture` callback sends each accepted
capture through the ordinary private observation worker and then through
the existing identity and press observers. The callback checks the input
receipts, held-key owner, interference count and identification epoch
before publishing agency measurements. It adds no entity or control score.

The acquisition deadline applies to fetching pixels. Observation has its
own bounded worker wait. Cancelling the finite physical press wait does
not cancel `PlayObservation.see`, which would permanently retire the
shared observer. Its task remains retained and is joined after input
cleanup. An unresolved observation stops the path rather than allowing
ordinary input to continue against a changing snapshot.

The sample spacing comes from the cadence selected before outcomes.
Missed intermediate slots are skipped; the final capture must still
start after the selected settling interval. Input is released immediately
after that capture, before processing its pixels. Slow processing cannot
extend the physical endpoint indefinitely. The existing 1.05-second
press deadline, acknowledgement bounds and retained-release cleanup remain.

If ordinary observations establish current measured control, collection
ends, releases input and retires its partial statistical cohort. It also
ends on an observed screen transition, including one rendered inside the
same canvas. Shared scene bookkeeping clears prior spatial plans and
preserves the phase's input budget before ordinary play resumes.

A completed region uses a strictly later capture for its hint. Reusing
the already observed endpoint would be rejected as a repeated timestamp.
The new capture must retain its surface and epoch; the hint is bound to
its actual pixels and must receive the existing sampler receipt.

## Verification before original-game use

The expanded targeted set passed 103 tests in 51.54 seconds before the
shared scene helper. A subsequent set including that helper, the physical
browser cases, collector, navigation and worker deadlines passed 83 tests
in 35.47 seconds.

Five Chromium cases use physical keyboard events, captured Canvas pixels,
the ordinary `WhatMoves` worker and `WhichIsHers`:

- Responsive material establishes control through independent repeated
  trials, without a qualified region assigning identity.
- A world that records the keys but ignores them completes all 24 presses
  without identifying a body or qualifying a region.
- A large visual transition inside the same canvas retires the cohort and
  confirms release before returning to ordinary observation.
- A slow observation worker exceeds the finite press allowance without
  retiring the shared eyes. Input is released, no partial cohort qualifies,
  and a later ordinary observation succeeds.
- A scene transition whose observation finishes after the press deadline
  still reaches phase bookkeeping when its retained task is joined.

The serial gates passed Ruff, writing, changed-method size, layering,
dependency consistency, compilation and smoke (165 passed, one skipped,
42.43 seconds). Module size retains two existing failures: total oversize
124,546 against 124,352 and `ResponseGenerationPhase` at 22 methods against
20. Governance retains its existing 123 regressions. Those baselines were
not changed. The actor is 1,993 lines against its 2,000-line cap.
After the review fixes, the final collector, actor wiring and five-case
Chromium set passed 49 tests in 35.88 seconds. Ruff and writing passed again.

These are integration fixtures. They do not establish an original-game
win, native-app prompt behavior or presentation readiness. No holdout game
was opened, no authority threshold was lowered, and no resident model job
was stopped or restarted.
