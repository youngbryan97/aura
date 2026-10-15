# Continuing motion cannot train a static pointing inverse, 15 October 2026

The original case 3 repeat at `f29daccdb` finished in 219.58 seconds with six
moves, no win and blank current text. The complete recording has 235 pictures,
12 normal input events, 84 pointer references and 134 pointer comparisons;
none were dropped. Three corrected cursor requests immediately follow pixel
comparisons whose pointer/target error is zero; their port completions were
true. These are delivered cursor presses, not proof of an accepted task action.
The final image remains in the house. No inventory collection was established.

Raw evidence: `artifacts/flash-campaign/2026-10-15-case3-paired-cursor/`.
Its recording retains `presentation_qualified: false`.

Material acquisition improved: several looks retained four current reference
correspondences. The correction trace then exposed a different error. Target
position kept changing between successive pointer placements. The controller
fed that continuing motion to a static inverse and made large corrections in
the wrong direction before exhausting six attempts.

## Check against the program

A read-only audit of the supplied original file took 0.042 seconds through the
basic Flash reader. Source SHA256:
`af62baa9457b229a5e0f6bc5927f63a6c00de5523ee5168291dd8be358373058`.
The file declares a 432 by 330 stage at 30 frames per second. A dragged input
proxy feeds piecewise horizontal and vertical speed expressions. A frame
script adds those speeds to the view's position and clamps the result. Its
middle input region produces zero speed. Holding an edge input therefore
continues moving the view; a static mapping from pointer position to target
position is the wrong system model.

These dimensions, thresholds, names and file identity do not enter production
code or its controller. They explain the measured failure and identify the
family of dynamics a general test must reproduce.

## Repair

Each correction now obtains two fresh correspondences while the pointer stays
in one place. Continuing drift discards the static inverse estimate and uses
a bounded damped correction. Settled observations can still identify the
static response. Each step retains both capture intervals and the preceding
pointer-move time. The original time budget remains; the iteration cap is eight.
Loss, ambiguity or a missing second measurement still refuses the attempt.

Partial surveys also spend their first physical queries across the view. The
old query order could use the entire four-reference budget in one corner.
Farthest-first ordering changes visit order while preserving output indexing
and the same final pointer location for physical and batched surveys.

## Checks and limits

Three pixel worlds whose content keeps moving under held edge input converge,
including motion while picture processing takes time. Their dimensions, rates
and bands differ from the original program. Static positive/negative and
cross-axis response tests still pass; lost, repeated and unreachable references
continue to refuse. The broad focused run passed 108 checks in 17.17 seconds.

The Chromium fixture was tightened after review: its earlier initial point
could succeed in one arrival, so it did not require iterative correction.
It now receives an independent keyboard-driven layout shift after discovery,
then must produce at least two correction cycles. Physical input, screenshots,
normal-port down/up and exact recorded-pixel replay pass. The final focused
run passed 46 tests in 12.71 seconds. Smoke passed 165, with one skipped, in
43.18 seconds. Ruff passed; lint, compile, layering, dependency and writing
had passed before the final fixture/canary additions, which introduced no
new imports or runtime control branches.

An original repeat remains required. These results do not qualify the full
campaign or native-app readiness. G09's model job, original game 9 and the
reserved holdouts remain untouched. No native runtime was started or stopped.
