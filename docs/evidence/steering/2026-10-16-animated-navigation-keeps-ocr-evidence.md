# Animated navigation keeps OCR evidence, 16 October 2026

The fresh development case-18 CPU attempt at `e9ed28187` ended on its title
screen. It completed in 69.92 seconds with `won=false` and one unread run.
The recording closed with 93 images, ten input-port events, five normal
screen-option records and no dropped events or recording error. No motor
frame, control-observation or play-choice event occurred. This run therefore
does not measure the new source geometry during gameplay.

The measured browser driver was SwiftShader. The in-process compute
boundary remained active, listed no loaded known backend and refused five
attempted MLX imports. This confirms that the older worker prohibition
alone did not cover every in-process fallback. No G09 job was changed.

## Where the control was discarded

All five normal screen records read the title button as `BTART`. Its
normalized centre was `(0.50078, 0.67448)` and it had `of_its_own=True`.
The complete label occupied the button in the recorded browser image.
Only the title `OPERATION:` survived as an offered click. The actor tried
arrow keys while the Start screen remained visible.

`how_much_it_leads_on` already recognizes a one-character error in a known
navigation word. That score was never reached for this button:
`things_to_click` discarded animated text unless the stricter
`label_is_a_way_on` matched exactly. Animation classification and OCR noise
therefore removed a usable measured control before ranking.

## General correction

Exact semantic-label recognition retains the existing grammar. The two
consumers of measured OCR can explicitly request bounded tolerance. A whole
label may have one wrong, missing or added character and must have one
unambiguous expected navigation label. Readouts, measures, paragraphs,
unbounded prefixes and destructive reset variants remain excluded from
this additional path. No game name, character, coordinate or special
substitution was added to production code.

Both the animation filter and the captured-menu handoff use that predicate.
The original damaged label and its measured coordinates remain the click
target. Current word pixels, surface ownership, freshness and occurrence
budgets remain required before a menu can interrupt body trials. Candidate
recognition certifies neither click delivery nor navigation success.

Focused navigation, handoff and recording tests: 125 passed. They cover
different damaged navigation labels, ambiguity, exact-label compatibility,
stale words, readouts, animated prose and reset exclusions. The next CPU
attempt must demonstrate that the actual UI press reaches gameplay.

Artifacts remain under
`artifacts/flash-campaign/2026-10-16-case18-current-source/`.
Native prompt proof, the 56-case campaign and presentation qualification
remain open. G09 and the ten holdouts were excluded.

Checkpoint gates: smoke 165 passed, one skipped; lint, compile, layering,
generated DEPS consistency and writing passed. No baseline was changed.

## Fresh UI verification and the next failed boundary

The attempt at `abd07b990` offered and clicked `BTART`, then offered and
clicked the instruction screen's damaged Play label. It entered the actual
fast actor. The recorder has 77 motor captures and 77 control observations,
with 41 observations selecting body 4 after independent repeated input
trials. There are no normal play-choice records. The attempt ended lost
and then unread, with `won=false`; its supervisor took 64.01 seconds.
The recording closed with 141 images, 30 input-port events and no drops.

The new source records exposed another boundary failure: every control
observation reported `capture_not_measured`, and none received a source
reference. Ordinary play supplies `MeasuredFrameTime` or `CanvasFrameTime`,
float subclasses with capture bounds and source provenance. The main eyes
already convert that time to a plain scalar for their pixel receipts.
The optional source worker still received the rich receipt and refused
its exact-type check. Fresh matching separately refused the same value
when checking consumed pixels. Earlier replay used plain floats and did
not expose either refusal.

The observation boundary now passes the complete clock receipt to the
eyes, and passes its scalar capture value to both source-geometry paths.
Input timing, source clock, capture bounds, surface and sequence remain
on the original receipt. Pixel geometry gains no causal timing authority.
No receipt validator was weakened and no missing capture was invented.

The regression test failed for both real page and canvas clock types
before the fix; the plain-float case passed. One initial canvas fixture
omitted its required surface and sequence, then was corrected before the
two-clock failure was measured. After the fix, 113 focused observation,
reference, input-deadline, recording and navigation tests passed. The
retained red and green supervisor records are under
`artifacts/flash-campaign/2026-10-16-case18-navigation-fixed/`.

This verifies reaching gameplay, not a win or a complete control loop.
A fresh original run must show the catalog and source geometry reaching
the actor under its actual clock.

Clock-boundary gates: smoke 165 passed, one skipped; lint, compile,
layering, generated DEPS consistency and writing passed. The inspected
last motor image reads zero landings and one crash. Native proof and the
campaign remain unqualified.
