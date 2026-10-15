# Body selection still needs usable controls

The assigned original-case-9 run at 5024e69d1 completed normally in
165.419932458 seconds. It made 511 choices and 626 ordinary candidate
observations. Median decision picture age was 0.2500800772 seconds.
The game was not won; its timer reached zero with no target parked.
Evidence is retained in
`/private/tmp/codex-steer/sparse-baseline-original9-1014`.

The pixel cohort admitted 21 of its planned 24 presses. Its next pair
was refused by the unchanged exposure test. No pixel region or sampling
hint qualified. The collector supplied 113 ordinary observations during
the intervention and confirmed physical release.

The ordinary identity path selected a body for 88 observations over
12.95 seconds, beginning with track 137 at source time 532495.570342.
Its first selection had an independent repeated-trials receipt with
F = 895.623881 and effect = 32.844787. Subsequent continuity selections
changed between tracks 162 and 163. These are internal identity decisions;
they do not alone prove correct body ownership. No usable movement keys
had been learned by the end of that stretch.

Two further seams are now visible:

1. The fresh paired baseline was newer than the earlier neutral frame,
   but could still have been queued while that neutral frame was being
   processed. The rejected pair's baseline-to-input age was 0.132264868
   seconds, against 0.051629467 for the first pair. Its total pair span
   drifted by 0.107145071 seconds. The next capture must start after
   neutral processing finishes, rather than merely after the prior
   frame's capture time. A regression supplies such a queued frame and
   requires it to be rejected before dispatch.
2. Once a body is selected, the ordinary fallback chooses the first
   control whose response remains unknown. When that response cannot
   be learned, it repeatedly consumes the remaining experiments and
   leaves other unknown controls untried. This run repeatedly retried
   `up` during the last body stretch. A fair, finite experiment budget
   must belong to the current body and survive key reordering; an
   unavailable result must count as an issued experiment.

The captured browser window was observed independently through the
computer-use tool while on its level instructions. No independent live
observation of successful steering or winning was obtained. The model
lane correctly refused inference while the resident G09 job owned it.
This was a bounded ordinary-actor browser run, not a native Aura prompt
demonstration. Presentation readiness remains open.
