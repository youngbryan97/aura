# WorkToDo: what was built, and what it measures

All 23 items in [TODO.md](TODO.md) are closed. This is what each one added and,
where it produced a number, what the number was.

## Two patterns that recurred

**Mechanisms that could not fire.** Five were written, passed their own tests,
passed every gate, and had no reachable path:

| Where | Why it could not fire |
|---|---|
| `substrate_allocation` weights branch | strictly costlier and more disruptive than an adapter at every level of usefulness, so no benefit could select it |
| `expected_information_gain` SETTLED | the noise floor was reused as the bar for a real question, and 99.9999% carries more entropy than float error |
| `model_horizon` neighbour radius | absolute distance in a feature space whose scale is a per-model fact |
| `unknown_failure` novelty | a latency channel spanning 120–30000 swamped every categorical difference |
| `matched_experiment` VOID_PROVENANCE | refused tasks were filtered out before any trial existed |

Each was found by running the thing rather than reading it. All 61 verdict
members across the 14 new enums are now reachable and asserted by a test.

**Distances that measured whichever channel was largest.** Four, all of the
same shape — a distance or a step over quantities whose scales come from
callers, with no normalisation on the way:

| Where | What dominated |
|---|---|
| `model_horizon` neighbour radius | an absolute distance in a feature space whose scale is a per-model fact |
| `unknown_failure` signature distance | a latency spanning 120–30000 beside categorical differences of 0 or 1 |
| `manipulable_learning` parameter step | a fixed 0.2 on a value of 2 that rounds to an integer |
| `narrative_provenance` state distance | a duration in seconds beside two values in [0, 1] |

The last one is the sharpest: on a generator whose words track valence
perfectly, the measured fidelity was 0.03 against its shuffled null, where the
same text against bounded channels alone scores 0.90. The instrument for
deciding whether introspection says anything was reporting that it does not,
because of a channel nobody was asking about.

I wrote a detector for the class and did not ship it. It caught two of the
four and reported four sites I had to read and dismiss, and the obvious next
move — tuning it against my own four examples — is the failure the epistemic
independence work in this same pass exists to prevent.
`tests/test_distances_are_scale_free.py` is a regression test for the four
instead: precise about what is known to have gone wrong, silent about what is
not.

## Numbers

**Interiority effect bus.** All four channels were independently broken, and
the ledger they appraise had no writers at all. On a real loss event, after:
relevance 0.0 → 0.515, agency absent → 1.0 measured, faculties firing 3 → 6,
affect valence +0.05 → −1.00, and it reaches the affect engine as evidence
rather than as an unsigned nudge.

**Canonical state.** Five subsystems that each owned a private copy of affect
now estimate into one channel. On a live probe they land at valence −0.007
with a spread of 0.32 across four producers, and the disagreement is raised as
an event naming each position rather than averaged away. Private copies of
canonical state: 14 → 5.

**Global workspace determinism.** Ties broke by `set` iteration order. Eight
fresh processes on identical input gave cognitive/cognitive/social/cognitive/
social/cognitive/social/cognitive. They now agree.

**Introspective calibration.** A faithful introspector scores 0.75 against a
shuffled null of −0.02; one that writes the same sentence every time scores
exactly 0.0; random text scores −0.08.

**Matched experiment.** Aura's 27B against the base 27B it was adapted from,
24 procedurally generated tasks, 255 seconds. base 14/24, Aura 15/24, delta
+0.042, p = 1.0, one discordant task in twenty-four. Not attributable — see
[the evidence](../evidence/matched_experiment/README.md) for what that
establishes and what it does not.

**Epistemic independence gate.** Four passes to get honest: 1670 findings, then
6, then 16 after the judged side was expanded through its full derivation —
which was the pass that caught the commonest form, `latest > mean(scores)`.
All sixteen grandfathered sites were read and are legitimate.

## The mistake

`core/cognition/primitive_invention.py` already existed and I wrote over it,
breaking four test files that never mention it. The full suite caught it and
nothing smaller would have. Restored byte-for-byte; my module is
`invention_depth.py`.
