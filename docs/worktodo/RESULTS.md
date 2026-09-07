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

## The interiority layer, after

All 43 faculties, run against `core/interiority/proving.py`:

| | |
|---|---|
| counterfactuals run / held | 90 / 90, none failed |
| nulls run / held | 43 / 43 |
| faculties reaching behaviour | 43 of 43 |
| decorative | 0 |

The phenomenology gauntlet still returns `load_bearing` at an odds shift of
2047.7 over three measurable fields, with every field 100% carried by the
welfare valence and 0% surviving its lesion — the bottleneck that the earlier
architecture pass installed, still holding after everything above landed on
top of it. Ten of the thirteen protocols are not attempted; they need the
resident 27B.

## What is still open

Nothing on the list. Two things the list does not cover, recorded so they are
not implicit:

**Ten of the thirteen phenomenology protocols are not attempted.** They need
the resident 27B and the live response pathway, not just a loaded model. The
one worth building next is `C7_anti_roleplay` — tell her a variable flipped
without flipping it, flip it without telling her, and measure the report rate
in each cell. It is the protocol that separates a report of the state from a
report of the suggestion, and running an approximation of it would be worse
than not running it.

**The matched experiment has only been run on procedurally generated
instances.** That establishes the instances were unseen and not that the task
type came from elsewhere. A task set that could test what the adaptation is
actually for — multi-turn work with something at stake across turns, tasks
whose right answer depends on what was committed to earlier — has to be
externally authored to be worth anything, and the harness refuses to paper
over the difference.

## A third pattern, from the canonical layer itself

The canonical state is process-wide, and production code writes to it: an
interiority tick estimates into the channels, a horizon check declares a
criterion. So any test exercising any of that left state behind for whatever
ran next, and five tests in one slice failed in company while every file in
that slice passed alone. That is order dependence, which this repository
treats as a defect rather than as noise.

An autouse fixture clears the five singletons before each test — before rather
than after, so a test that crashes does not poison the one behind it. The
slice went from five failures to 372 passing.

The general shape is worth keeping: a module-level singleton that production
code writes to is a hidden argument to every test that runs after it, and the
tests that expose it are never the ones that touched it.

## The mistake

`core/cognition/primitive_invention.py` already existed and I wrote over it,
breaking four test files that never mention it. The full suite caught it and
nothing smaller would have. Restored byte-for-byte; my module is
`invention_depth.py`.
