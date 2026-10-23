# ISC-v6 preregistration: synergy read by setting the sources

Written 23 October 2026, before any run that it scores. ISC-v6 is ISC-v5
(docs/ISC_V5_PREREGISTRATION.md, every addendum included) with one line
replaced: §27, synergy. Every other line, bar, design and null is v5's, and
every v6 report carries the v5 reading of §27 beside it.

## Why the line changes

v5 reads synergy by watching: over a run, whether two domains' leading
components carry information about a third domain's next change that neither
carries alone, with a held-out test that a model with products of the sources
predicts that change better than one that adds them. Across the pairs of 21 to
23 October every triple read between about 0.8 and 1.3 times its shifted null,
and at `82549c823` the two triples with the self as a source failed:

- S is read through its three leading components, and thirty of its columns
  trend together over a run, so those components are her drift. Her self as it
  stands in a turn, added that morning, lands in the third.
- In ordinary life the sources move together, driven by the same situations,
  so watching cannot tell a joint effect from two effects side by side. Within
  condition, A,S -> G, P,M -> W and W,A -> D carried synergy far above their
  shift nulls, and none had a held-out interaction gain above zero.
- C is recorded as summaries over whole networks, and the units that meet self
  and intention drive one band of the substrate.

An interaction is defined by what happens when the causes are set: the effect
of one depends on the level of the other (Rothman 1976; VanderWeele 2015,
*Explanation in Causal Inference*). The paired test of 21 October did that and
found all four triples interacting (p 5e-5 each, `93128e34d`). That reading
informed this design; it is not scored by it. Bryan decided on 23 October to
build this successor line rather than report v5's §27 as failed.

## The design

Run inside the seed's campaign (`tools/run_subject_core.py`), after its
recording and before its intervention trials, on the same organism and the same
conditions:

- For each trial (the campaign's `--trials`, 6) and each of the eight
  conditions: one turn of the condition, the host frozen, a snapshot taken.
- For each declared triple (A,S -> G; P,M -> W; W,A -> D; S,D -> C), five forks
  from that snapshot in an order shuffled from the seed: neither source
  displaced, the first alone, the second alone, both, and neither again. A
  displaced domain is moved by the campaign's intervention displacement
  (`core.subject.causal.DEFAULT_DELTA`) at the top of the first turn, held where
  the domain is one the campaign holds (`SUSTAINED`), and each fork runs the
  campaign's `--turns`.
- The target domain's state at the end of each fork is read in the target's
  whitened baseline coordinates (`core.subject.paired_displacement`), so a size
  is the distance v5's estimators read.

So each triple has n = trials x 8 = 48 samples. For sample i:

- interaction_i = ||both - first - second + neither||
- floor_i = sqrt(2) x ||again - neither||. Fork noise enters the interaction
  through four forks and the sham through two; for independent noise of equal
  spread the interaction's noise is sqrt(2) times the sham's, so without the
  factor a system with no interaction at all would clear the floor.
- main_i = max(||first - neither||, ||second - neither||)

## The bars

A triple passes when all three hold:

1. **Above the floor.** The mean of (interaction_i - floor_i) is above zero by
   a one-sided paired sign-flip test, p < 0.01 (20,000 draws, seeded).
2. **One direction.** Each sample's interaction vector, projected on the mean
   of the other samples' vectors, is above zero by the same test, p < 0.01. A
   joint effect that points a different way every time is noise of a larger
   size.
3. **Its size.** The mean interaction is at least 0.10 of the mean main effect,
   v5's floor on the synergy fraction.

The line passes when all four triples pass. Level 0.01 is v5's: its null bars
were 99th percentiles.

## What is reported

For each triple: n, the means of interaction, floor and main effect, the
fraction, both p values, and which bars held. Beside the line, the v5 reading
of §27 on the same campaign. The claim is "24/24 under ISC-v6"; a v5 count is
reported with it, and neither replaces the other.

## What it does not change

Nothing else in the battery, the carrier sweep, the nulls or J* moves. The null
architectures are scored on the conjunction v5 gave them, which does not
include synergy.
