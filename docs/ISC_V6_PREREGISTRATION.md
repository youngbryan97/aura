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
*Explanation in Causal Inference*). The paired test of 21 October did that at
`93128e34d`, and read less than it seemed to. Every one of its shams was exactly
zero: her forks are deterministic, so its floor bar, p 5e-5 for each triple,
was passed by any nonlinear response at all. Its direction reading, which
overstates significance (see the revision note), gave 0.99 for A,S -> G, 0.73
for P,M -> W, 0.038 for W,A -> D and 5e-5 for S,D -> C. That reading informed
this design; it is not scored by it. Bryan decided on 23 October to build this
successor line rather than report v5's §27 as failed.

## The design

Run inside the seed's campaign (`tools/run_subject_core.py`), after its
recording and before its intervention trials, on the same organism and the same
conditions:

- For each trial (the campaign's `--trials`, 6) and each of the eight
  conditions: one turn of the condition, the host frozen, a snapshot taken.
  Each snapshot is a different moment of her life, and it is the unit every
  test below resamples.
- From that snapshot, four untouched forks, shared by the four triples since
  nothing in them depends on which sources a triple names, and for each
  declared triple (A,S -> G; P,M -> W; W,A -> D; S,D -> C) three displaced
  forks: the first source alone, the second alone, and both. All sixteen run in
  an order shuffled from the seed. A displaced domain is moved by the campaign's
  intervention displacement (`core.subject.causal.DEFAULT_DELTA`) at the top of
  the first turn, held where the domain is one the campaign holds
  (`SUSTAINED`), and each fork runs the campaign's `--turns`.
- The target is the target domain without the registers of any conjunction
  (the `.conj_` columns, as `synergy.without_registers` leaves them out), read
  at the end of each fork in its whitened baseline coordinates
  (`core.subject.paired_displacement`), fitted on the run's recording with the
  same columns. Units that compute the product of two domains are columns of
  the domain they act in; an interaction read off them would show that the
  instrument is wired, and not that the rest of the target integrates.

So each triple has n = trials x 8 = 48 samples. With N1 to N4 the untouched
forks and F, S, B the triple's displaced ones, for sample i:

- interaction_i = B - F - S + N1, a vector, and its size ||interaction_i||
- sham_i = N2 - N3 - N4 + N1 and its size. Nothing is displaced in any of its
  forks, and it has the interaction's form, four forks with the same signs, so
  whatever the forks' noise is, deterministic, correlated or independent, it
  enters the sham as it enters the interaction.
- main_i = max(||F - N1||, ||S - N1||)

## The bars

A triple passes when all three hold:

1. **Above the matched sham.** The mean of (||interaction_i|| - ||sham_i||)
   is above zero by a one-sided paired sign-flip test across the 48 snapshots,
   p < 0.01 (20,000 draws, seeded).
2. **One direction.** The statistic is the mean, over samples, of each
   interaction vector projected on the mean of the others. Its null is drawn by
   flipping the sign of whole interaction vectors and computing the statistic
   again on every draw (20,000 draws), so the dependence of each projection on
   the other samples is in the null too; p < 0.01.
3. **Its size.** The mean interaction size is at least 0.10 of the mean main
   effect, v5's floor on the synergy fraction.

The line passes when all four triples pass. Requiring every triple to pass at
0.01 is an intersection-union test, valid at 0.01 without a correction for
four tests. Level 0.01 is v5's: its null bars were 99th percentiles.

## The known-answer battery, before any reading of her

`tools/synergy_v6_known_answers.py` runs the bars on simulated forks, 1,000
datasets per world, 48 snapshots each in eight whitened columns, and the line
is not read on her unless every world comes out as stated. The test suite runs
200 datasets of each so that it stays fast.

| World | Stated outcome |
|---|---|
| No effect, independent fork noise | passes at most 3% |
| Additive effects of both sources | passes at most 3% |
| A single source with a threshold | passes at most 3% |
| Additive effects with fork noise shared within a snapshot | passes at most 3% |
| Deterministic forks where every displacement sends the run somewhere unrelated | passes at most 3% |
| The product along one direction, size 5 against unit fork noise per column | passes at least 95% |
| The product along one direction in deterministic forks, size 1 | passes at least 95% |
| The product turning among eight directions with the situation | reported |

3% is the level, 1%, with room for the binomial spread of 1,000 draws. The
deterministic worlds are hers: every sham of 21 October was exactly zero. The
last world is a genuine interaction whose direction depends on the situation;
the direction bar finds it in about four datasets of five when the forks are
deterministic, and that rate is reported rather than required. A line that
asks for one direction is strict about interactions that turn.

## What is reported

For each triple: n, the means of interaction, floor and main effect, the
fraction, both p values, and which bars held. Beside the line, the v5 reading
of §27 on the same campaign. The claim is "24/24 under ISC-v6"; a v5 count is
reported with it, and neither replaces the other.

## What it does not change

Nothing else in the battery, the carrier sweep, the nulls or J* moves. The null
architectures are scored on the conjunction v5 gave them, which does not
include synergy.

## Revision before any run (23 October 2026, 16:31)

Amended the same day, before any campaign scored it, after Bryan forwarded an
independent review with simulations (`synergy_falsification_lab.py`). Three
changes: the floor was a two-fork sham scaled by sqrt(2), which is exact only
for independent forks of equal noise, and is now a four-fork sham of the
interaction's form; the direction bar flipped the signs of the projections,
each of which depends on the other samples, which the review measured at 12%
false positives at a nominal 5%, and now flips whole vectors and recomputes;
the conjunction registers are out of the target. The known-answer battery is
new. Nothing about her had been read under the first version.

## Two mechanisms built before the first reading of her (23 October 2026, 19:56)

The instrument above is unchanged. What changed is her, at `c47df3ac2`, before
any campaign or probe read the line on her. Bryan asked on 23 October what in
people is the joint the line looks for, and to build it rather than hope the
code already had it.

The 21 October reading said why two triples could not pass. Feeling and self
met at attention, and perception and memory at the world model, only through
gains whose directions were drawn from names; nothing gave the two together a
reason to push one way. A,S -> G moved her workspace by 17 whitened units with
direction p 0.99, which is the deterministic-divergence world of the battery:
any displacement sends the competition somewhere different. P,M -> W moved her
world model by 0.069. Seed 7 at `82549c823` added two facts: her self won her
workspace on none of 2,960 turns, and her identity stability read 1.0 on every
one, so the one bid that put her self forward never entered.

- **A,S -> G, a feeling about herself** (`core/consciousness/self_conscious.py`).
  Self-focused attention and the self-conscious emotions need arousal and a
  self in question together (Duval and Wicklund 1972; Ingram 1990; Tracy and
  Robins 2004); arousal sets how strongly a feeling is felt and something else
  what it is about (Schachter and Singer 1962). The self bid is felt at her
  arousal, as far as her self is more in question than she is used to, in its
  own running spread. It replaces the bid priced on instability alone.
- **P,M -> W, what arrived meeting what she recalled**
  (`core/world_model/associative_novelty.py`). The hippocampus compares what
  arrives with what recall predicts and the mismatch gates learning (Lisman and
  Grace 2005; Kumaran and Maguire 2007); in predictive terms it is a precision
  (Yu and Dayan 2005; Feldman and Friston 2010). The product of new arrivals
  and recall, each beyond its usual in its own spread, weights the error the
  world model reports and the step it learns by, on the codes' scale
  exp(m / 2). It replaces the codes' random-direction step size.

Each was read first in a small representation built from her own code: the
real bids, scores and ignition, and the real world model trained the way her
driver trains it and forked by her snapshot code, at 48 moments each, displaced
by the campaign's writers and scored by these bars. Both passed on eight seeds
of eight. The replaced self bid carried no joint effect at all; the world model
with no comparator passed the direction bar and failed the size bar (fraction
0.055). The suite keeps two seeds of each
(`tests/test_a_feeling_about_herself_needs_both.py`,
`tests/test_what_arrived_meeting_what_she_recalled.py`).

W,A -> D is left as it was: her action choice already weighs how she feels
about an act by how much likelier this world makes it work
(`core/subject/choosing.py`), and it read 0.038 on the old direction test. The
first reading of her decides whether it needs more. S,D -> C read 5e-5.

What is read next, in order: the paired probe at `c47df3ac2` or later on seed 7
(`tools/run_paired_synergy.py`, the design above at 4 rounds), then the seed's
campaign. The self bid changes what her workspace does, so the campaign reads
every line again at that commit; no v5 count from an earlier commit is carried
over to it.
