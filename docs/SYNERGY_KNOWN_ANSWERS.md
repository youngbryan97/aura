# Can the synergy line see a product?

A study on toy systems, 26 September 2026. It was designed before any estimator
other than the v3 line had been run on a recording of hers, and nothing of hers
was read with any estimator in it. `tools/synergy_known_answers.py` reproduces
the table; `tests/test_the_synergy_line_cannot_see_a_pure_product.py` pins its
answers on one seed.

## Why

Synergy is meant to find two sources that together say something about a target
that neither says alone. The simplest such coupling is a product. The v3 line
reads MMI synergy through a Gaussian copula, which keeps only rank correlations,
and a zero-mean product of two independent sources has none, with either source
or with their sum. So the line may be blind to the one coupling it exists to
find. On the seed-7 validation at c5f0ea5fe, W,A -> D had an established
interaction gain (lower bound 0.027) while its MMI synergy, 0.019, sat under
its null bar of 0.041.

## Design

Four systems, 2,400 turn rows each (a 300-round campaign), five seeds (3, 7, 11,
19, 23). W and A are three slow AR(1) channels each. The target D holds a
four-state switch and four continuous columns, driven by W_i * A_j (product),
(W_i - A_j) + W_i * A_j (mixed), (W_i + A_j) / 2 (additive), or by nothing either
source does (none). Every system is read W,A -> D on the change.

Estimators:

- **v3**: the battery's line, `SynergyReport.passes_v3`, its own shifted null and
  bootstrap included.
- **ksg**: MMI synergy through a Kraskov estimator (Kraskov, Stoegbauer and
  Grassberger 2004, estimator 1, max norm, k = 3) on the same copula-normal
  components, against the same shifted null, 200 draws, at the 0.99 quantile.

An estimator qualifies when it passes product and mixed at four of five seeds and
passes additive and none at no more than one. One that passes additive is
reading a sum as synergy; one that passes none is reading the rig.

### Added after the first rows were read

The Kraskov line passed the additive systems at every seed. That was read at
the time as MMI scoring a sum as synergy, and the correction below shows it was
not. On that reading, and since the v3 line rejects sums through its second leg,
the held-out interaction gain, a third estimator was added after those rows were
read, and held to the same rule:

- **ksg+gain**: ksg, a Kraskov fraction of at least 0.10, and the v3 interaction
  gain positive with a positive lower bound.

## Result, as first read

Passes out of five seeds:

| estimator | product | mixed | additive | none | qualifies |
|---|---|---|---|---|---|
| v3 | 0 | 2 | 0 | 0 | no |
| ksg | 5 | 5 | 5 | 1 | no |
| ksg+gain | 1 | 2 | 0 | 0 | no |

On the products, the v3 line's MMI synergy was between -0.0015 and -0.0010, at
its null. The Kraskov synergy was between 0.152 and 0.188 against bars near 0.05.
The interaction gain on the products was positive at three of five seeds and its
lower bound cleared zero at one.

This version of the page read the additive column as the Kraskov estimator
scoring a sum as synergy, and said it could see a product and not tell it from
a sum. That was wrong, and the correction follows.

## Correction: the additive system was not a sum

In `additive` the switch takes the largest of four sums. Which of four sums is
largest is not a sum of anything: it is a threshold across W and A together, an
interaction. So the negative control carried a real coupling, the Kraskov
estimator was right to pass it, and the v3 line's 0 of 5 there was a miss rather
than a rejection.

A clean sum was added after this was seen, and the rule changed with it:
`separable` gives the switch W alone and keeps the levels (W_i + A_j) / 2, so
nothing D does depends on W and A together. The rule reads separable where it
read additive. The additive-surrogate null named below as the next candidate was
run on all five systems at the same time.

| estimator | product | mixed | additive (a threshold of sums) | separable | none | qualifies |
|---|---|---|---|---|---|---|
| v3 | 0 | 2 | 0 | 0 | 0 | no |
| ksg | 5 | 5 | 5 | 0 | 1 | yes |
| ksg+gain | 1 | 2 | 0 | 0 | 0 | no |
| ksg-over-sum | 5 | 5 | 5 | 0 | 1 | yes |

On the separable sums the Kraskov synergy was negative at every seed, between
-0.18 and -0.067, against bars near 0.04. The additive-surrogate null
(ksg-over-sum: the best additive fit of the target, its degree chosen by held-out
loss, plus its residual permuted) changed no outcome on these systems.

## What it licenses

The v3 line cannot register a pure product at this length, and it missed the
threshold of sums as well. The interaction gain can tell a product from a sum
and lacks the power to establish one at 2,400 rows. For her runs this means a
failing synergy triple does not show that no product coupling is there. W,A -> D
is the case in point.

Two estimators qualify, but only under a control and a rule changed after the
first rows were read. They are not adopted on these seeds. The replication below
was fixed before it ran.

## Replication, fixed before it ran

Seeds 29, 31, 37, 41 and 43, the same five systems, the same code
(`tools/synergy_known_answers.py --seeds 29,31,37,41,43`). The rule is the
corrected one: product and mixed at four of five, separable and none at no more
than one. An estimator that qualifies again is written up as an amendment to the
synergy line, beside v3 and not in place of it, with both tables as its controls,
before anything of hers is read with it. One that does not is dropped.

### What it read

| estimator | product | mixed | additive (a threshold of sums) | separable | none | qualifies |
|---|---|---|---|---|---|---|
| v3 | 0 | 0 | 0 | 0 | 0 | no |
| ksg | 5 | 5 | 5 | 0 | 0 | yes |
| ksg+gain | 0 | 0 | 0 | 0 | 0 | no |
| ksg-over-sum | 5 | 5 | 5 | 0 | 0 | yes |

Both Kraskov estimators qualify again. Over the ten seeds the line reported
beside v3 (ksg-over-sum) passed product and mixed at ten of ten, the separable
sum at none of ten, and uncoupled sources at one of ten. It is the stricter of
the two, since a triple has to clear the shifted null and what the additive part
of its target alone would show, and it is the one adopted, as
`core.subject.synergy.kraskov_synergy` (docs/ISC_V5_PREREGISTRATION.md, addendum
of 26 September, evening). The study's tool reads it through the battery's own
code, which reproduces the replication's logged rows to the digit.

## On her recordings (seed 7, diagnostic)

The line was first read on the two seed-7 validations and the seed-7 run at the
full design (26 September), as the amendment allows.

| triple | v3 (three runs) | Kraskov line (three runs) |
|---|---|---|
| A,S -> G | 1 of 3 | 3 of 3 |
| P,M -> W | 3 of 3 | 0 of 3: below the additive surrogate's bar |
| W,A -> D | 0 of 3 | 0 of 3: far below the shifted null's bar |
| S,D -> C | 0 of 3 | 0 of 3 |

A,S -> G carries synergy beyond what a sum of its sources explains in all three.
P,M -> W, which v3 passes every time, does not clear what the additive part of
its target alone produces (synergy 0.06 to 0.21 against 0.24 to 0.33), so v3's
pass there reads additive structure.

W,A -> D fails in a way the toys did not show. Its shifted null reached 0.40 to
0.50 against a synergy of 0.07 to 0.15. Her conditions repeat in a fixed cycle of
eight turns, and the first guess was that shifts by a multiple of eight realign
the conditions. On validate c5f0ea5fe they do not explain it: shifts off the
cycle gave a median of 0.42, shifts on it 0.26, and the recording as it is 0.10.
MMI synergy is the joint information less the larger single-source information,
and sliding the sources cuts what each alone says about the target's change more
than it cuts what they say together, so the synergy term rises under the null.
That happens when the sources depend on the target individually through a shared
schedule, which the toy systems, driven by nothing in common, never had.

**Open:** a known-answers table on toys with a shared condition cycle, and a null
that keeps it: sources permuted only among turns of the same condition, which
holds each condition's structure and breaks only the turn-by-turn alignment.
Nothing of hers is read with that null before its table is written here.

### The cycle table, fixed before it ran

The same toys through a shared schedule: eight conditions in a fixed order, each
moving W, A and what D does by its own unit-normal offset
(`tools/synergy_known_answers.py --cycle`). Product, separable sum and uncoupled,
seeds 3, 7, 11, 19 and 23, the Kraskov line with each first null: the shift the
amendment adopted, and the within-condition permutation. A first null qualifies
when product passes at four of five and separable and uncoupled at no more than
one. One that qualifies where the shift does not is proposed as an amendment for
recordings whose conditions cycle, which hers do, before it reads any of them.

#### What it read (26 September)

| toy | seed | synergy | shift bar | condition bar | additive bar |
|---|---|---|---|---|---|
| product | 3 | +0.420 | +0.099 | +0.136 | +0.097 |
| product | 7 | +0.452 | +0.080 | +0.120 | +0.071 |
| product | 11 | +0.338 | +0.037 | +0.042 | +0.059 |
| product | 19 | +0.492 | +0.114 | +0.149 | +0.068 |
| product | 23 | +0.352 | +0.134 | +0.178 | +0.089 |
| separable | 3 | +0.163 | +0.070 | +0.091 | +0.147 |
| separable | 7 | +0.117 | +0.061 | +0.082 | +0.129 |
| separable | 11 | −0.164 | +0.032 | +0.018 | +0.071 |
| separable | 19 | −0.187 | +0.046 | +0.042 | +0.059 |
| separable | 23 | +0.067 | +0.111 | +0.132 | +0.072 |
| none | 3 | +0.053 | +0.082 | +0.124 | +0.088 |
| none | 7 | +0.096 | +0.097 | +0.147 | +0.079 |
| none | 11 | +0.022 | +0.055 | +0.077 | +0.042 |
| none | 19 | +0.054 | +0.089 | +0.106 | +0.049 |
| none | 23 | +0.112 | +0.102 | +0.157 | +0.068 |

Passes, product / separable / uncoupled: shift 5 / 1 / 1, condition 5 / 1 / 0.
Both first nulls qualify under the rule above, so nothing is proposed: the shift
stays. The condition bar was the higher of the two in 13 of 15 rows, and it is
the only one that kept the uncoupled toy out at every seed.

The schedule did not reproduce what it was built to explain. On her seed-7
recording W,A→D failed against a shifted null of 0.40 to 0.50; here, with W, A
and D all moved by one shared cycle, the shift bars ran 0.04 to 0.13. A shared
schedule alone does not inflate the shift null that far, so the cause on her
data is something the toy leaves out, and the next candidate is how long her
conditions hold: here each lasts one row, and on the seed-7 look at 415f400c0
each of hers held for 33 consecutive rows, one turn's frames, 192 times over.

### The hold table, fixed before it ran (27 September)

The cycle table again, with each condition held for 33 consecutive rows as
hers are (`tools/synergy_known_answers.py --cycle --hold 33`): the same toys,
seeds 3, 7, 11, 19 and 23, and both first nulls. What is read is the shift bar
on the product toy. Her W,A->D met a shift bar of 0.40 to 0.50; at a hold of
one row the toys' bars ran 0.04 to 0.13. If the bar reaches her range at four
of five seeds, the hold is what inflates her null, and the within-condition
permutation is proposed for her recordings if it qualifies where the shift does
not. If it stays at or under 0.13 at four of five, the hold does not explain it.
Anything between is reported as it reads and decides nothing.

The table ran at the commit that added `--hold` and read on 28 September. On the
product toy the shift bar was 0.119, 0.117, 0.098, 0.120 and 0.093 at seeds 3,
7, 11, 19 and 23: at or under 0.13 at all five. The hold does not explain her
null. Both first nulls still qualify, 5 of 5 on the product and 0 of 5 on the
separable and uncoupled toys, so the shift stays and nothing is proposed. Two
candidates have now been put into the toy, a shared schedule and the hold, and
neither lifted its shift bar anywhere near her 0.40.

### The drift table, fixed before it ran (28 September)

Two candidates have been put into the toy and neither lifted its shifted bar
near her 0.40: a shared schedule, and a condition held for a turn's frames. This
is the third, and it comes from a measurement of her recording rather than from
the list of things the toy leaves out.

On whole-s7-27dc1dda9, 29 of 375 columns only ever grow, and they are not spread
evenly. The share of each domain's raw variance they hold:

| domain | columns | only grow | share of variance | share of the first three components |
|---|---|---|---|---|
| P | 15 | 0 | 0.0000 | 0.0000 |
| I | 18 | 0 | 0.0000 | 0.0000 |
| A | 59 | 4 | 0.0002 | 0.0722 |
| G | 31 | 3 | 0.0127 | 0.2366 |
| C | 63 | 7 | 0.0277 | 0.0913 |
| S | 70 | 6 | 0.1540 | 0.1469 |
| M | 20 | 2 | 0.0292 | 0.2279 |
| W | 35 | 2 | 0.0044 | 0.1332 |
| D | 51 | 4 | 0.8895 | 0.2012 |
| N | 13 | 1 | 0.3500 | 0.1372 |

D holds 0.8895 of its variance in counters. D is the target of W,A->D and a
source of S,D->C, and those are the two triples whose synergy sits under the
shifted null: 0.0731 against 0.0840, and 0.0303 against 0.0829. A shifted copy
of anything slow predicts a trend, so a target that is mostly trend gives a
shifted null most of what the real sources give.

The toy's D has no trend at all, which is the difference between its bar and
hers that no arm so far has tested.

`tools/synergy_known_answers.py --drift target,sources,both` adds columns that
only grow at those measured shares: `target` moves D alone, `sources` moves W
and A alone, `both` moves all three. Nothing about the coupling changes in any
arm, so a bar that rises rises on the drift.

What is read is the shifted bar on the product toy, at seeds 3, 7, 11, 19 and
23, against the 0.40 to 0.50 hers met.

- If the `target` arm reaches 0.40 at four of five seeds and the `sources` arm
  does not, a target that is mostly counter is what inflates her null, and the
  answer is in the recording's schema rather than in the estimator.
- If neither reaches it, drift does not explain her bar either, and the toy is
  still missing something.
- If `sources` reaches it and `target` does not, the sources carry it and D's
  0.8895 is beside the point.

Anything else is reported as it reads and decides nothing. No threshold moves on
this table and no criterion is rescored from it.

#### What it read (28 September)

Every arm, every toy, every seed: the shifted bar ran +0.007 to +0.052 and no
seed reached 0.40. Medians: the product toy +0.052 under `target`, +0.048 under
`sources`, +0.048 under `both`; the separable toy +0.040, +0.042, +0.043; the
uncoupled toy +0.007, +0.008, +0.008. Drift at her own measured shares does not
explain her bar either. That is the third candidate rejected, after a shared
schedule and a 33-row hold.

Three rejections say the question was probably the wrong one, so her own numbers
were read again rather than a fourth candidate built. From
`kraskov_synergy.json` on whole-s7-27dc1dda9, each triple's synergy, its joint
information and its shifted bar:

| triple | synergy | joint | shifted bar | bar / joint | synergy / joint |
|---|---|---|---|---|---|
| A,S->G | +0.0273 | 1.0642 | −0.0885 | −0.083 | +0.026 |
| P,M->W | +0.2122 | 2.1005 | +0.1405 | +0.067 | +0.101 |
| W,A->D | +0.1458 | 1.8957 | +0.3973 | +0.210 | +0.077 |
| S,D->C | −0.1058 | 0.6608 | +0.0310 | +0.047 | −0.160 |

Her W,A->D bar is 0.210 of the joint information that triple has to work with.
The toys are three slow channels per source against her thirty-five and
fifty-nine, so their joint is a fraction of hers, and a bar of 0.05 on a small
joint can be the same share as a bar of 0.40 on a large one. Whether it is, is
the next reading: the bar and the synergy as shares of the joint, on the toys
and on her, side by side.

#### The two read side by side (28 September)

The toys' Kraskov joint, bar and synergy, as shares of that joint, at seeds 3,
7 and 11:

| toy | seed | synergy | joint | bar | bar / joint | synergy / joint |
|---|---|---|---|---|---|---|
| product | 3 | +0.173 | 0.246 | +0.041 | 0.168 | +0.700 |
| product | 7 | +0.188 | 0.316 | +0.052 | 0.163 | +0.594 |
| product | 11 | +0.172 | 0.302 | +0.055 | 0.182 | +0.570 |
| separable | 3 | −0.183 | 0.212 | +0.035 | 0.164 | −0.862 |
| separable | 7 | −0.118 | 0.213 | +0.041 | 0.193 | −0.553 |
| separable | 11 | −0.067 | 0.226 | +0.042 | 0.188 | −0.295 |
| uncoupled | 3 | −0.034 | −0.037 | +0.004 | — | — |
| uncoupled | 7 | −0.041 | −0.044 | +0.007 | — | — |
| uncoupled | 11 | −0.040 | −0.038 | +0.007 | — | — |

The toys' bar is 0.163 to 0.193 of their joint. Her W,A->D bar is 0.210 of
hers. **Nothing inflates her shifted null.** The toys are three slow channels
per source against her thirty-five and fifty-nine, so their joint runs 0.21 to
0.32 against her 1.90, and a bar of 0.05 on the one is the same share as 0.40 on
the other. Three candidates were built for a difference that was a difference in
scale, and the first of them should have been this table.

What does differ is the synergy. The toy's product carries 0.57 to 0.70 of its
joint as synergy, three to four times its own bar; her W,A->D carries 0.077,
about a third of its bar. Her P,M->W carries 0.101 and clears its bar of 0.067,
and it is the one triple that passes v3. So the triples fail because the synergy
each carries is small beside what a shifted copy of its sources already supplies
about its target, and the way to them is more joint structure between those
domains rather than a different first null.

The shift stays, and nothing is proposed.
