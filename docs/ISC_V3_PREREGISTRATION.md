# ISC-v3, preregistered

Written on 15 September 2026, before any ISC-v3 result exists, and committed with
the code that computes it. ISC-v1 and ISC-v2 are kept. Every line of both is
computed and reported on every run, and a failure under either stays a failure.

ISC-v3 is ISC-v2, amendments included, with the synergy line changed. The
conjunction judges the null architectures by the same lines as Aura, so the
change also reaches the two places that read a null's synergy: section 1's
comparison set and section 3's null line.

## Why a new version

Every amendment to ISC-v2 was recorded before any v2 result existed. This change
was not. Aura's synergy on the change was computed on two 300-round recordings
while the v2 line's power was measured (`9ba2d949e`), and no triple passed. A
criterion may only be made stricter after its results have been seen, and this
change makes the line easier to pass against slow sources, so it cannot amend
v2. As a new version it reads only recordings from which no synergy-on-change
number has been read.

## What v2 asks, and what it cannot see

v2 asks, of each declared triple, for synergy about the target's change that
clears:

- a floor of 0.10 on the synergy fraction;
- the 99th percentile of its shifted null, on the fraction and on the raw value,
  with the fraction's margin at least the null's spread;
- a positive held-out interaction gain, with the lower bound of its mean over
  the forward folds above zero.

The components of Aura's source domains have lag-one autocorrelations between
0.98 and 0.998. On synthetic recordings of 2,400 rows, the length of a 300-round
run, with sources at 0.99, a coupling of the preregistered form added to the
target's change at two of its own spreads passed v2 on 0 of 25 seeds. The bar
that failed was the fraction's. With the fraction's null built from a bootstrap
instead of the shift, the raw synergy cleared its null, 0.128 against 0.056, and
the interaction bound was +0.109, while the fraction read 0.347 against 0.555
(docs/SUBJECT_CORE.md). The fraction divides one small quantity by another, and
against slow sources its null stays wide. On such sources a v2 failure cannot be
told from the line's own blindness.

## What v3 asks

Synergy about the target's change, from the same estimator on the same
components, clearing:

- the floor of 0.10 on the fraction, as in v2;
- the 99th percentile of the shifted null on the raw value, as in v2;
- the 99th percentile of a bootstrap null on the raw value, by at least that
  null's spread;
- the interaction gain above zero and its fold lower bound above zero, as in v2;

with at least two draws of each null. This is
`core.subject.synergy.SynergyReport.passes_v3`.

The two bars removed are the fraction's bars against the shifted null. The
bootstrap bar keeps their form, a 99th percentile cleared by the null's own
spread, and applies it to the raw value.

The bootstrap null fits the two sources' components together as one first-order
vector autoregression with an intercept, and simulates them 200 times from the
recording's first row with Gaussian shocks of the fitted residual covariance. A
simulation keeps each source's persistence and the relation between the two
sources, and carries nothing about the target. Its draws come from a random
stream of their own, so the shifted null reads exactly what v1 and v2 read. A
fit with a spectral radius of one or more is kept and its radius reported: its
simulations wander further and widen the null. A simulation that overflows is
dropped.

## The nulls

Since the fourth amendment to ISC-v2, v2 judges a null architecture by synergy
on the change when it builds section 1's comparison set and when it reads a
null's conjunction for section 3. v3 judges it by the v3 line in both places, so
Aura and every null are held to one synergy line. The runner records
`synergy_v3_passes` for every architecture, the v3 comparison set, and
`conjunction_v3`. A null table without the v3 reading cannot be read under v3,
and both null lines fail on it for want of a measurement.

A null that passes the v3 line where it failed v2's can join the comparison set
and can pass its conjunction, and either makes a null line harder for Aura. The
recurrent reference can also pass a seed it failed under v2, which makes the
instrument less likely to fail.

## Known answers, measured before the line read any recording of Aura

Synthetic recordings only, scored with the committed code, and a subset pinned
in `tests/test_synergy_known_answers_for_isc_v3.py`. Before the code was
committed, an exploratory run on a random stream of its own compared this line
with the same line less the fraction floor. The two gave the same answer on
every recording, and the stricter one was kept.

ISC-v2's known answers, on battery-shaped recordings of 479 rows and ten seeds
each: a product under drift passes on all ten, and an additive target and drift
alone pass on none. v2 gives the same counts.

The power backgrounds have ten domains of three columns, each column a
first-order autoregression at the stated persistence. A coupling is added to the
target's first change component, sized in that component's spread. The
preregistered form is (a−b)+a·b of the two sources' first components; the
additive form is a−b. Passes over seeds, at 2,400 rows:

| sources' persistence | coupling | v2 | v3 |
|---|---|---|---|
| 0, 0.9, 0.99, 0.998 | none | 0 of 50 | 0 of 50 |
| 0.9 | preregistered, 1 spread | 5 of 5 | 5 of 5 |
| 0.99 | preregistered, 1 spread | 0 of 25 | 22 of 25 |
| 0.99 | preregistered, 2 spreads | 0 of 25 | 25 of 25 |
| 0.998 | preregistered, 1 spread | 1 of 5 | 3 of 5 |
| 0.998 | preregistered, 2 spreads | 0 of 25 | 16 of 25 |
| 0.99 | additive, 2 spreads | 0 of 5 | 0 of 5 |
| 0.99 | additive, 4 spreads | 0 of 25 | 0 of 25 |
| 0.998 | additive, 4 spreads | 0 of 5 | 0 of 5 |
| 0.99 | a·b alone, 2 spreads | 0 of 5 | 0 of 5 |

And at 479 rows:

| sources' persistence | coupling | v2 | v3 |
|---|---|---|---|
| 0, 0.9, 0.99, 0.998 | none | 0 of 100 | 0 of 100 |
| 0.9 | preregistered, 1 spread | 0 of 5 | 3 of 5 |
| 0.99 | preregistered, 1 spread | 0 of 5 | 0 of 5 |
| 0.99 | preregistered, 2 spreads | 1 of 45 | 14 of 45 |
| 0.99 | preregistered, 4 spreads | 1 of 20 | 12 of 20 |
| 0.998 | preregistered, 1 or 2 spreads | 0 of 10 | 0 of 10 |
| 0.99 or 0.998 | additive, 2 or 4 spreads | 0 of 15 | 0 of 15 |
| 0.99 | a·b alone, 2 spreads | 0 of 5 | 0 of 5 |

**What the line still cannot see.** A product of the two sources alone passed on
none of ten recordings. A Gaussian copula estimate of information is nearly
blind to it, and v3 changes the null and leaves the estimator as it was. At 479
rows the line registered two spreads on 14 of 45 seeds, and at a persistence of
0.998 and 2,400 rows on 16 of 25. A v3 failure on a triple whose sources are that
slow is weak evidence about the triple.

## Checking the line on her recordings

After this commit, on each run read under v3, the preregistered coupling is added
to each declared triple's target at one and two spreads on the run's own
recording, and the line is scored. A triple whose line does not register two
spreads there is reported, beside its result, as one the instrument could not
see. Its result stays its result.

## Which runs it reads

- A run whose synergy-on-the-change rows carry `passes_v3` is read from its
  report.
- The ISC-v2 campaign recording at `1a9ebe561` (campaign
  `39493ebf6330169637c46feb8febc62f`, seeds 7, 11 and 13) had written no report
  when this was committed. Each of its runs is rescored beside its report, with
  the code as committed here, and marked as added after the run: Aura's triples
  from the run's saved recording, and each null architecture's from the toy
  recording its architecture and seed produce. The rescoring first checks that
  it reproduces the nulls' recorded v2 synergy, and reads nothing under v3 for a
  run where it does not.
- No other recording. The 300-round A/B recordings whose synergy on the change
  was read, and every ISC-v1 run, are not read under v3.

As in ISC-v2, v3 holds when every line holds on every run of the declared seeds,
and its null line is decided across those seeds.

## What does not change

Every other line of ISC-v2 as amended, every threshold in `THRESHOLDS`, the
estimator, the shifted null, the interaction check, and the rule that a
criterion may only be made stricter after its result has been seen, which binds
v3 from this commit on.
