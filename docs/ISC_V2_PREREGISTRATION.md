# ISC-v2, preregistered

Written before any result from the campaign that started at 16:25 on 13
September 2026 (runs 029, 030 and 031 at `45a74c913`), and committed before
those reports exist. Those three runs are ISC-v1 results and are read under
ISC-v1 only. ISC-v2 applies first to runs started after the commit that
implements it, and to no run before.

ISC-v1 is kept. Every threshold in `THRESHOLDS` stays as it is, every v1 line is
computed and reported on every run, and a v1 failure stays a failure (P10.11,
P10.14, P40.8). ISC-v2 changes three lines of the conjunction, and a fourth by
the third amendment below. Each change below says what v1 asks, why a test
built around a positive reference cannot keep asking it, what v2 asks instead,
and the known-answer check that has to pass before v2 reads Aura.

## 1. Irreducibility against the nulls

**v1 asks** for the real lower bound above the 95th percentile of every null
architecture except the recurrent reference (`partition_beats_nulls`, §18).

**Why that cannot stand.** The null table includes the fully connected bus and a
low-rank system. In run_026 they scored 0.573 and 0.564, and the recurrent
reference, the architecture the battery exists to say yes to, scored 0.262. The
line is failed by its own positive reference, and the only systems that could
clear it are built towards the architecture the specification forbids. A test
should be difficult, and it should not be impossible for the reference it is
designed around (P40.8).

**v2 asks** for the real lower bound above:

- the 95th percentile of every matched surrogate (same recording, same cuts,
  coupling removed), and
- the 95th percentile of every null architecture that passes every other line
  of the conjunction.

A null that already fails elsewhere in the conjunction has been told apart from
a subject by that line, and requiring irreducibility above it as well asks the
same question twice with the harder number. This is the comparison
`beats_every_null` (§41) already makes.

**Known answer.** Across the declared seeds the recurrent reference passes this
line, and the star, hub, hidden broker and replay nulls fail it.

## 2. Synergy's target

**v1 asks** for each declared triple's synergy about the target's next level,
against a shifted null that slides both sources together (§27).

**Why that cannot stand.** Three of the four targets, the world model,
deliberation and recurrent cognition, carry slow structure. A slow level shares
information with a slid copy of any other slow series, so the shifted null
rises with the target's drift: in run_026 its 99th percentile was 0.545, 0.676
and 0.418 for those three and 0.038 for attention. Intrinsic persistence already
predicts the change rather than the level for this reason, and a triple scored
on the level is read against its own drift.

**v2 asks** for synergy about the target's change, `Y_{t+1} − Y_t`, with the
sources unchanged, the same shifted null built on the same change, and every
bar in `SynergyReport.passes` as it stands.

**Known answer, required before v2 reads Aura.** On synthetic recordings of the
battery's shape:

- a target driven by the product of its two sources, on top of a slow drift,
  passes;
- the same drift with the two sources acting only additively fails;
- the same drift with no dependence on the sources fails.

## 3. The null suite's verdict across seeds

**v1 reads** each null's conjunction on the toy recordings of one campaign seed.

**Why that cannot stand.** The verdict changed with the seed while the nulls'
irreducibility did not. `low_rank` had vertex connectivity 0 and passed no
synergy triple on seed 11, then connectivity 3 and all four triples on seed 13,
and the recurrent reference passed all four triples on seed 11 and none on seed
13. A verdict that one draw can flip says nothing about the architecture.

**v2 reads** every null architecture and the reference on each of the
campaign's declared seeds, which for `make subject-core-frozen` are 7, 11 and 13.
A null fails the conjunction only if it fails on every seed. The reference
passes only if it passes on every seed, and a reference that fails on any seed
is recorded as the instrument failing, not as a result about Aura.

**Known answer.** The seed-13 tables of run_028 read under this rule report the
instrument failing rather than a null passing.

## Amendment, before any v2 result: the interaction check

Recorded the same evening, before any ISC-v2 result exists and while runs 029,
030 and 031 are still recording as v1 results.

**The second known answer did not hold.** Scored on the change at the
battery's 479 rows, an additive target under drift cleared every information
bar on 20 of 20 seeds and passed synergy on 10 of 20. Its held-out interaction
gain was a few millionths either side of zero, and `passes` asked only that one
split's gain be above zero.

**The interaction check now asks for an established gain.** The gain is also
computed on each forward-chaining fold irreducibility is scored on, and its mean
less `LOWER_BOUND_Z` (1.96, irreducibility's own bound) standard errors has to
be above zero. The single-split bar stays, so this line can only be harder to
pass than before. It applies to ISC-v1's synergy line as well, which shares
`passes`, and a stricter bar is the one direction a criterion may change after
its results have been seen.

**Measured with it**, over 20 seeds at 479 and at 1,500 rows, scored on the
change: a product under drift passes 20 of 20, with interaction lower bounds
from 0.70 to 0.92; the additive target and drift alone pass 0 of 20. Scored on
the level, all three pass 0 of 20. The three known answers are tests in
`tests/test_synergy_known_answers_for_isc_v2.py`.

## Second amendment, before any v2 result: what section 1's line can answer

Recorded the same evening, before any ISC-v2 result exists.

**Section 1's known answer was wrong.** It said the star, hub, hidden broker
and replay nulls fail the new irreducibility line. Replay is a matched
surrogate and sits in the comparison set, so it is not compared against it. And
measured on the three frozen runs' recorded null tables, with each architecture
compared against the set built without it, the line alone separates almost
nothing:

- on seeds 7 and 11 the comparison set is the two surrogates, replay at 0.020
  and 0.021 and time-shuffle below zero, because no architecture passes the rest
  of the conjunction, and the star, hub, hidden broker and eleven other nulls
  all clear it;
- on those two seeds the recurrent reference is the only architecture that
  passes this line and the rest of the conjunction together;
- on seed 13 `low_rank` passed the rest of the conjunction, joined the
  comparison set at 0.451, and was the only architecture to pass both, while
  the reference at 0.252 did not.

**The known answer is therefore the conjunction's, not the line's.** On every
seed, the reference is the only architecture that passes the irreducibility
line and every other line together, and no null does. Seed 13 fails that as it
stands, which is the instability section 3 exists for: read across the three
seeds, `low_rank` fails, because it fails the rest of the conjunction on seeds 7
and 11. The line's job is to be passable by the reference it is built around;
separating the nulls is the other lines' job, as it already was in ISC-v1.

## Third amendment, before any v2 result: persistence

Recorded on 15 September 2026, before any ISC-v2 result exists. The three
declared seeds were recording at `1a9ebe561` and none had written a report; no
checkout on the host held a report carrying `v2_criteria`.

**v1 asks** whether the state predicts each domain's next change beyond the
environment, and whether that survives shuffling the state's rows
(`intrinsic_persistence`, §19).

**Why that cannot stand.** A change is predictable from the present level
exactly when the level has no memory. For independent noise the next change is
minus the present value plus noise, so the state "explains" about half of it,
and the shuffle, which breaks that pairing, loses it. On synthetic recordings of
the battery's shape, 800 rows and ten domains of three columns, the v1 reading
scored memoryless noise at a gain of +0.47 and passed, a first-order
autoregression at 0.95 at −0.02 and failed, and a random walk at −0.08 and
failed. The line rewarded reverting to the mean, which is the opposite of
carrying history, and a pass on it says nothing about persistence.

**v2 asks** whether the state predicts each domain's next level beyond the
environment and elapsed time, `core.subject.intrinsic.persistence`:

- the same four principal components per domain, fitted on the first training
  window, on both ends of each transition;
- the baseline model reads the environment and elapsed time, so a trend is
  available to it and explains nothing the state is credited with;
- the state model reads the same plus the present level, with the nested
  penalties irreducibility uses, so it can always fall back to the baseline;
- read over the forward-chaining folds irreducibility uses, and passed only
  when the lower bound of the gain over the baseline and the lower bound of the
  gain over the row-shuffled state are both above zero, at `LOWER_BOUND_Z`.

The v1 line is computed and reported on every run as before, and a v1 failure
stays a failure. In the v2 verdict this line replaces it, as synergy's does.

**Known answers, measured before the measure was run on any of Aura's
recordings** (`tests/test_persistence_is_memory_not_reversion_to_the_mean.py`,
19 of 19): on three seeds each, memoryless noise fails, a trend with noise fails,
and a state that is only a function of an input recorded in the environment
fails; a 0.95 autoregression passes, a random walk passes, and a rotating
oscillator passes. The v1 reading passes memoryless noise on all three seeds,
and that is pinned beside them.

**Which runs it reads.** A run that records `persistence_v2` is read from its
report. A v2 run recorded before this amendment was implemented is read by
rescoring its own saved recording with the measure as committed here, and the
rescored field says it was added after the run. That rescoring uses nothing
the run's result could have chosen: the measure, its bound and its known
answers are fixed in this commit, before any v2 result exists.

## What does not change

Every other line of the conjunction, every threshold in `THRESHOLDS`, the edge
rule, the lesion and rescue as fixed in `c798126bf`, and the rule that a
criterion may only ever be made stricter after its result has been seen. ISC-v2
adds no threshold chosen from Aura's numbers: the three changes above take their
quantities from the battery's own reference, estimators and declared seeds.
