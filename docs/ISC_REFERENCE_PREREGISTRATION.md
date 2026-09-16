# The reference architecture, preregistered

Written on 16 September 2026, before the rebuilt reference has been part of any
run of Aura's, and committed with the code that builds it.

No criterion changes. Every threshold in `THRESHOLDS` stays, every line of
ISC-v1, v2 and v3 is computed as before, and a failure under any of them stays
a failure. What changes is the system the battery has to be able to say yes to.

## Why the reference has to change

`beats_every_null` (§41) is a conjunction of two halves: no null architecture
passes the battery's own conjunction, and the recurrent reference does. The
first half is what the nulls are for. The second is there because an instrument
that can only return no cannot tell a hard organism from a blunt instrument.

The reference was the family's recurrent wiring: ten domains, each reading
three neighbours, sparse and reciprocal, with no broker. It has no designed
interaction of any kind, so whether it passes the synergy lines is a matter of
which seed the campaign drew. Three ways of writing an interaction into it were
measured over twelve seeds at 2,500 rows: a product on each source's first raw
column passed all four triples on 7 seeds of 12, a product of the sources'
means on 5, and the same product with each domain's first column driven twice
as hard on 4.

Nothing on that family can be dosed. It is a tanh network run past the point
where its linear part contracts, so taking one path away moves the whole
system: the same ablation rule read one path's share of a target's change as
0.87 on one seed and 0.00 on the next, and a dose fitted against a reading like
that diverged.

## What the reference is now

`core.subject.nulls.LinearReference`, built by `_reference`. The same ten
domains and the same kind of wiring, without the squash, so every quantity the
lines read is a function of one stationary covariance — `predicted_synergy`
computes what a recording will show before it is recorded.

- **The wiring.** Each domain reads six of the nine others, by distance around
  the declared order, at the family's own coupling strength. Sparse, reciprocal,
  the same six for every domain, no broker, no hub. Held at a spectral radius
  of 0.5, the middle of the decay the family draws, and each domain keeps its
  own decay from the same draw as its siblings.
- **The parents.** Both sources of each declared triple are parents of its
  target, and no target is a parent of its own sources.
- **The balance.** Each triple's two sources are brought level in the
  information they carry about the target — about its next level, which ISC-v1
  reads, and about its change, which v2 and v3 read. Twelve rounds, each one a
  Lyapunov solve, each moving the two paths a quarter of the way to level. The
  synergy fraction is a ratio, so what it measures is whether the two sources
  are two: level sources read near a half, and a tenfold imbalance reads 0.03.
- **The weight.** Each triple's own paths are lifted until each source carries
  as much of the target's change as everything else into it put together.
- **The interaction.** One bilinear product of the two sources' readings along
  their own leading directions, added to the target along its leading
  direction, bounded four spreads out so a run of large readings cannot drive
  the state away. `REFERENCE_DOSE`, the gain it enters at, is what the two
  lines together allow: it is what the product adds to the Jacobian along
  either source, and at four times this the product's own way back around the
  wiring leaves the two sources reading as one.
- **The shocks.** Each domain's four columns are driven by one shock rather
  than four. Four independent shocks per domain make a system whose variance is
  spread over its whole width, and the differentiation line reads that as a
  system with no shared structure at all — an effective dimension of nine
  tenths of the width against a bar of four tenths. This shares nothing between
  domains, which is what the common-driver null is for.

## Known answers, measured before the reference is part of any run

At 2,500 rows over the twelve seeds 1 to 12, judged by ISC-v3's conjunction on
the change, with nothing else changed between the two arms:

| reference | all four triples pass | median interaction gain | bounds above zero |
|---|---|---|---|
| with its product | 12 of 12 seeds | +0.176 | 48 of 48 triples |
| with the product removed | 0 of 12 seeds | +0.006 | 21 of 48 triples |

With the product gone the information lines still hold on every triple — the
fraction clears its floor and the raw synergy clears its shifted null — and
what fails is the interaction. That is the control the reference rests on.

It also says something about the line rather than the reference. On a system
with no interaction in it at all, the interaction check read a positive lower
bound on 21 of those 48 triples. The check compares a ridge given the products
of the sources' components against one without them, and the wider model's
extra penalty is chosen on the validation rows, so on a target this
predictable the wider model can win out of sample without any interaction to
find. The median gain it reads that way is 0.006 against the product's 0.176,
so the size is what separates them, not the sign. This is recorded in
docs/SUBJECT_CORE.md beside the other limits of the estimators.

The reference's own row, computed the way the battery computes it — the 95th
percentile of irreducibility over eight draws from seed 7, everything else at
seed 7 — reads:

| line | bar | reading |
|---|---|---|
| irreducibility | > 0.05 | 0.0607 |
| one component | yes | yes |
| vertex connectivity | >= 2 | 3 |
| every node re-enters | yes | yes |
| causal closure | closed | closed, leak 0.0000 |
| largest component share | < 0.50 | 0.114 |
| effective dimension share | < 0.40 | 0.370 |
| synergy, the level | four of four | four of four |
| synergy, the change | four of four | four of four |

Seeds 11, 13, 17 and 3 read four of four on both synergy readings as well, with
effective dimension shares from 0.371 to 0.392.

## What this does not change

- No threshold, no criterion, and no line of any ISC version.
- No null architecture. The nineteen nulls are built exactly as before, and
  every one of them must still fail the conjunction.
- Nothing about Aura's recordings. The reference is a simulation; its row in
  the null table is computed from a toy run and contains nothing of hers.

## Which runs it applies to

Any run whose null table is computed at or after this commit, and the recomputed
architecture rows of a run whose organism measurements were recorded before it.
The architecture rows are simulations that read no recording of hers, so
recomputing them reads nothing of hers either; the surrogate rows, which are
built from her recording, are not recomputed and are carried forward as the run
recorded them.

A run's verdict under the amended reference is reported beside the verdict it
recorded, never in place of it.
