# ISC-v4, preregistered

Written on 21 September 2026 and committed with the code that computes it.
ISC-v1, ISC-v2 and ISC-v3 are kept. Every line of all three is computed and
reported on every run, and a failure under any of them stays a failure.

ISC-v4 is ISC-v3 with the irreducibility line's denominator changed. Nothing
else moves.

## Why a new version

The change raises the reading on a system whose future is largely
unpredictable, which is Aura's case, and its direction there was known before
this was written: the correction was measured on a 5,280-frame recording of her
and on that recording's shuffled rows. A criterion may only be made stricter
after its results have been seen. So this cannot amend v3, and as a new version
it reads only recordings from which no v4 irreducibility number has been read.

The code that computes it has existed since 16 September, guarded by a
parameter that defaults to off and that nothing outside its tests has ever
passed. Nothing in this change alters the estimator's own settings — the ridge,
the folds, the component count and the cut search are untouched — so P7.20 is
not in question. What changes is which quantity the ratio is taken over.

## What v3 asks, and what it cannot see

v3 asks that the lower bound of `phi_do` clear 0.05, where `phi_do` is

    (loss_cut - loss_full) / loss_cut

read over the held-out folds and cross-fitted over the 511 bipartitions.

Write the target's variance as S, the part a model of the whole previous state
can explain, plus N, the part nothing can. The intact model's error is
`E_full + N` and the cut model's is `E_cut + N`, both over `S + N`. Dividing by
`loss_cut` gives

    (E_cut - E_full) / (E_cut + N)

Every column nothing can predict adds to N and shrinks that ratio, without
changing the system at all. Four noise columns per domain took the recurrent
reference from 0.062 to 0.005. Aura is recorded on 297 columns, 74 of them flat
through a whole run, and she is being asked to clear a bar on a quantity that
falls as she is measured more widely.

## What v4 asks

The same lower bound over the same folds and the same cuts, on

    (loss_cut - loss_full) / (1 - loss_full)

which is `(E_cut - E_full) / (S - E_full)`: the cut's cost as a share of what
the intact model can explain at all. N cancels exactly.

A model that explains nothing has nothing to lose by being cut. Below an
explained share of 0.01 the answer is that there is no transition law for a cut
to damage, reported as an absent reading rather than as a ratio of two
vanishing numbers. That floor is the instrument's, not a bar on Aura: it is the
point below which the denominator is noise.

## It is not uniformly easier

v4 exceeds v3 exactly when `E_cut + N` exceeds `S - E_full` — when what nothing
can predict is large beside what the intact model explains. That is the
dilution regime and it is Aura's: her intact model leaves a held-out loss of
0.78, so she is being scored on a denominator that is mostly N.

It is not the reference's. On a 1,200-step recording of the recurrent
reference at seed 3, both versions pick the same cut with the same two losses —
`loss_full` 0.35105 and `loss_cut` 0.39014 — and read

| version | denominator | reading |
|---|---|---|
| v3 | 0.39014 | +0.09947 |
| v4 | 0.64895 | +0.05993 |

So v4 makes the positive control harder, not easier, and leaves it clearing the
0.05 bar by a third of what it had. A change that helped whatever it was
pointed at would not do that. What it does instead is stop the score falling as
a system is recorded more widely, which is the defect it was written for.

## What was measured before this was written

On the 5,280-frame recording in `/private/tmp/claude-501/soul3/kappa_probe`,
with the same estimator, the same folds and the same cuts:

| recording | v3's ratio | v4's ratio |
|---|---|---|
| as recorded | +0.00466 | +0.01571 |
| rows shuffled | +0.00034 | +0.00037 |

The correction more than triples the reading on the real recording and leaves
the shuffled null where it was. That is the shape a correction has to have: the
dilution it removes is not signal, so removing it cannot manufacture any.

Three standing tests hold its nulls, written when the code was
(`tests/test_an_unpredictable_column_diluted_the_score.py`): shuffled rows, a
random walk and pure noise all read at or below 0.01 under v4.

## What this version does not license

- It does not replace v3. Both are computed and both are reported, and the
  verdict a run publishes is the conjunction under every version it declares.
- It does not touch the estimator. Ridge penalties, fold boundaries, the
  component count per domain and the bipartition search are the same objects
  under v3 and v4, so a v3 number and a v4 number from one run differ only in
  the denominator.
- It does not apply to any recording from which a v4 irreducibility number has
  already been read. The numbers in the table above are the reason for the
  version and are not evidence under it.

## The nulls

Every null architecture is scored under v4 by the same code as Aura, as v3
requires of v3. A null that passes v4 fails the conjunction exactly as it would
under any other version, and `beats_every_null` reads the version the run
declares.
