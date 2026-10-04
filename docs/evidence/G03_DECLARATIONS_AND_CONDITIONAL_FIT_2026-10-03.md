# A declaration holds no operation, and the readout fitted as it is used

Two more changes to the antecedent readout, after
[G03_IDENTICAL_SHARE_AND_SCORING_2026-10-03](G03_IDENTICAL_SHARE_AND_SCORING_2026-10-03.md),
each measured on its own run.

## What v2 left

Both training losses were `arithmetic:nominal_nested` rows with inline
literals: "Return the whole-number quotient of the whole-number quotient of 88
divided by 41 divided by 7." The first input's stretch ran from the start of
the sentence to "88", both operations included, so the "the" before the
second operation occurred exactly once earlier, inside that stretch, and the
identical-share feature sent it to the first input. The decoder then bound
the one-token mention "the" to that input.

On `scalar_branch_weave_five-0-1` the readout favoured the right parse by 4 to
5 nats, and other evidence favoured the wrong one by a margin in the same
range, so the outcome moved with the fit (v2 right, v3 and v4 wrong).

## v4: a declaration holds no operation

Commit `b9af3a257`. An input's stretch starts after the last input or
operation that ends before it. Requests that declare their inputs before any
operation, the composition bundles among them, keep their stretches; the fit
changes because the training rows with inline literals change.

## v5: fitted as each mention's distribution over registers

Commit `32e3d5292`, `--antecedent-fit conditional`. The readout is used as log
P(register | mention), normalised over a request's registers, but was fitted
as independent yes-or-no judgements on (mention, register) pairs. The
conditional fit maximises each annotated mention's log-probability of its
own register among the registers its operation could read (unit L2 penalty,
L-BFGS; 1,376 training mentions). It includes v4's stretches.

## Measurement

Both from `tools/run_semantic_peak_recognition.py --argument-ownership
--argument-antecedent`, 20-second solve limit, nothing else computing on the
machine. Their incumbent composition rows are v2's, copied in before each run
(same incumbent, same requests; `INCUMBENT_ROWS_REUSED.txt`).

v4: `~/.aura/rlc-evidence/semantic-peak-antecedent-v4-20261003`, candidate
`e3c98fccaffbb44d5896ec82385662e742bb51eba5c2ece94fb5f87219872eae`,
development audit receipt
`f1317d6eafa80554d0d8cb57f9f6ca199028e7057b7d04186b4528a74b175c78`,
`report.json` sha256
`dbd4fb477ef42f34286c3e17931f7cf8721b3e6eb66ec4f408261d1149820c35`.

v5: `~/.aura/rlc-evidence/semantic-peak-antecedent-v5-20261003`, candidate
`cffd98c16010ee6823a1d67f8ce2aa91883984cc94145fea25b95158e3e5d4f8`,
development audit receipt
`5b26c1cab2962a546d951f8361d7336f973c2498a821256c8433bf9e7fa60c58`,
`report.json` sha256
`2ce5d0852182c0e15e4bc8840a0eab8d6137ba0250435e30a1861979a8fb0c74`.

| Cohort | Incumbent | v2 | v4 | v5 |
| --- | ---: | ---: | ---: | ---: |
| Composition v1, 48 | 31 | 47 | 46 | 47 |
| Composition v2, 48 | 31 | 48 | 47 | 47 |
| Validation, 500 | 477 | 498 | 498 | 498 |
| Train, 764 | 764 | 762 | 764 | 764 |
| Held-out folds, 764 | | 736 | 736 | 736 |

v5 gains 16 composition requests over the incumbent in each bundle and loses
none, and gets every training row. It is the candidate with the fewest
failures across the development cohorts and both bundles: four, against five
for v2 and v4.

## What is left with v5

* `scalar_branch_weave_five-0-0` (first bundle): two input declarations
  ("turbine reserve = 6283", "83651") taken as arguments over the names that
  use them. The readout gives a literal no evidence, so it cannot prefer the
  name, and the role heads prefer the declarations.
* `scalar_branch_weave_five-0-1` (second bundle): the first subtraction takes
  "that output", the clause naming its own result, as its minuend; the
  margin is small and moves between fits.
* Validation: two `sequence-cataphoric-5` rows, which fail at the operation
  ("removing" is found without "after" and labelled add).

## What this does not show

* All four changes were chosen with these cohorts in view; the folds, which
  refit every readout without each construction group, are the held-out
  measure, and they did not move between v2 and v5.
* G03 stays open on the four requests above.
