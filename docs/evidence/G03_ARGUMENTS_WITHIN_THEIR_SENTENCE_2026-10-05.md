# No argument option starts before its operation's sentence

After [G03_AN_OPERATION_OWNS_ITS_SENTENCE_2026-10-05](G03_AN_OPERATION_OWNS_ITS_SENTENCE_2026-10-05.md),
v8 left one composition request and two validation rows.

## What v8 did with the composition request

`scalar_branch_weave_five-0-0`, decoded with v8's candidate:

> ... inputs are intake flow = 209641; return flow = 83651; turbine reserve =
> 6283; storage reserve = 549; correction = 79; trim = 15. Build the primary
> path by add intake flow and return flow. ... In parallel subtract storage
> reserve from turbine reserve.

| Step | Reference | v8 |
| --- | --- | --- |
| add | intake flow, return flow | intake flow, "turbine reserve = 6283" (token 30) |
| subtract | turbine reserve, storage reserve | "83651" (token 24), storage reserve |

Both wrong options are in the declaration list. Every input's literal is
offered to every operation, and the first operation's clause began at token
0, so the declarations competed with the names in each operation's own
sentence.

## The rule, checked before any run

With `--arguments-within-sentence` (commit `08df41d87`), an operation's
argument options start no earlier than its own sentence or the operation
before it, whichever is later. What comes after the clause is kept.

Checked on every annotated request before a run (`sentence_clause_check.py` in
the session scratchpad): train 764, validation 500, test 500 and both
composition bundles 96. No annotated argument starts before its operation's
sentence or the operation before it. 96 arguments, all
`arithmetic:nominal_nested`, come after their clause and stay offered. With
v8's candidate and only this switched on, 0-0 decodes to its reference.

## Measurement

From `tools/run_semantic_peak_recognition.py --argument-ownership
--argument-antecedent --antecedent-fit conditional --antecedent-stretches
sentence --arguments-within-sentence`, 20-second solve limit, the live runtime
shut down and nothing else computing; incumbent composition rows are v5's.
`~/.aura/rlc-evidence/semantic-peak-antecedent-v9-20261005`, from `08df41d87`,
candidate `5a0c03805ba3122b5dd98b166edf389161ab55ec29521ba731b10273ac01714c`,
development audit receipt
`03ae8d5ed0e05ddf6f6cf22ac50c704bc1da854d9bf22bbc51fbd755277c279e`,
`report.json` sha256
`cdcf0215dbccb5df9a4f69e3e685026c911f90a06a66a0ce88b823c916fb2958`.

| Cohort | Incumbent | v5 | v8 | v9 |
| --- | ---: | ---: | ---: | ---: |
| Composition v1, 48 | 31 | 47 | 47 | 48 |
| Composition v2, 48 | 31 | 47 | 48 | 48 |
| Validation, 500 | 477 | 498 | 498 | 498 |
| Train, 764 | 764 | 764 | 764 | 764 |
| Held-out folds, 764 | | 736 | 735 | 734 |

Every composition request of both bundles is answered: 34 gains over the
incumbent, no losses.

## The two held-out rows

Both are `arithmetic:nominal_nested` rows of fold 2, decoded by readouts
refitted without that construction group (`fold2_probe.py` in the session
scratchpad):

> Return the whole-number quotient of 84 divided by the whole-number quotient
> of 47 divided by 3. Use integer arithmetic.

The refitted recognizer places the second operation on "integer", in the
closing sentence, and misses the second "whole-number quotient". Without the
floor, that misplaced operation still reached 47 and 3 and the program
computed the right value: v5's credit for this row was a right answer from a
wrong parse. With the floor, an operation in "Use integer arithmetic." can
only take words of that sentence, and the misplacement shows. The other row
(`ac555d53eaef`) is wrong with or without the floor under v8's and v9's
readouts. The cause in both is the recognizer on a construction it has not
seen, which the floor does not touch.

## What is left

* Validation: the two `sequence-cataphoric-5` rows, failing at the operation
  ("removing" found without "after" and labelled add).

## What this does not show

* The rule was chosen with 0-0 in view. The check above shows it removes no
  annotated argument anywhere; the folds are the held-out measure.
