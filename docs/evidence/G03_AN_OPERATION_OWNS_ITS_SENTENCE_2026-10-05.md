# An operation's register owns its sentence

After [G03_TWO_RULES_THAT_FAIL_2026-10-03](G03_TWO_RULES_THAT_FAIL_2026-10-03.md),
v5 was the candidate with two composition misses and two validation rows
open. One of the composition misses comes from where register stretches were
cut, and this measures the change that recuts them.

## The gap

`scalar_branch_weave_five-0-1`:

> Form the lead calculation by subtract return flow from intake flow. Save
> that output as the primary result. ... Refine the lead calculation and
> whole-number divide primary result by correction.

The antecedent readout reads a mention back to the register whose stretch of
the request gave its name. An operation's stretch ran from its own word to
the next operation's word, and an input's ended at its literal, so the
tokens from the last input to "subtract", "Form the lead calculation by",
belonged to no register. "the lead calculation", used three sentences later,
could not be read back to the subtraction.

## v7: an operation owns its sentence

Commit `f9831a794`, `--antecedent-stretches sentence`. The readout binds the
tokenizer's sentence-ending tokens (163 for the persona tokenizer). An
operation owns its whole sentence, from after the last input declared in it,
up to the next operation's stretch; sentences with no operation stay with the
operation before. A full stop inside a literal ends no sentence.

## v8: only an operation alone in its sentence

Commit `ce67c2be1`. v7 lost three training rows, all
`arithmetic:nominal_nested` ("the whole-number quotient of the whole-number
quotient of 88 divided by 41 divided by 7"): with several operations in one
sentence, the first took "Return the" and a later "the" was read back to it.
Operations sharing a sentence each start at their own word, as in v4.

## Measurement

From `tools/run_semantic_peak_recognition.py --argument-ownership
--argument-antecedent --antecedent-fit conditional`, 20-second solve limit,
the live runtime shut down and nothing else computing. Incumbent composition
rows are v5's (`INCUMBENT_ROWS_REUSED.txt`).

v7: `~/.aura/rlc-evidence/semantic-peak-antecedent-v7-20261005`, from `f9831a794`,
candidate `70011b4def06e9a1e60ebf29d6c2315e0a922a09c8092421d84eba67da41b67f`,
development audit receipt
`983256200f50fe0f5785671bf378df6a8caf1cc58960a4f38d927f5fc3da8a15`,
`report.json` sha256
`2c6a83841f6014e10641dce153bee95799b10c70ea87c364ee65ac5ec8a1dd54`.
v8: `~/.aura/rlc-evidence/semantic-peak-antecedent-v8-20261005`, from `ce67c2be1`,
candidate `a46f950707f858ab9ff24d90c1a372dcd0fe585c995275138b73c4a3ce7e3f4a`,
development audit receipt
`2c7d1891f686e4a0b430734b8c09ee656401063915f093f925b83aad2a2b28d1`,
`report.json` sha256
`73edbbe9ac660dd95b093fa214424a297bb47019879008d47a40481fd7621b6d`.

| Cohort | Incumbent | v5 | v7 | v8 |
| --- | ---: | ---: | ---: | ---: |
| Composition v1, 48 | 31 | 47 | 47 | 47 |
| Composition v2, 48 | 31 | 47 | 48 | 48 |
| Validation, 500 | 477 | 498 | 498 | 498 |
| Train, 764 | 764 | 764 | 761 | 764 |
| Held-out folds, 764 | | 736 | 735 | 735 |

v8 keeps v7's gain and gives back v7's training losses. Against v5 it gains
`scalar_branch_weave_five-0-1` and loses one held-out row: `ac555d53eaef`, an
`arithmetic:nominal_nested` row in fold 2, one of the three v7 lost in
training. On the development cohorts and both bundles v8 has three failures
against v5's four; on the held-out folds it is one row behind.

## What is left with v8

* `scalar_branch_weave_five-0-0` (first bundle): the two input declarations
  taken as arguments over the names that use them, as with v5.
* Validation: the two `sequence-cataphoric-5` rows, failing at the operation.

## What this does not show

* The change was chosen with `scalar_branch_weave_five-0-1` in view. The
  folds, which refit every readout without each construction group, are the
  held-out measure.
* G03 stays open on what v8 leaves (above).
