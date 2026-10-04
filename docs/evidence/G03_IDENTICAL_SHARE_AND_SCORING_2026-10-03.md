# How common a matched word is, and how the antecedent enters the score

Two changes to the antecedent readout of
[G03_NAME_ANTECEDENTS_2026-10-03](G03_NAME_ANTECEDENTS_2026-10-03.md), each
measured on its own run.

## What the first run left

Four composition requests, one training row and three validation rows. The
training loss (`arithmetic:nominal_nested`, `fb1ff82bbdb2`) was the readout's:
the one-token mention "the" matched an earlier "the" exactly, and the decoder
bound it to the first input. One composition miss
(`scalar_branch_weave_five-0-0`) took two input declarations ("turbine reserve
= 6283", "83651") as arguments over the names that used them.

## v2: how common the matched words are

`identical_share` (commit `4038af95a`): of the earlier places where the
mention's exact tokens occur, the share inside each register's stretch.
"auxiliary result" occurs once before its use, in the clause that named it;
"the" occurs in nearly every stretch.

## v3: which register, not which span

`--antecedent-scoring relative` (commit `9f0a92cf8`). Adding log P(register |
mention) to an argument option compares different spans for the same slot
unfairly: a span whose distribution is peaked, or a literal scored zero, pays
less for its best register than a name mention does. Relative scoring adds
log P less the span's best register's, so every span's best register scores
zero and only a worse one pays.

## Measurement

Both runs from `tools/run_semantic_peak_recognition.py --argument-ownership
--argument-antecedent`, 20-second solve limit.

v2: `~/.aura/rlc-evidence/semantic-peak-antecedent-v2-20261003`, from
`4038af95a`; candidate `9767eb56e7fb7ed6c9659184465d78e4b0e2651f2df4809a97f995ad3b99f0e9`.
The run was cut off at the end of a session after its development and fold
cohorts; its composition arms were finished from the saved candidate by the
runner's own `score_composition` (`composition_finished.json`, sha256
`127e5b1c9df4d94db64541cb8ff337e6f209d7211cbf52a5f564261bf4cb11b4`). Nothing
else ran on the machine.

v3: `~/.aura/rlc-evidence/semantic-peak-antecedent-v3-20261003`, from
`9f0a92cf8`. Its incumbent composition rows are v2's, copied in before the
run (same incumbent, same requests, measured that evening with the machine
alone; `INCUMBENT_ROWS_REUSED.txt`).

v3 candidate `f4a4be2c9f2731578ae016d3a0dc88c64521f1510202149ca76fdc4534d21a00`;
development audit receipt
`06434da686c7e1deddee717287402778f31b2094d4c89f571d41dc5126722328`; `report.json`
sha256 `985d628ca79eb18d0c866f0322d471dd730f32f76f2446203653e338830cb055`.

| Cohort | Incumbent | First run | v2 | v3 |
| --- | ---: | ---: | ---: | ---: |
| Composition v1, 48 | 31 | 45 | 47 | 46 |
| Composition v2, 48 | 31 | 47 | 48 | 47 |
| Validation, 500 | 477 | 497 | 498 | 498 |
| Train, 764 | 764 | 763 | 762 | 763 |
| Held-out folds, 764 | | 734 | 736 | 736 |

Against the incumbent, v2 gains 16 and 17 composition requests and loses none.
Folds for v2: 236 / 256, 252 / 254, 248 / 254.

v2 answers both `scalar_branch_weave_five-0-1` requests (the subtraction no
longer takes "that output", the clause naming its own result, as its minuend)
and `scalar_branch_weave_five-7-0` (the operands of "subtract staffing reserve
from access reserve" in order). Its one composition miss is
`scalar_branch_weave_five-0-0`, the two declarations, which v3 does not fix
either. v3 recovers one training row and loses both `-0-1` requests again.
v2 is the better candidate on the cohort G03 is about; relative scoring trades
two composition requests for one training row and is kept as an option, not
the default.

## What is left with v2

* Composition: `scalar_branch_weave_five-0-0` in the first bundle.
* Train: two `arithmetic:nominal_nested` rows (`fb1ff82bbdb2`,
  `8a877e193d4c`); the incumbent answers both.
* Validation: two `sequence-cataphoric-5` rows (`1ed4ec316565`,
  `e150f02f52dc`), which fail at the operation.

## What this does not show

* Both changes were chosen with the composition bundles and the development
  failures in view. The folds refit every readout without each construction
  group; they are the held-out measure here.
* The composition bundles were extracted by a later worker revision than the
  training sources; both arms read the same features.
