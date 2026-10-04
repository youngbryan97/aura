# Two rules for the last composition misses, and why neither is kept

After [G03_DECLARATIONS_AND_CONDITIONAL_FIT_2026-10-03](G03_DECLARATIONS_AND_CONDITIONAL_FIT_2026-10-03.md)
two composition requests were still wrong with v5. Each suggested a rule
read off the antecedent readout. Both are implemented as options that are
off by default, and both fail on the development cohorts.

## A mention that names an operation's own result is not one of its inputs

`scalar_branch_weave_five-0-1`: the first subtraction takes "that output", in
"Save that output as the primary result", as its minuend. The readout's most
probable referent for that mention is the subtraction's own result. Commit
`72f802783` (`--own-result-is-not-an-input`) withholds an argument option
whose span's most probable referent is the operation's own register.

Run `~/.aura/rlc-evidence/semantic-peak-antecedent-v6-20261003`, from
`72f802783`, with the v5 settings (conditional fit). Development: train 632 of
764 (132 lost against v5's 764), validation 476 of 500 (22 lost). The readout's
most probable referent is often the operation's own register for its genuine
inputs too, so the rule removes correct options wholesale. The run was
stopped after the development cohort; no fold or composition result is
claimed.

## An input the request names is used through the name

`scalar_branch_weave_five-0-0`: two input declarations ("turbine reserve =
6283", "83651") are taken as arguments over "turbine reserve" and "return
flow", the names that use them. Commit `d19c6bc6c`
(`--named-inputs-are-used-by-name`) withholds an input's literal when some
non-literal mention most probably names that input.

Checked before a run, with v5's readout on the annotated mentions
(`used_by_name_check.py` in the session scratchpad): the rule would withhold a
gold literal argument in 16 of 500 validation rows and 16 of 500 test rows,
all `sequence-role-binding-5` and `-8`, where an input is both named and used
by its value; none of 764 training rows and none of the 96 composition
requests. A rule that removes gold arguments from 16 development rows to fix
one composition request is not kept, and no run was made.

## Where G03 stands

v5 stays the candidate: composition 47 and 47 of 48, validation 498 of 500,
train 764 of 764, folds 736 of 764. G03 stays open on two composition requests
and two validation rows.
