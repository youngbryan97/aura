# A named result read back to where its name was given

## What failed, and why

After peak recognition and argument ownership
([G03_PEAK_RECOGNITION_2026-10-02](G03_PEAK_RECOGNITION_2026-10-02.md)), 19
of the 96 five-step composition requests were still answered wrongly: 10 of
48 in `semantic-natural-weave-replication-v1` and 9 of 48 in `-v2`. The two
bundles share their construction identities, and the same constructions fail
in both.

All ten in the first bundle were decoded one at a time and compared with the
reference program. Every one binds a named intermediate or a named input to
the wrong register:

* six bind the branch result ("auxiliary result", "side measure") to the
  input that the branch's own subtraction should have read, so the
  subtraction is left to consume a later step and the program comes out
  reordered with the subtraction last;
* two exchange two inputs, one of them read from the clause that names the
  subtraction's own result;
* one binds the lead result to an input and moves the subtraction last;
* one exchanges two named intermediates between steps.

A step register is defined at its operation span ("subtract"). A causal
model's state at that token cannot hold a name that is given only in the
next sentence ("Save this output as the auxiliary result"), so the relation
head, which compares a mention with a register's definition, has nothing to
tell the auxiliary result from the primary one.

## The mechanism

`core/learning/semantic_argument_antecedent.py`.

* Each register owns a stretch of the request: an input its declaration, an
  operation its clause and the clause naming its result, up to the next
  operation.
* A mention is compared token by token with every earlier window of each
  stretch, in the input embedding and the middle layer.
* A logistic readout over how well the best window matches, whether that
  stretch is the first to match so well, and how far it trails the best,
  fitted on training rows only, gives P(mention names register). Its log is
  added to each argument option's score beside ownership.
* A mention that is an input's literal value is bound exactly by the literal
  grammar. The readout is neither fitted on nor applied to one.

## The readout alone, before decoding

Rank of the annotated register and its margin over the best other, by
cohort (`antecedent_variants.py`, fitted on the 764 training sources):

| Fitted on | Cohort | Named intermediates first | Median margin (nats) |
| --- | --- | ---: | ---: |
| every mention | weave v1 | 191 / 192 | +0.55 |
| every mention | weave v2 | 192 / 192 | +0.55 |
| non-literal mentions | weave v1 | 192 / 192 | +1.17 |
| non-literal mentions | weave v2 | 192 / 192 | +1.18 |
| non-literal mentions | train | 813 / 956 | +2.44 |
| non-literal mentions | validation | 561 / 692 | +2.26 |

Standardised features and weaker regularisation did not help (146-188 of 192
first in the weave bundles).

## Measurement

Run root `~/.aura/rlc-evidence/semantic-peak-antecedent-20261003`, from
`65ca44a9e` by `tools/run_semantic_peak_recognition.py --argument-ownership
--argument-antecedent`, 20-second solve limit, nothing else running on the
machine. Candidate
`814239fb49c11ee20da1ce8f10d59f351ec00878f6c672e9a69d09365a7d2c69`; recognizer
`c7fbb5f1…`, unchanged; antecedent fit receipt `0acbdf53…` (764 training
sources, 6,272 mention-register pairs, 1,376 named); development audit receipt
`685f4116fed0f3e90115fe4d836cffb9ec4a78e73fed613ae39d716dbc8321e3`;
`report.json` sha256
`ef6d3967dd50a0a013d34a01636c7cdf000ea4d6eda5b6032f5b85aa9e2dfa6a`.

Composition, answers executed against the reference program:

| Bundle | Incumbent (this run, quiet) | Peaks + ownership (2 Oct) | + antecedents | Gains over incumbent | Losses |
| --- | ---: | ---: | ---: | ---: | ---: |
| weave v1, 48 | 31 | 38 | 45 | 14 | 0 |
| weave v2, 48 | 31 | 39 | 47 | 16 | 0 |

The incumbent made one refusal in this run (v1); its other 33 misses are
wrong programs. On 2 October its arm ran beside a test suite and lost nine
requests per bundle to wall-clock search budgets; this run had the machine to
itself.

Development cohort:

| Cohort | Incumbent | Peaks + ownership | + antecedents |
| --- | ---: | ---: | ---: |
| Validation, 500 | 477 | 497 | 497 |
| Train, 764 | 764 | 764 | 763 |

Held-out construction folds (each refits the recognizer, ownership and
antecedent readouts without the fold): 236 / 256, 252 / 254, 246 / 254, which
is 734 of 764, against 733 with ownership alone.

## What is left

Four composition requests, decoded one by one with this candidate:

* two arguments are taken from the input declarations ("turbine reserve =
  6283", "83651") rather than from their later uses by name. The readout is
  silent on literal spans, and nothing else says a declaration is not a use
  (v1 `scalar_branch_weave_five-0-0`);
* the first subtraction takes "that output", the clause naming its own
  result, as its minuend; the program is reordered to break the cycle (v1 and
  v2 `scalar_branch_weave_five-0-1`);
* "subtract staffing reserve from access reserve" has its operands in the
  wrong order, a role decision and not a grounding one (v1
  `scalar_branch_weave_five-7-0`).

The one training loss (`arithmetic:nominal_nested`, `fb1ff82bbdb2`) is this
readout's: the one-token mention "the" matches an earlier "the" exactly, and
the decoder bound it to the first input. An exact match on a word the request
uses everywhere is no evidence of a name, and the readout has no feature that
says how common the matched words are. The three validation misses are the
same `sequence-cataphoric-5` rows as before and fail at the operation
("removing" found without "after", labelled add in two of them).

## What this does not show

* The readout's design (stretches, the literal exclusion) was chosen with the
  composition bundles in view, which makes them development evidence. The
  folds refit it on training rows without each construction group.
* The composition bundles were extracted by a later worker revision; both
  arms read the same features, so the comparison is paired and the absolute
  level is cross-revision, as on 2 October.
* G03 stays open on four composition requests, one training row and three
  validation rows.

## Checks

`tests/test_semantic_argument_antecedent.py`: six tests, including a name
never seen in training read back to the step that gave it, literal values
left to the literal grammar, and a decode whose program moves when the
readout's weights are reversed. Smoke, lint, layering and the writing gate
pass.
