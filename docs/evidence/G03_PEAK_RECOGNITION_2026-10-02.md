# Operations found where the request reads as one

## What failed, and why

The incumbent compositional transducer
(`semantic-literal-identity-pilot-20260921`, receipt `750a7cf3…`) has 477 of
500 development validation sources equivalent under the cohort audit
(`source_anchors_v2`, `compare_program_meanings`). All 23 failures are in the
144 sequence-chain tasks: 19 have the wrong operations and 4 the wrong
binding. Validation is written with three wordings per operation that
training never uses ("reading one indexed entry", "with a factor of", "after
removing").

With the annotated span supplied, a linear readout of the middle layer at the
span's last token names 99.3% of validation operations correctly. The label
was in the representation. The span was not found: the incumbent's pointer
was trained on three wordings and scores a new wording low in absolute terms.

## The mechanism

`core/learning/semantic_operation_peaks.py`, commit `78e9146f7`.

* A token tagger over the normalised middle and final layers scores how much
  each token reads as part of an operation phrase. Operations are the local
  maxima of that score, not tokens over a threshold: a new wording still
  stands above the words around it.
* Each maximum grows into a span while its neighbours keep half of its score,
  and the span is named by a readout of the middle layer at its last token.
* The first token is read from the input embedding. A causal model's first
  position carries the attention sink, so its contextual layers do not
  describe the word there.
* Charts are tried in that evidence order and the first whose arguments
  assign is kept. Under the incumbent's joint score, argument evidence
  overruled better operation readings and cost 22 tasks.

Every readout is fitted on the 764 training sources (2,971 operation tokens);
the fitter refuses any other split. The incumbent's grounding and argument
heads are used unchanged. `decode(operation_recognizer=...)` is the only hook,
and it refuses to run beside the other proposal hooks.

## Measurement

Run root `~/.aura/rlc-evidence/semantic-peak-recognition-20261002`, from
`78e9146f7` by `tools/run_semantic_peak_recognition.py`, 20-second solve limit,
diagnosis off. Recognizer identity
`c7fbb5f1c3ab400a83ce71c3573eaf984ee03e52b7b1e65779e3f6725741f6ec`; candidate
`aa25819ac4f82005aa3f20deb2a217fe55c76b8bf1ef503517af212e07a33093`; development
audit receipt `dddd09539650b8bd04695a6451b5974bb4924e0916982f2a32389a52e1ba96c8`;
`report.json` sha256
`d427ae8c175d8bb22f1df112f3c65d9f19fbeda5bed3789afe6d183117d0a1a3`.

| Cohort | Incumbent | Peak recognition | Gains | Losses |
| --- | ---: | ---: | ---: | ---: |
| Validation, 500 | 477 | 496 | 20 | 1 |
| Train, 764 | 764 | 764 | 0 | 0 |

Validation by family:

| Family | Sources | Incumbent | Peak recognition |
| --- | ---: | ---: | ---: |
| arithmetic | 128 | 128 | 128 |
| cataphoric | 48 | 40 | 44 |
| fork_join | 192 | 192 | 192 |
| natural_alias_source | 12 | 12 | 12 |
| natural_source | 24 | 24 | 24 |
| reserved_alias | 48 | 36 | 48 |
| role_binding | 48 | 45 | 48 |

The one loss and all four remaining failures are cataphoric sequence chains
("after removing").

Held-out constructions. The three frozen source folds
(`semantic-architecture-source-folds-20260921`, 42 construction groups) each
refit the tagger and both readouts without that fold, and the fold is audited
with the incumbent's argument heads:

| Fold | Held out | Equivalent |
| --- | ---: | ---: |
| 0 | 256 | 236 |
| 1 | 254 | 252 |
| 2 | 254 | 244 |

That is 732 of 764 on constructions the recognizer never saw. The incumbent's
argument heads did see these rows, so the folds test recognition, not the
whole transducer.

Exposed composition cohort (`semantic-natural-weave-replication-v1`, 48
five-step requests), answers executed against the reference program:

| Arm | Answers / 48 | Refusals |
| --- | ---: | ---: |
| Incumbent, this run | 24 | 9 |
| Incumbent, quiet machine, earlier the same day | 31 | 0 |
| Peak recognition, this run | 32 | 0 |

All nine incumbent refusals in this run are search budgets
(`argument_optimizer_budget_exhausted`, `argument_chart_construction_budget_exhausted`,
`argument_optimizer_status:1`) while the test suite ran beside it; the
incumbent's joint search is wall-clock bounded. The quiet-machine 31 is the
fair comparison.

The second bundle (`semantic-natural-weave-replication-v2`, 48 requests):
peak recognition 30, the incumbent 24 with nine budget refusals in this run
and 31 on a quiet machine. Across both bundles the two arms are level, 62 of
96 against 62.

## Ownership: an argument belongs to the operation it stands beside

`core/learning/semantic_argument_ownership.py`, commit `8a6b5fd6f` on main.
On the composition cohort every operation was found and named, and all 16
misses were bindings. Each operation is offered every mention between its
neighbouring operations, so "multiply refined result by auxiliary result" was
also offered the previous clause's "primary result", and the role head, which
compares two pooled span vectors and never sees where a mention stands, took
it. Half the misses bound a named intermediate to the wrong step, half a named
input to the wrong register.

A logistic readout over where a mention stands relative to each operation
(which side, log distance, operations between, nearest on either side),
fitted on the 7,968 (mention, operation) pairs of the 764 training sources,
gives P(operation owns mention) over the request's operations; its log is one
more term in each argument option's score. Fitted weights: distance
-6.32 per log-token, softened by +3.12 when the mention follows the
operation; +1.74 when no operation lies between; +1.42 when this is the
nearest operation before the mention.

Run root `~/.aura/rlc-evidence/semantic-peak-ownership-20261002`, same tool
with `--argument-ownership`, candidate
`b0869eb36ad026203d61ece9bca34649e32782cf4bdfda571cfb97e60fd46e09`, development
audit receipt `fe3ea5efec83509d0fa3cd433627033accfa1d3ef10bec440d10d387a62fc234`,
`report.json` sha256
`73414dfa73108cbc745aa3ea26b6df820d2efb19f0a549e7e1d2e2d293227340`. The
readout is refitted without each construction fold for the fold audits.

| Cohort | Incumbent | Peaks | Peaks + ownership |
| --- | ---: | ---: | ---: |
| Validation, 500 | 477 | 496 | 497 |
| Train, 764 | 764 | 764 | 764 |
| Held-out construction folds, 764 | | 732 | 733 |
| Composition v1, answers / 48 | 31 | 32 | 38 |
| Composition v2, answers / 48 | 31 | 30 | 39 |

The incumbent's composition answers are the quiet-machine measurements. In
this run its arm refused 47 and 48 of 48 for search budgets while her live
runtime held the machine; the peak arms decide charts in evidence order and
refused none.

## What this does not show

* Lexicographic chart selection, the last-token labeler and the tagger's
  channels were chosen with validation in view. The first-position fix was
  found on the folds. Validation is therefore development evidence, not a
  held-out estimate.
* The composition bundles were extracted by a later worker revision. Their
  representation basis differs from the training sources' in two fields:
  `worker_source_sha256`, and the parameter count's counting basis (26.9B
  logical against 4.2B stored quantized elements). Both arms read the same
  features, so the comparison is paired; the absolute level is cross-revision.
* G03 stays open: with ownership, 19 of the 96 composition requests still
  fail, and they have not yet been diagnosed. Before ownership, on the first
  bundle, every operation was found and named and all 16 misses were binding.
* The ownership readout sees position only. It was chosen after the first
  bundle's failures were read, so the composition gain is development
  evidence on an exposed cohort, not transfer.

## Checks

`tests/test_semantic_argument_ownership.py`: 5 behavioural tests, including
training-only fitting, a mention after an operation preferred over the one
before it, and a decode whose program moves when the readout is reversed.
`tests/test_semantic_operation_peaks.py`: 10 behavioural tests, including a
first-position case, peaks below one half, refusal of non-training rows, and a
decode whose operations follow the recognizer's names. Smoke 164 passed, one
skipped; lint, layering and the writing gate pass. The god-object budget is
the measured 124,705.
