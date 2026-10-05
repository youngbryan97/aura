# Naming an operation from its words as well as its context does not hold

After [G03_ARGUMENTS_WITHIN_THEIR_SENTENCE_2026-10-05](G03_ARGUMENTS_WITHIN_THEIR_SENTENCE_2026-10-05.md),
v9 answered every composition request and left two validation rows.

## The two rows

Both are `sequence-cataphoric-5`, a wording held out of training:

> Resolve 7 after removing the subsequently computed number only after
> computing the subsequently computed number with reading one indexed entry
> from [19, 18, 6, 20, 2, 13] at selector 2.

The reference subtraction is "after removing". v9 found "removing" and named
it add. The labeler is a linear readout of the middle layer at a span's last
token, so "after removing" and "removing" are named at the same place. Across
the four `sequence-cataphoric-5` rows with that wording it reads (from
`removing_labels.py` in the session scratchpad):

| Row | First | Second |
| --- | --- | --- |
| two that pass | sub 0.348 | add 0.336 |
| two that fail | add 0.347 | sub 0.346 |

The context barely separates them, for a word training never had.

## Which representation names an unseen wording

Offline, each readout fitted on the training operation spans and scored on
the 1,192 validation operations with their annotated spans
(`labeler_variants.py`):

| Readout | Validation | "after removing" |
| --- | ---: | ---: |
| middle layer at the last token (v9) | 1184 | 10 of 12 |
| final layer at the last token | 1166 | 8 of 12 |
| middle and final at the last token | 1188 | 8 of 12 |
| middle layer, mean over the span | 1171 | 8 of 12 |
| input embeddings of the span's words, mean | 1036 | 12 of 12 |
| the middle layer and the words, stacked | 1184 | 12 of 12 |

The words out of context know "removing" is a subtraction and know little
else; the context knows the rest.

## v10: stacked, weighted on held-out constructions

Commit `d84b77512`, `--stacked-labeler`. Both readouts name the span and
their log-probabilities are added with weights fitted on the frozen
construction folds: each readout is refitted without each group and scores
it, and the weights maximise the likelihood of those out-of-fold labels
(16.18 for the context, 7.92 for the words). Fitted on the rows they saw, the
context readout is near perfect and would take all the weight; how each does
on a construction it has not seen is the question.

## Measurement

From the v9 settings with `--stacked-labeler`, 20-second solve limit, the
live runtime shut down. `~/.aura/rlc-evidence/semantic-peak-antecedent-v10-20261005`,
from `d84b77512`. Stopped by hand after the first fold; its log says so.

| Cohort | v9 | v10 |
| --- | ---: | ---: |
| Validation, 500 | 498 | 492 |
| Train, 764 | 764 | 763 |
| Held-out fold 0, 256 | 236 | 188 |

v10 gains one of the two `sequence-cataphoric-5` rows and loses seven other
held-out cataphoric rows and one training row. Offline the stack was scored
on annotated spans; the decoder names every candidate span the tagger grows,
of every length, and over those the mean of the words' embeddings misleads.
The option stays off; v9 stays the candidate.

## Where this leaves G03

With v9, every binding and composition failure on the development cohorts is
gone: train 764 of 764, both composition bundles 48 of 48, validation 498 of
500. The two validation rows left fail at naming an operation whose wording
training never had. That is vocabulary transfer, which G04 measures, and the
two rows go there.
