# The word says which operation; the sentence says whether

After [G03_A_STACKED_LABELER_DOES_NOT_HOLD_2026-10-05](G03_A_STACKED_LABELER_DOES_NOT_HOLD_2026-10-05.md),
v9 left two validation rows, both `sequence-cataphoric-5`:

> Resolve 7 after removing the subsequently computed number only after
> computing the subsequently computed number with reading one indexed entry
> from [19, 18, 6, 20, 2, 13] at selector 2.

Training never had "removing". v9's labeler, the middle layer at a span's last
token, named it add by 0.347 to 0.346.

## The problem, before the build

Bryan asked for the problem to be worked out conceptually first: how people
do it, how animals do it, what would solve it, and what research says.

A reader who has never seen "after removing" in an arithmetic request reads a
subtraction from the verb. "Remove" means take a thing away, and the meaning
comes with the word whatever sentence it is in. The sentence says what the
verb acts on: "7 after removing X" keeps 7 and takes X from it. That is two
jobs, one lexical and one structural, and v9's labeler does both from one
vector. That vector carries the word and also the construction around it,
which training saw only with other verbs. For an unfamiliar verb the
construction pulls as hard as the word.

Linguistics calls the word that decides a phrase's category its head. In
"after removing the subsequently computed number" the operation is named by
"removing"; "after" orders the steps and the rest is the argument. v10 read
the words as the mean input embedding of the span. On annotated spans that
read "after removing" right 12 times out of 12, but the decoder names every
span the tagger grows, and a long span's mean is mostly "after", "the" and
"only". The tagger already finds the head: its peak is the token it scores as
most operation-like, and spans grow outward from it.

Animals sort new things into categories they have by what a thing is like.
Pigeons trained on photos of trees pick out trees they have never seen
(Herrnstein, Loveland and Cable 1976). Shepard (1987) found that across
species a learned response carries over to a new stimulus with a probability
that falls off exponentially with its distance from the trained ones, in the
space the animal perceives. The space is the point: among constructions,
"removing" is far from anything trained; among words it sits beside
"subtract" and "minus".

Lakoff and Núñez (2000) ground arithmetic in object collection: adding puts
collections together, subtracting takes one away. English keeps it in the
words. "Subtract" is Latin *subtrahere*, to draw away from under; "remove" is
*removere*, to move back or away. Taking away is older than words for it:
monkeys, birds and fish choose the larger of two sets (Brannon and Terrace
1998; Agrillo et al. 2008), and five-month-old infants look longer when a doll
taken from a set is still there (Wynn 1992). The verb in the failing row is
the act subtraction was named after, so a word space should put it near
subtraction.

The research that applies: distributional semantics (Harris 1954; Firth 1957)
explains the failure, a representation learned partly from a word's company
meeting company it never had; prototype categorisation (Rosch 1975; Snell,
Swersky and Zemel 2017) and zero-shot naming through a word space (Socher et
al. 2013; Frome et al. 2013) say to name the unfamiliar by where its word
sits; stacked generalisation (Wolpert 1992) on held-out construction groups
says how much to trust each reading; and worst-group measurement (Sagawa et
al. 2020) is why the folds hold out whole constructions.

## The change

Commit `7507b7d05`.

* `--lexical-at word`: the words are read as the mean input embedding of the
  whole word holding the tagger's peak ("multip" and "licity" are one word).
  The tokenizer's word-continuing tokens (117,735 of 248,077) are bound at fit
  time, and the decoder passes the request's token ids. `--lexical-at peak`
  reads the peak token alone.
* Training reads each operation's words at the fitted tagger's peak inside
  its annotated span, where the decoder will read them.
* The stacked weights are fitted out of fold on the construction groups, as
  in v10, and kept at the contextual readout's scale: (1, words/context). The
  fit's overall size is a temperature. The chart adds a span's log-confidence
  to scores whose scale was set by the contextual readout alone.

## Before the full run

Offline, every readout fitted on training operation spans, from
`head_word_probe.py` in the session scratchpad:

| Readout | Validation spans, 1,192 | "after removing", 12 | Decoder-grown spans, 2,220 |
| --- | ---: | ---: | ---: |
| context only (v9) | 1184 | 10 | 2065 |
| context and the peak token | 1182 | 12 | 2097 |
| context and the whole word | 1186 | 12 | 2107 |

Decoded with v9's base, ownership and antecedent (`v11_targeted.py`):

| Rows | v9 | weights as fitted (16.3, 7.7) | at the contextual scale (1, 0.48) |
| --- | ---: | ---: | ---: |
| validation, cataphoric constructions, 48 | 46 | 42 | 48 |
| held-out fold 0, 256 (v10: 188) | 236 | | 236 |

At the fitted scale the labels sharpened sixteenfold and five cataphoric rows
went wrong; the ratio alone lost none. That choice was made after seeing the
48 validation rows, so the folds below are the held-out measure of it.

## v11: the words at their peak

From `5b3c54aa2`, `--stacked-labeler --lexical-at word`, the live runtime shut
down and nothing else computing:
`~/.aura/rlc-evidence/semantic-peak-antecedent-v11-20261006`.

| Cohort | v9 | v11 |
| --- | ---: | ---: |
| Train, 764 | 764 | 764 |
| Validation, 500 | 498 | 500 |
| Composition v1, 48 | 48 | 48 |
| Composition v2, 48 | 48 | 48 |
| Held-out folds, 764 | 734 | 732 |

Every development cohort was whole, and the held-out folds were two rows
behind. Fold 0 gained four cataphoric rows and lost four sequential ones,
fold 1 gained two reserved-alias rows, fold 2 gained two "3 reduced by"
cataphoric rows and lost six nominal ones.

Each nominal row ends "Use integer arithmetic." (`diag_fold_losses.py`). In
training, "integer" is inside an operation only as "integer-divide", so the
words read it as integer division: 0.40 against the context's 0.26 in "the
product of 25 and the sum of 72 and 2". The chart took the modifier for an
operation and dropped "sum".

That is the first question again. A reader decides from the sentence whether
a word is acting as an operation: "integer arithmetic" is an adjective on a
noun. Then the word says which operation. Psycholinguistics finds the same
split, with a word's own meaning and its context combined as constraints and
neither enough alone (MacDonald, Pearlmutter and Seidenberg 1994). v11 let the
word answer both questions, and its weights could not have learned
otherwise: they were fitted on operation spans only, where the question of
whether never comes up.

## v12: the words choose the name

Commit `01701a935`, `--words-name-only`. Each span keeps its best name at the
contextual readout's confidence, and the names are ranked by the pooled
reading. The words can change which operation a span names and never how
sure the span is of naming one.

On fold 2's eight changed rows the six nominal rows come back and both
cataphoric gains stay. Held-out fold 0: 240 of 256 (v9 236, v11 236). The 48
cataphoric validation rows: 48.

## Measurement

From `01701a935`, `--argument-ownership --argument-antecedent --antecedent-fit
conditional --antecedent-stretches sentence --arguments-within-sentence
--stacked-labeler --lexical-at word --words-name-only`, 20-second solve limit,
the live runtime shut down and nothing else computing. Incumbent composition
rows are v5's (`INCUMBENT_ROWS_REUSED.txt`).

v11: `~/.aura/rlc-evidence/semantic-peak-antecedent-v11-20261006`, from
`5b3c54aa2`, candidate
`3faa78effa17309abba975f97b4b6de690039fe942623b893a6ea9bb5ad337ba`,
development audit receipt
`dd81be70531b6402434b9b982127a0da3d39028fa04389c492b4eb10534cc0a3`,
`report.json` sha256
`129b2b35d919e925915aaa2e495f4d5b571c7ca195ded05015294a442b2b77da`.
v12: `~/.aura/rlc-evidence/semantic-peak-antecedent-v12-20261006`, from
`01701a935`, candidate
`10ec888499910568ad89aa08aadf54b3f4a0dafc3adb40539000008ec369ebd0`,
development audit receipt
`3d4c282e6b6ce769c01da00a0474839a803604485c1ad670cedc0ebf90856337`,
`report.json` sha256
`df3bc199f9a83f2012bed7d797e339313154b1de0b6d120765d5dac6755b771e`.
Neither used a test-split row.

| Cohort | Incumbent | v9 | v11 | v12 |
| --- | ---: | ---: | ---: | ---: |
| Train, 764 | 764 | 764 | 764 | 764 |
| Validation, 500 | 477 | 498 | 500 | 500 |
| Composition v1, 48 | 31 | 48 | 48 | 48 |
| Composition v2, 48 | 31 | 48 | 48 | 48 |
| Held-out folds, 764 | | 734 | 732 | 742 |
| fold 0, 256 | | 236 | 236 | 240 |
| fold 1, 254 | | 252 | 254 | 254 |
| fold 2, 254 | | 246 | 242 | 248 |

Against v9, row by row, v12 loses nothing: it gains the two `sequence-cataphoric-5`
validation rows ("after removing"), four `sequence-cataphoric-0` rows in fold
0, two `sequence-reserved-alias-2` rows in fold 1 and two `sequence-cataphoric-1`
rows in fold 2 ("3 reduced by").

## What this does not show

* The ratio and the split between which and whether were each chosen after a
  measurement: the ratio after the 48 validation rows, the split after
  v11's folds. The folds refit every readout without each construction
  group, and v12's folds are its own held-out measure.
* The two validation rows that closed are one wording, and the held-out gains
  add a second ("reduced by"). Transfer to wordings and constructions the
  development cohorts do not hold is what G04 measures.
* Twenty-two held-out rows are still wrong. They are generalisation to
  constructions no training row has, which is G04's, not development failures.
