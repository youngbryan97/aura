# G04 second candidate: what the first run's failures taught, and what it cost

The [first G04 run](G04_TRANSFER_RESULT_2026-10-06.md) rejected on family only.
Its vocabulary and depth failures were read row by row, which consumed them.
This is the development record from those failures to candidate v17, which
the [second plan](G04_TRANSFER_V2_PREREGISTRATION_2026-10-07.md) freezes. Every
number below is on training, validation, held-out training folds or the first
run's consumed requests; none is on the second plan's requests.

## Depth: a repeated description read back to the first clause

All 54 depth failures had the right operations in the wrong order. For "the
intermediate value" at steps three to six of a seven-step chain, the
antecedent readout gave the previous step's result log P of -3 to -5 and the
first operation's -0.1 to -0.9. Its features favour the first clause whose
words match, and a description repeated in every clause matches them all.
No single evidence term was to blame: ablating the antecedent, ownership,
triadic or proposal terms left depth at 6 to 8 of 62.

Training could not have taught otherwise: in its 764 requests no result is
more than two operations back, so the first clause to match and the latest
result were always the same. A recency count fitted there came out with the
wrong sign. Recency became learnable once training had chains of three to
five steps (`training_breadth_v1`, inside the depth everything consumed
already has).

## Vocabulary: a verb read before its phrase settles

All 41 vocabulary failures named the wrong operation. "take 7 away from 46"
read as addition at "take"; "share 31 evenly among 5" as addition at
"share". A causal state at the verb cannot hold the rest of the phrase. The
token tagger did mark the phrase's later words ("away", "evenly", "whole
parts"), and a readout at the phrase's close named 104 of the 124 held-out
operations against 55 at the verb, while losing in-distribution (validation
0.888 against 0.995). Training said every operation three ways, each led by
a verb that names it, so no readout could learn where a phrase settles.

`training_breadth_v1` adds 24 wordings whose meaning settles late, none
sharing a word with the second plan's vocabulary table, which was committed
first (`f1d2ba1f0`). Held out a wording at a time, the verb reading named
65.8% of unseen wordings and the three readings pooled at the phrase's close
86.5%, with held-out constructions at 0.995 and validation 0.993.

## Candidates that were not frozen

| Candidate | Change | Train | Validation | Fold 0 | Fold 1 | Fold 2 | Not frozen because |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| v12 (G03) | | 764 | 500 | 240 | 254 | 248 | first run's candidate |
| v13 | recency; phrase closes guessed per candidate | 763 | 496 | 241 | 254 | 234 | lost 5 development requests |
| v14 | + mention direction; families balanced | 764 | 500 | 241 | 254 | 234 | lost 14 held-out nested nominals |
| v15 | + phrase closes at the chosen chart's next operation | 764 | 463 | | | | lost 33 sequence validation requests |
| v16 | + candidates keep two-way weights; equal votes in renaming | 764 | 498 | 241 | | | lost 2: "multiplicity calculation" read as a multiplication |
| v17 | + breadth teaches spans and phrases, not naming | 764 | 498 | 238 | 253 | pending | frozen |

v15 and v16 were stopped once validation decided them.

* v13's recency could not tell "the intermediate value" (back to the latest
  result) from "the subsequently computed number" (forward), and 300 breadth
  chains outweighed every cataphoric training mention. v14's antecedent
  carries a mention direction that gates the recency features, and weighs
  each construction family alike: all five lost requests came back.
* v14 lost 14 held-out nested nominals ("the quotient of 84 divided by the
  quotient of 47 divided by 3"): a close guessed per candidate ran an outer
  operation's phrase through a weaker inner one (fold 2's refit: 44 of 64
  with phrase readings, 56 without). In v15 a chosen chart is renamed, each
  phrase closing at that chart's next operation: 56 of 64.
* v15 named candidates with the four-way stacked weights less two terms, so
  the word readout outweighed the context. Fitted, the renaming weights put
  the context at zero, because nothing held out in training shows the phrase
  wrong and the verb right; validation's "counting copies of one value with
  selector 1" reads as a lookup by its phrase. v16 keeps the candidates'
  two-way weights and gives each readout one vote in renaming. On held-out
  breadth wordings, decoded, renaming with equal votes read 393 of 501 and
  with fitted weights 389; without renaming, 350.
* v16 lost two validation requests, both "apply multiplicity calculation"
  read as a multiplication. Either of v12's candidate labelers put them
  right: breadth wordings built with misleading verbs had taught the
  candidate labelers that verbs mislead. In v17 breadth teaches the tagger
  and the phrase readouts only. That costs transfer: decoded on held-out
  breadth wordings, v17's design reads 315 of 501 against v16's 393. It is
  the price of not undoing G03, and it is recorded as open work.

## v17 on the first run's consumed requests

Decoded with execution unavailable and graded as the first run graded:

| Stratum | v12 (first run) | v17 |
| --- | ---: | ---: |
| construction | 62 | 62 |
| vocabulary | 21 | 50 (5 refused); these phrasings are now breadth training |
| depth | 8 | 58 |
| family | 59 | 62 |

## What this does not show

* Transfer: every number here is on development or consumed requests. The
  second plan's requests were generated after v17 was frozen.
* That nothing was lost. Against v12, v17 reads two validation requests wrong
  (G03's "after removing", renamed through the next operation's lead-in),
  two held-out fold-0 constructions and one fold-1 construction; fold 2 is
  still running and is appended when it ends. It loses no request the
  incumbent had right. A gate that renamed only when the renamed reading was
  surer kept the two "after removing" requests but blocked most renames that
  carry unseen wordings (45 of 121 against 94 on the first held-out quarter),
  so it is not used. These are open work under G04, not hidden.

## Appended 2026-10-07: v17's held-out folds, complete

Fold 2 ended after the plan was published: 242 of 254 (v12 248). Across the
three folds v17 reads 733 of 764 held-out training constructions against
v12's 742: two fewer in fold 0, one in fold 1, six in fold 2. Folds hold out
constructions of the training families, not wordings or depth, so this is the
cost the second plan's construction claim can see on fresh requests: every
construction request there has to be right for G04 to close.
