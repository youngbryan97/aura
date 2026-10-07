# G06: the reader against the ordinary model, more compute, matched controls and lesions

G06 asks that a gain be compared with the ordinary model, with equal- and
greater-compute alternatives, with matched controls and with causal lesions,
and that selection, retries and regressions be accounted for. Every
comparison below was declared in a published plan before its requests
existed: G04's second plan (`d35f8110bb12`, [result](G04_TRANSFER_V2_RESULT_2026-10-07.md))
and G05's plan (`9bbc11dc58b2`, [result](G05_PUBLIC_ANSWERS_RESULT_2026-10-07.md)).
Each was run as frozen and is reported whichever way it came out.

## Causal lesions (G04 second run, alpha 0.025 each)

The candidate is G04's v17. Each lesion takes one fitted part out of the
frozen candidate and leaves the rest identical.

| Lesion | Stratum | v17 | Lesioned | v17 only | Lesion only | p | Rejects |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| phrase readouts removed | vocabulary | 33 | 6 | 27 | 0 | 7.5e-9 | yes |
| recency weights zeroed | depth | 54 | 19 | 35 | 0 | 2.9e-11 | yes |

Each lesion costs nothing where its part is not used. Without phrase
readouts, depth, family and construction stay at 54, 62 and 62; without
recency, vocabulary, family and construction stay at 33, 62 and 62.

## The ordinary reader and the earlier candidate (G04 second run)

The incumbent (G03's starting point) and v12 (G03's closing candidate) were
decoded on every request. Incumbent: vocabulary 21, depth 8, family 47,
construction 62. v12: vocabulary 9, depth 7, family 62, construction 62.
Against the incumbent, v17 lost 7 vocabulary requests and none elsewhere.

## The ordinary model, more compute and a matched control (G05)

Her answers rendered as her runtime renders a settled answer, private
channel closed, on 48 fresh compositions; ordinary_open is the same request
with her channel open at her serving effort. Exact counts follow the plan's
answer rule as the independent verifier applies it.

| Arm | Exact | Median tokens | Median seconds | Total seconds |
| --- | ---: | ---: | ---: | ---: |
| ordinary, closed | 29 of 48 | 723 | 43.0 | 2,862 |
| assisted, closed | 48 of 48 | 7 | 1.6 | 79 |
| sham, closed | 33 of 48 | 696 | 40.8 | 2,805 |
| ordinary, open | 42 of 48 | 876 | 50.6 | 2,909 |

The assisted arm's seconds are her decode only. The reader's own cost,
measured in G10 through her worker on the same requests, is one forward to
read the request's states (median 0.57 s) and the reading itself
(median 0.78 s), so an assisted answer costs a median of
3.0 s in all and the 48 cost 141 s.

* Ordinary model at equal compute: her shortest closed answer of the 48 was
  507 tokens and 29.4 s, so at the assisted arm's 3.0 s every
  ordinary answer would be stopped before it was written. The closed
  ordinary arm as run has 20 times the assisted arm's total
  seconds, and assisted beats it 19 to 0 (primary, p = 1.9e-6).
* Greater compute: with her private channel open she uses 21
  times the assisted arm's total seconds and is exact 42 times. Assisted wins
  6 to 0, two-sided p = 0.031 at alpha 0.025, so this preregistered
  comparison does not reject. More compute closes most of the gap; it does
  not close all of it, and six discordant pairs are too few to say more.
* Matched control: a sham reading, another request's program in the same
  form, is the control for the presence of evidence. On compositions
  (descriptive) it leaves her at 33 exact against 48 for the right reading.
  On lists (preregistered) a wrong reading costs her 55 lists she reads
  right alone and helps 4, one-sided p = 8.5e-13. The gain is the reading's
  content, and wrong content harms.

## Selection, retries, regressions

* Selection: G04's candidate was chosen on development only (training,
  validation, held-out training folds and the first run's consumed
  requests). Four candidates were rejected there and are recorded with the
  reason ([development record](G04_V17_DEVELOPMENT_2026-10-07.md)). Nothing
  was selected after either plan was published.
* Retries: none. One decode per request and arm. Every failure, refusal
  and budget stop counts against its arm.
* Regressions: against v12, v17 reads two G03 validation requests and nine
  held-out training constructions fewer; against the incumbent it loses 7
  vocabulary requests the incumbent had right. G05's assisted arm lost no
  request its ordinary arm had right, on compositions or lists. The sham
  stratum is the regression a wrong reading would cause: 55 lists.
* Grading: four G05 rows the runner misgraded are listed, read and
  resolved by the plan's rule in the G05 result. Both defects are fixed.

## What this does not show

* That the lesioned parts are the only cause of the gains: each lesion
  removes one part, and interactions are not separated.
* A win over her open channel at alpha 0.025.
* Anything outside the two populations measured.
