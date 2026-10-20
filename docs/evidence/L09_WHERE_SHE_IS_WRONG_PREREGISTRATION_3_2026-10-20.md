# L09, attempt 3: what she learns from being corrected, on the kinds people ask that she gets wrong. Preregistration

Written 20 October 2026 before any answer of hers to these requests exists. The plan is
[the design](../design-docs/L09_NEXT_LEARNING_WHERE_SHE_IS_WRONG.md); the kinds and their programs were fixed and
pushed before the draw (`tools/l09_where_she_is_wrong.py`, 413cabb23).

## The requests

`l09_where_she_is_wrong.py draw` over WildChat-4.8M: every English user turn whose whole text is one question of a
kind a program settles, less the requests L09's first two attempts used, each kind in the order typed. A user turn
there carries no time of its own, so the order is that of the conversations' times (the first draw, before this was
seen, gave every row an empty time and is kept in `first-draw-no-times/`; the counts are the same). Drawn 20 October,
04:30 to 04:35 by the files' times (`~/.aura/rlc-evidence/l09-where-she-is-wrong-20261020/drawn.json`), 293 requests:

| kind | drawn | used |
|---|---|---|
| letters | 71 | yes |
| sum | 62 | yes |
| product | 54 | yes |
| percent_of | 35 | yes |
| conversion | 20 | no |
| quotient | 14 | no |
| difference | 14 | no |
| what_percent | 13 | no |
| weekday | 6 | no |
| days_between | 4 | no |

Kinds with fewer than 30 drawn are not used: too few to estimate, teach and test. percent_of leaves 5 requests for
step 3 after steps 1 and 2; that is what there is, and its count is reported as it falls.

## Step 1: where she is wrong

For each kind, its first 20 requests in the order typed are put to her persona 27B bare, by her chat's own way, with
her kept procedures off (her app down; `l09_where_she_is_wrong.py wrong`, which decodes a request as attempt 2's draft
step decodes its "off" arm: no system text, her template, thinking on, greedy, 32,768 tokens at most). Each reply is
graded by the rule below. A kind is kept for steps 2 and 3 when she is wrong on at least as many of its 20 as she
is right. If no kind is kept, the attempt ends there and that is its result: on what people ask that a program
settles, her model alone is mostly right.

## Step 2: learned in use

For each kept kind, its next 10 requests in order are typed into her chat at localhost:8000, each once; after each
reply the person states the answer, "No, it's N." when hers is wrong and "Yes, the answer is N." when right. Her log
must show "kept 'N', the answer a person gave" for each. Her app then stays up and idle until her log shows a
"Learning in use" step with status "kept" for the kind, or 24 hours have passed (a miss for that kind, with the
steps' reports as its cause). Nothing is written for her: the procedure is what she induced.

## Step 3: measured on fresh requests

The kept kinds' requests after those of steps 1 and 2, never shown to her, in order, each answered by her chat's own
route twice: with her book as it was before step 2, and as it is after it (frozen copies, her app down), with the
restatement check of 37021f4ba. Counting stops at the smallest number n of requests at which, with p her error rate
on the kind in step 1, a binomial count of n at p reaches 5 or more with probability 0.9, or when the kind's
requests run out.

**Criterion.** Over the discordant pairs of all kept kinds together, gained exceeds lost at a one-sided sign test
p below 0.05. Every discordant pair is read by hand from a sheet that shows the two replies as X and Y in an order
drawn per request, all read before the key is opened. Reported beside: each kind's gained and lost, and where the
kept procedure answered wrong.

## Grading, one rule for both arms, fixed now

- **A whole number** (product, sum, difference, days between, letters, and any kind whose exact answer is whole):
  the last boxed answer, else the last paragraph that states a number the request did not, states the number
  (thousands commas allowed). Attempt 2's rule 2, with the number read with its sign and its decimals (`right` in the
  tool), because attempt 2's reader of whole numbers drops a minus sign and would grade "-7" as 7.
- **A number that is not whole** (quotient, percentage, conversion): the same region states a number within 0.1% of
  the exact answer, relative.
- **A day of the week:** the reply's closing region names that day and no other.

## What it can and cannot show

That being corrected the way a person corrects her improves her answers to what people ask of the same kind. Not
that people correct her (the corrections are the programs' answers, given as a person would), and not that her book
helps on kinds she already gets right.
