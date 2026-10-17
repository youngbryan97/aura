# G09: kept procedures, attempt 6 (part 1 committed 17 October 2026 at 03:51, before any bar is measured)

Attempt 5 met its registered criteria, but read row by row 13 of its 38 gains were the grader's, and without
them the primary criterion was 25 gained to 11 lost, p 0.029 (G09_PROCEDURES_RESULT_5_2026-10-17.md). The 11
losses were MATH test problems the route sent to DeepMind kinds, 9 by their words and 2 by meaning. Attempt 6
is drafted on fresh test requests, graded by one grader version 2 for both arms (tools/g09_one_grader.py at
bc9879416), at two-sided p below 0.05 / 6, about 0.0083 (0.05 over six attempts).

This first part fixes how each kind's bars are set, before any bar is measured. Part 2, committed before any
test request is drawn, names the rule chosen, the requests, their number and the code.

## The book

Attempt 5's (`g09-attempt5-routing-20261016/book.json`), routed by words and then, where words decline, by
meaning (tools/g09_routes.py, `words_then_meaning`). Only its bars change.

## The rules compared, on development data only

A kind's bars are its least typical sealed problem's, by words and by meaning, and nothing set them against
problems of other sources. Every MATH training problem with an answer (`~/.aura/benchmarks/g09/math-train`) is
split in two by seed 20261017: the calibration half sets bars; the measurement half, with 12 development
requests from each kind (seed 7, as attempt 5's part 1 drew them), measures them.

1. As attempt 5: the bars as kept.
2. Above every other: each kind's word bar and meaning bar raised above every calibration problem's nearness to
   the kind by that measure.
3. Above every wrong: raised above only the calibration problems that clear the kind's bar and that its
   procedures answer wrongly.

Each development answer is graded by its own kind with tools/g09_one_grader.one_grade (version 2).

## The rule

The rule chosen answers no development request wrongly and, of those, answers the most rightly. If every rule
answers some request wrongly, the one with the fewest wrong is chosen, and of those the one with the most
right. The tool is tools/g09_bars_above_others.py; its output goes to
`~/.aura/rlc-evidence/g09-attempt6-routing-20261017/`.

## Domains and criteria

As attempt 5's: math (the 27 held-out DeepMind kinds that keep a procedure, and the MATH test problems the
route reaches, drafted so harm there counts) and planning (Logistics and Depots). Primary pooled over the
domains, breadth in both, no harm in either. The number of math requests is set in part 2 from attempt 5's rate
of gains read by hand (10 of 300), so that the breadth test has the power to pass, rather than from a round
number.
