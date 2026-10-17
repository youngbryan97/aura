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

## Part 2, committed 17 October 2026 at 04:14, before any test request is drawn or drafted

### The rule chosen

tools/g09_bars_above_others.py at f27526f32, seed 20261017: 3,749 MATH training problems set the bars, and
3,749 more with 1,392 development requests measured them (`g09-attempt6-routing-20261017/rules.json`).

| Rule | Right | Wrong | Declined | Wrong among the MATH problems |
| --- | ---: | ---: | ---: | ---: |
| As attempt 5 | 274 | 10 | 4,857 | 10 |
| Above every other | 297 | 2 | 4,842 | 2 |
| Above every wrong | 273 | 1 | 4,867 | 1 |

Every rule answered some development request wrongly, so by part 1's rule the one with the fewest wrong is
chosen: above every wrong. It raised 8 word bars and 11 meaning bars, among them the division, multiplication
and remainder kinds the MATH problems reached. Every wrong answer was to a MATH problem; the kinds' own
development requests were answered rightly or declined under each rule.

### The book

`g09-attempt6-routing-20261017/book_above_every_wrong.json` (sha256 begins f95aa70b7460bcc8): attempt 5's book
with those bars.

### How many requests

Attempt 5's real gains on held-out math, read by hand and graded alike by one grader version 2, were 5 of 12
primality problems, 4 of 11 prime-factor problems and 1 of 11 base conversions, none lost. Routed through this
book, attempt 5's 300 math requests still give those 10 gained and none lost. With no loss, 8 gains reach
p below 0.05 / 6 (p 0.0078). Drawing 13 requests from each of the 27 kinds, the chance of at least 8 gains is
0.93 at those rates; 11 a kind, about what attempt 5 drew, gives 0.80. So 351 math requests, 13 a kind. Planning
stays at 50 Logistics and 50 Depots: attempt 5's real planning gains, 15 of 100, put 8 within easy reach.

### The requests

tools/g09_requests_to_draft.py with `--route words_then_meaning` and this book, seed 20261017, excluding every
request attempts 3 and 5 drew: 351 over the held-out DeepMind kinds that keep a procedure (`--spread 351`), 100
over Logistics and Depots, and every MATH test problem the route answers (`--routing-only math`).

### The run

`~/.aura/rlc-evidence/g09-runs/attempt6.sh`, from a worktree pinned at the commit of this part: drafts by
tools/run_g09_organ.py (`--arms ordinary --batch 8`, seed 20261017) only while her app is down, the kept arm
by tools/g09_kept_procedure_arm.py (rows/kept6), both arms graded by tools/g09_one_grader.py version 2
(rows/kept6_one_grader), then tools/g09_analysis.py and tools/g09_verdict.py with
`--alpha 0.0083 --breadth-need 2 --breadth-among math,planning` (0.05 / 6 exactly).

### The claim

G09 closes if, at two-sided p below 0.05 / 6, the kept arm gains over her drafts pooled over every domain
drafted, and in each of math and planning on held-out kinds, and no domain loses more than it gains at p below
0.05. Before the result is written, every discordant row is read with its truth and both answers. A row the
reading shows misgraded is listed, and the verdict is computed again with it counted as the reading says. G09
closes only if both verdicts hold. Then it is shown live in her app.
