# G09: kept procedures, attempt 4 (preregistered 15 October 2026, before attempt 3's verdict)

Attempt 4 tests the same claim as attempt 3, on attempt 3's requests and
attempt 3's drafts, with one change: how her book decides which kind of
problem a request is. It runs only if attempt 3 does not close G09. It is
committed before attempt 3's analysis or verdict exists.

## Why a fourth attempt

While attempt 3's drafts ran, its first kept arm showed a grading fault (the
13:55 amendment to preregistration 3), and the diagnosis that found it showed
a second fault that is not the grader's: the router. Her book routes a request
to the nearest kind whose bar it clears, by the share of each kind's problems
holding each word. Kinds whose problems share a template, every DeepMind
module's "Calculate ... Please reason step by step, and put your final answer
within boxed", clear each other's bars within a hundredth. A request of one
module is then answered by another module's procedure, wrongly, where her
draft was often right. Attempt 3 measures that system as registered; its
losses count there.

## The one change

Code at ec39df960 (worktree pinned for this attempt), on main the same commit.

- Signatures (`distinct_signatures`): each word is weighted by the logarithm
  of how many kinds there are over how many use it, so a word every kind
  shares counts for nothing; each kind's bar is taken again from its sealed
  half, the least typical sealed problem's similarity, as at induction.
  `tools/distinct_kept_book.py` writes the book from attempt 3's two books
  and the kinds' development halves.
- Route (`ProcedureBook.answer`): a request near enough to one kind is that
  kind's; near enough to several, each of their procedures answers and the
  answer stands only if all agree, and not at all when one of them keeps no
  procedure. Then the draft stands.

The procedures, the requests, the drafts, the graders (with the kept-answer
grading fix of preregistration 3's 13:55 amendment), the analysis and the
criteria are attempt 3's.

## What the change does on development problems

`tools/measure_kept_routing.py`, 1,692 development requests: 12 from the
development half of each of the 116 kinds, and 300 MATH training-split
problems posed as the harness poses them (seed 7). Output in
`~/.aura/rlc-evidence/g09-attempt4-routing-20261015/`.

| | attempt 3's book and route | attempt 4's |
| --- | ---: | ---: |
| kept kinds' own problems, answered right | 431 | 195 |
| kept kinds' own problems, answered wrong | 22 | 0 |
| other kinds' problems, answered wrong | 4 | 0 |
| MATH training problems, answered wrong | 0 | 0 |

Of the kept kinds' 480 problems, attempt 4's route answers 108 of 132 BBH,
24 of 24 planning and 63 of 324 DeepMind; it declines the rest. The DeepMind
modules still contest one another, their problems differing mostly in
operators the word signatures do not read.

## Files

- Books (sha256): attempt 3's procedure book
  `prefill-20261014/book.json` 4781dee3…a9191; signatures book
  `g09-attempt3-signatures/book.json` 40da0744…36883; attempt 4's book
  `g09-attempt4-routing-20261015/book.json` 01f73c6f…3381c.
- Requests: `g09-attempt3-requests-20261014/*.ids`, as attempt 3.
- Drafts: `g09-a3-drafts-<domain>-20261014/rows/ordinary`, as attempt 3.
- Kept rows: `rows/kept4` beside attempt 3's `rows/kept`.
- Runner: `~/.aura/rlc-evidence/g09-runs/attempt4.sh`; analysis and verdict in
  `~/.aura/rlc-evidence/g09-attempt4-analysis/`.

## Criteria

As attempt 3's, with the fourth attempt's bar: primary (pooled over the
domains, gains over losses), breadth (at least two of math, transfer and
planning, held-out kinds only, gains over losses) and no harm (no domain,
counting every request a kept procedure answered, losing more than it gains
at two-sided p below 0.05). Primary and breadth at two-sided p below 0.0125,
0.05 over four attempts. G09 closes only if all three hold, and then it is
shown live in her app before the model is used for anything else.

## What test data I have seen

Disclosed so the reader can weigh the change against it:

- Kept answers, under attempt 3's book and route, to six test requests of
  each attempt 3 domain, graded both ways, with the procedure that answered
  each wrong one. No draft of those requests.
- BBH boolean expressions' kept arm: 28 drafts right; 28 kept answers right
  once graded by the kind's own matching.
- The similarities of one development request to five kinds.

No other test outcome. Attempt 3's routing-only requests (61 MATH problems)
were chosen as those attempt 3's route answers; attempt 4 is graded on the
same ones, and a request its route would answer outside them is not measured
(on 300 MATH training problems it answered none).
