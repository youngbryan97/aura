# G09: kept procedures, attempt 5 result (17 October 2026)

**As registered, attempt 5 meets all three criteria. Read row by row, it does not: 13 of its 38 gains are the
grader's, and without them the primary criterion is 25 gained to 11 lost, p 0.029, short of its 0.01. G09
stays open.** The run finished at 02:08; every discordant row was read before this was written (02:15-02:30).

## As registered

Attempt 5 ran as preregistered (G09_PROCEDURES_PREREGISTRATION_5_2026-10-16.md; part 1, the route rule,
committed before any route was measured; part 2 before any test request was drawn). Both arms of every request
were graded by one function per domain (tools/g09_one_grader.py).

| Criterion | Registered verdict | Bar | Holds |
| --- | --- | --- | --- |
| Primary, every domain drafted | 38 gained, 11 lost, p 1.4e-4 | p < 0.01 | yes |
| Breadth, held-out kinds: math | 14 gained, 0 lost, p 1.2e-4 | p < 0.01 | yes |
| Breadth, held-out kinds: planning | 24 gained, 0 lost, p 1.2e-7 | p < 0.01 | yes |
| No harm, every domain | math 14 to 11 (p 1.0); planning 24 to 0 | none harmed at p < 0.05 | yes |

413 fresh test requests, none drawn by attempt 3: 300 problems over the 27 held-out DeepMind mathematics kinds
that keep a procedure, 50 Depots and 50 Logistics problems (PlanBench), and the 13 MATH test problems the route
reaches, drafted so that harm there counts. Her draft (her model alone, greedy) against the same request after
her kept procedures, reached by words and, where words decline, by meaning.

## Read row by row

Every gained and lost row was read with the truth, her draft and the kept answer side by side. One check ran
the other way as well: every row where the kept answer was adopted and graded wrong was read, to find kept
answers the grader had misread. There were none; all 11 are the MATH losses below.

Held-out math, 14 gained:

- 10 are hers to lose. On five primality and four prime-factor questions her draft spent its whole thinking
  budget (32,768 tokens) and gave no answer; the kept procedure answered in milliseconds. One base conversion
  she got wrong (-1344012 in base 7 is -179055, -2bb6f in base 16; she wrote -63EC).
- 4 are the grader's. A base-2 difference written with its base as a subscript ("_2"); "Yes, it is
  composite." for True; two factorizations ("2^3 · 11 · 263 · 3739") where the kind lists the distinct primes.
  Each draft was right.

Planning, 24 gained:

- The 10 Logistics drafts are wrong plans. Each fails a precondition, most at its first or second step: in
  instance 202 her first step loads package_5 into truck_2 at location_2_1, where truck_2 is not.
- 9 of the 14 Depots drafts are valid plans the reader could not find. She wrote numbered steps under a
  heading, with her reasoning before them and a check after, and did not close the plan with "[PLAN END]", so
  PlanBench's reader took her reasoning for steps. Read from the numbered steps alone, all nine reach the
  goal. The other 5 Depots drafts are invalid as written.

Math, 11 lost, all on the 13 MATH competition problems the meaning route reached. Her drafts were right on
all 13; the kept procedures on 2. "The largest prime factor of 9951" was answered by the procedure that lists
prime factors ("3, 31, 107"); "the hundreds digit of 5·6·7·8·9·10" by the place-value procedure ("30").

| Criterion | Without the 13 | Bar | Holds |
| --- | --- | --- | --- |
| Primary | 25 gained, 11 lost, p 0.029 | p < 0.01 | no |
| Breadth: math | 10 gained, 0 lost, p 0.002 | p < 0.01 | yes |
| Breadth: planning | 15 gained, 0 lost, p 6.1e-5 | p < 0.01 | yes |
| No harm | math 10 to 11 (p 1.0) | none harmed at p < 0.05 | yes |

Attempt 3 was reopened when its gain proved to come from grading. The same reason applies here, so the
registered verdict is not taken as G09 closed.

## What it shows

On held-out kinds, with fresh requests, the kept procedures beat her in both domains and lost nothing there:
in math by answering problems too long for her thinking budget, in Logistics by writing valid plans where hers
were not. What fails is the meaning route. It sends a competition problem to the procedure of a kind it only
resembles, and those 11 losses cancel the math gain (10 to 11 over the math domain).

## What attempt 6 changes

Each change is fixed before any attempt 6 test request is drawn, and each grading change applies to both arms.

1. The one grader reads a base written as a subscript, a yes or no inside a sentence, and a factorization's
   distinct primes where the kind lists them.
2. A PlanBench plan with no "[PLAN END]" is read from its numbered steps.
3. The meaning route is measured on development problems that include competition math (MATH's training
   half), which part 1's 1,692 development requests did not. Its rule for declining a request that only
   resembles a kind is chosen there.
4. The number of math requests comes from attempt 5's rate of real gains, so that the breadth test has the
   power to reach 0.01, rather than from a round number.

## Record

Requests `~/.aura/rlc-evidence/g09-attempt5-requests-20261016/`; drafts and rows
`g09-a5-drafts-<domain>-20261016/rows/{ordinary,kept5,kept5_one_grader}`; analysis and verdict
`g09-attempt5-analysis/` (`all_domains.json`, `held_out_kinds.json`, `verdict.json`, `one_grader.jsonl`); routes
and book `g09-attempt5-routing-20261016/`. Code at 95a0cec76 (worktree `g09-attempt5`).
