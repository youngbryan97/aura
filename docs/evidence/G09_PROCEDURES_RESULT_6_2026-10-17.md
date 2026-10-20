# G09: kept procedures, attempt 6 result (17 October 2026)

**Attempt 6 meets its registered criteria, and read row by row it still meets them: 29 gained to 0 lost,
p 3.7e-9, with math 13 to 0 and planning 16 to 0 on held-out kinds and no domain harmed. Both verdicts
hold, so G09 closes.** The run drafted from 04:45 to 17:05; the registered verdict was computed at 17:05:34
and grader version 3's at 17:05:56. Every discordant row was read between 19:05 and 19:20, before this was
written.

## As registered

Attempt 6 ran as preregistered (G09_PROCEDURES_PREREGISTRATION_6_2026-10-17.md; part 1, how each kind's bars
are set, before any was measured; part 2, the bars chosen and the requests drawn, before any test request
existed). Drafts came from a worktree pinned at part 2's commit (43164a636), her model alone, greedy, with
her thinking budget of 32,768 tokens. Both arms of every request were graded by one function per domain
(tools/g09_one_grader.py, version 2 as registered).

| Criterion | Registered verdict | Bar | Holds |
| --- | --- | --- | --- |
| Primary, every domain drafted | 35 gained, 0 lost, p 5.8e-11 | p < 0.0083 | yes |
| Breadth, held-out kinds: math | 13 gained, 0 lost, p 2.4e-4 | p < 0.0083 | yes |
| Breadth, held-out kinds: planning | 22 gained, 0 lost, p 4.8e-7 | p < 0.0083 | yes |
| No harm, every domain | none lost anything | none harmed at p < 0.05 | yes |

444 fresh test requests, none drawn by attempts 3 or 5: 351 over the 27 held-out DeepMind mathematics kinds
that keep a procedure (13 a kind), 50 Depots and 43 Logistics problems (PlanBench). Part 2 registered 50
Logistics problems; the Logistics test split holds 143 and attempts 3 and 5 had drawn 100 of them, so 43 were
all that was left. The MATH test problems the route reaches were to be drafted so that harm there counts; under
part 2's bars the route reached none of them, so no MATH problem was drafted and none could be lost.

| Domain | Requests | Her drafts right | Kept arm right | Kept answer adopted |
| --- | ---: | ---: | ---: | ---: |
| Math, held-out kinds | 351 | 329 | 342 | 141 |
| Planning | 93 | 71 | 93 | 93 |

## Read row by row

Every gained and lost row was read with the truth, her draft and the kept answer side by side
(`~/.aura/rlc-evidence/g09-runs/read_discordant.py a6 20261017 kept6_one_grader`). Every row where a kept answer
was adopted and graded wrong was read as well, to find kept answers the grader misread; there were none.
There were no losses to read.

Held-out math, 13 gained, all hers to lose:

- 12 are problems her draft never answered. On two next-term sequences, four primality questions and six
  prime-factor lists her draft spent its whole thinking budget and gave no answer; each kept answer equals the
  truth (for example 2, 11, 13, 36341).
- 1 is a wrong answer. Asked the units digit of 56182115, her draft assumed an exponent had been lost
  ("56^182115"), answered 6, and then noted that the literal number ends in 5. The kept procedure answered 5,
  which is right.

Planning, 22 registered gains; 16 hold.

- The 11 Logistics drafts are wrong plans, each replayed step by step against the instance
  (scratchpad tool `why_plans_fail.py`, using the grader's own simulator). Seven fail at their first or second
  step by loading a vehicle where it is not: in instance 271 her first step loads package_5 into truck_0 at
  location_0_1, and truck_0 starts at location_0_0. Four fail later by loading a package into an airplane at
  an airport the package never reached. None closes with a valid plan anywhere in the reply.
- 6 of the 11 Depots drafts hold a valid plan the graders did not read. Five are the class grader version 3
  was built for: reasoning before the plan read as steps (instances 429, 397, 312, 459, 192; in 192 her first
  plan is invalid and the revision after it is valid). The sixth, instance 458, both versions miss: she wrote
  a valid 12-step plan in PlanBench's own phrasing, then a "Verification of the plan" restating each step in
  bold with notes, and no "[PLAN END]". Version 1 read both as one plan; version 3 read the last run of steps,
  the verification, whose bold notes it cannot parse. Every run of action lines in every planning draft was
  checked (`every_run.py`); 458 is the only valid plan version 3 missed.
- The other 5 Depots drafts are invalid as written: a hoist loads into a truck that has not driven there
  (494, 309, 122), a crate is dropped onto a pallet that is not clear (466), a crate is unloaded from a truck
  it was never loaded into (5).

| Criterion | As read | Bar | Holds |
| --- | --- | --- | --- |
| Primary | 29 gained, 0 lost, p 3.7e-9 | p < 0.0083 | yes |
| Breadth: math | 13 gained, 0 lost, p 2.4e-4 | p < 0.0083 | yes |
| Breadth: planning | 16 gained, 0 lost, p 3.1e-5 | p < 0.0083 | yes |
| No harm | nothing lost in any domain | none harmed at p < 0.05 | yes |

Grader version 3's automatic verdict (`g09-attempt6-analysis/read/verdict.json`) is 30 to 0 with planning 17
to 0; it still counts Depots 458. The table above removes it as well.

## What it shows

On held-out kinds, with fresh requests and one grader for both arms, procedures she wrote and kept by
evidence answer problems her model alone does not, in both domains, and cost her nothing. The size of the
gain is set by where her model fails, not by where the procedures are strong: on 20 of the 27 math kinds her
drafts were already right on every request, and the kept arm matched them. Of the 7 kinds where she missed
some, the kept arm recovered her misses in 4 and missed the same problems in 3 (the n-th term of a sequence,
a division, a composed remainder). In math the gain is almost all
problems too long for her thinking budget (factoring and primality of large numbers, the next term of
high-order sequences). In planning it is plans that keep track of where every vehicle and package is, which
her drafts lose track of in Logistics and in about a tenth of Depots.

It does not show that a kept procedure makes her reason better about a problem it does not compute, and it
does not show transfer to competition mathematics: under attempt 6's bars the meaning route declined every
MATH test problem, which is what removed attempt 5's 11 losses.

## What closing does

G09 is marked closed in AURA_1_0_MASTER_TODO.md in the same commit as this result. Her live book has carried
attempt 6's bars since 17:21 (`~/.aura/rlc-evidence/live-bars-20261017/`). The preregistration's last step,
showing it live in her app, is recorded below when done.

## Amendment, 18 October 2026: shown live in her app

Typed into her chat at 13:26:10, as a user types it: "What are the prime factors of 21000063?" She answered
"3, 7, 1000003" at 13:28:11. Her log has the kept procedure for `dm_numbers__list_prime_factors` answering
before her draft by words then meaning, and the answer served as the computed one (known answers agreed:
5000). The screenshot and the log lines are in `~/.aura/rlc-evidence/meanings-ending-20261018/`.

The first try, on 17 October, was declined. By its words the question was as near to the composed kind (the
number is first computed from equations), which keeps no procedure, as to the plain kind, and the plain kind's
procedure answers composed problems wrongly, so the decline was right. Two changes made it answerable:

- A meaning tie-break among the kinds the words leave near (45a9689c3): the kind the request is typical of
  answers when exactly one is.
- Each kind's meaning now leaves out what every one of its problems closes with (dfbfa5bcb). Every DeepMind
  problem ends with an instruction to reason step by step and box the answer, which a person's request never
  carries; with it in the meaning the question scored 0.733 against the plain kind's bar of 0.759.

The meanings were rebuilt with the ending left out and the bars raised again by attempt 6's chosen rule
(`above_every_wrong`, seed 20261017, from the attempt's pinned worktree). Before installing, both wordings of
the question routed to the plain kind and answered 3, 7, 1000003; the three composed examples were declined;
and L09's question ("How many days are there between May 6, 2013 and September 21, 2017?") still answered
1599 (`routing.jsonl` in the same folder). The book went into her app at 13:24, with the previous one kept
beside it as `kept_from_solved_examples.json.before-meanings-ending-20261018`.

The routing change touches no attempt-6 test request: the test arms were graded on the book as it was.
