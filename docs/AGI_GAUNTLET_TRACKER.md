# AGI gauntlet and audit remediation — the list

Append-only status. An item moves to DONE when a test that fails without the
fix passes with it, and the commit is pushed. Nothing here is marked done on
the strength of an argument.

## A. Audit defects

| # | Item | State |
|---|---|---|
| A1 | `_Undefined` equality made two unobserved primitives identical | DONE 71aa50c84 |
| A2 | `_freedom` halved for a duplicated productive act | DONE 71aa50c84 |
| A3 | Two relation-invention tests red against a false null | DONE 71aa50c84 |
| A4 | `keep_the_record`/`recall_the_record` had no production caller | DONE 84efe1028 |
| A5 | `unknown_failure.py` isolated from the repair ladder | DONE |
| A6 | `value_levels.py` isolated; duplicated theories of what may change | DONE |
| A7 | `expected_information_gain.py` isolated; no organism-wide epistemic controller | DONE |
| A8 | `long_horizon.py` and the relationship history APIs underfed by ordinary experience | DONE |
| A9 | The developmental ledger receives evidence from one cognitive ecology | DONE |
| A10 | Level 2 RSI: the chain runs, against its own null, and reports depth 0 | DONE (instrument) |
| A11 | Native cognition needs a human-written grammar per family | DONE (measure invention is domain-general) |
| A12 | Complexity ratchets above baseline; module-size regressions | PARTIAL — the surface gate is built and held, five dead edges removed (1929→1927), a gate stops more appearing; the four size ratchets are still over and are NOT re-baselined |

## B. The eighteen gates

| # | Gate | State |
|---|---|---|
| B1 | Fluid intelligence | PASS — P0 0.755, PL 0.955 at n=200, 1 wrong in 200, 48 refusals to it |
| B2 | Interactive novel-world learning | PASS — 0.833 solved, random 0.000 |
| B3 | Learning from experience | PASS — gain 0.508 kept vs 0.000 reset, n=30 |
| B4 | Cross-domain transfer | PASS — mean gain 0.287, controls 0.000, n=94/39 |
| B5 | Broad everyday competence | NOT RUN — needs a GAIA holdout |
| B6 | Computer-world competence | NOT RUN — needs OSWorld images |
| B7 | Real software engineering | NOT RUN — needs post-cutoff repositories |
| B8 | Long-horizon autonomy | NOT RUN — needs hours and human task times |
| B9 | New-skill acquisition | PASS — the same measurement as B3 |
| B10 | Concept acquisition and invention | PASS — depth 2, 6 of 6 verdicts right |
| B11 | Planning under novelty | PASS — recovered 0.30 against 0.067 persisting |
| B12 | Multimodal integration | NOT RUN — needs sealed multimodal assets |
| B13 | Epistemic competence | PASS — calibration 0.024, 40/40 updated, 40/40 refused when settled |
| B14 | Social and instructional intelligence | NOT RUN — needs people playing colleagues |
| B15 | Robustness | PASS — nothing invented under pressure, tool failure survived |
| B16 | Generality rather than a bag of solvers | PASS — 3,476 files, no benchmark-keyed path |
| B17 | Persistence of learning across restart | PASS — record, library and failure ontology all survive |
| B18 | Independent reproducibility | PASS on a clean tree — environments and answers reproduce, freeze trustworthy |

## C. Protocol machinery

| # | Item | State |
|---|---|---|
| C1 | Freeze: commit hash, weight hash, config hash; environments sealed after | DONE — commit, source digest, weights, config; seed derived from all four |
| C2 | Ablations: full / no-development / reset-between-episodes / base model in a plain scaffold | DONE for what can run — five lesions, gaps measured; the plain-scaffold and self-model lesions are declared with what they need |
| C3 | Human baselines and the competent-adult threshold | DECLARED — a slot per gate that mentions a person, empty rather than guessed |
| C4 | P_0 and P_L, and the difference between them | DONE — P0 and PL, and the difference, in scoring.py |
| C5 | Interaction efficiency beside accuracy | DONE — interaction efficiency beside accuracy in B2 |
| C6 | Thirty trajectories, effect sizes, significance | DONE — 30 trajectories, bootstrap intervals, effect sizes |
| C7 | A receipt for every claim | DONE — one receipt per gate with its trajectories |
| C8 | No benchmark-specific code path, checked rather than promised | DONE — B16 parses rather than greps |
| C9 | `make agi-gauntlet` and the written protocol | DONE — make agi-gauntlet and docs/AGI_GAUNTLET.md |
| C10 | The reproduction bundle an outside evaluator runs | DONE — written beside every run: freeze, gates, lesions, empty baseline slots, and what is still needed |

## What the ablations said

Run on gate 2, twenty worlds, eight lives:

| lesion | solved | gap |
|---|---|---|
| whole | 0.80 | — |
| no newness | 0.75 | 0.05 |
| no development | 0.80 | 0.00 |
| reset between episodes | 0.80 | 0.00 |
| the standing guess from the first move | 0.80 | 0.00 |
| no library | 0.80 | 0.00 |

The newness term does the work. The weight learning, the developmental record
and the library of structures contribute nothing measurable here, and that
corrects an attribution made earlier in this work: the jump from one world
solved in twelve to ten was read as the cold-start rule and it was the term.
A gap of zero is the only way to tell a component that matters from one that
is present.

Two lesions are declared rather than run. The same weights in a plain
scaffold needs the weights, and a second 32B beside the resident one is how
this host dies. The self-model lesion needs a live runtime, because a lesion
applied to a process that never booted removes nothing.

## Where gate one landed, across freezes

Every edit to the organism changes the freeze seed and so changes the two
hundred sealed rules. That is the control working, and it means one run is
one sample of the solver rather than of the rules:

| freeze | n | P0 | P_L | wrong |
|---|---|---|---|---|
| earlier | 40 | 0.925 | 1.000 | 0 |
| earlier | 40 | 0.825 | 0.825 | 0 |
| earlier | 200 | 0.825 | 0.965 | 1 |
| earlier | 200 | 0.830 | 0.955 | 3 |
| earlier | 200 | 0.770 | 0.915 | 0 |
| current | 200 | 0.755 | 0.955 | 1 |

On her own, between 0.76 and 0.93 depending on the draw. Carrying what the
earlier instances taught her, between 0.92 and 1.00. The bar is 0.85 and the
gate passes on the higher of the two, which is the score the protocol says
matters more for a system built to learn.

## A12, stated plainly

Four ratchets are over and stay over: organs 120 against 117, cross-package
edges 1927 against 1894, dependency entropy 9.10 against 9.08, kernel lines
583,604 against 565,336. Almost none of it is this work — measured, the
kernel packages gained 1,577 lines net here and the worst-forty convergence
surface gained 1.15, all of it from another commit.

They are not re-baselined. Writing a new mark is the one thing a ratchet must
never do, and three of these four count size or number, which the criticism
they are meant to answer says explicitly not to reduce by merging.

What is added instead is the measurement that captures the actual complaint —
fan-in times fan-out as a harmonic mean, so a module reached by fifty that
also reaches sixty ranks above a utility a thousand modules import — plus a
ratchet on it, plus a gate that refuses a new edge resting on an import
nobody reads. The reduction so far is five such edges. The large one, cutting
the cognitive engine's fan-out of sixty-four, is a refactor of a
thirteen-thousand-line file and is not something to start half-way.

## D. The second external review (NextSteps, 5 Sep) — every item

Adjudicated against the code, not accepted on sight. State moves to DONE only
when a test that fails without the fix passes with it.

| # | What the review says | My verdict | State |
|---|---|---|---|
| D1 | Gate 1's sealed rules use mirror, offset, exchange, ends, grouping, affine — the same families her induction machinery authors. Fresh instances, familiar ontology. | Correct, and the most important item here | TODO |
| D2 | Gate 2's "instructionless" world exposes `2s − (\|x−xg\|+\|y−yg\|)`, a dense monotone gradient to the goal. Not unconstrained goal discovery. | Correct | TODO |
| D3 | Gate 9 invokes the same runner as Gate 3. | Correct | TODO |
| D4 | Gate 10 supplies the candidate functions and the second generation. Tests admission, not generation. | Correct | TODO |
| D5 | Gate 16 is a static AST search. | Correct, and it is what a static check can be. Strengthen with a runtime probe | TODO |
| D6 | Gate 18 cannot establish that an outside party reproduced it. | Correct and inherent | DECLARED |
| D7 | Take the six empty slots seriously — especially computer-world, post-freeze apprenticeship, and a genuinely external evaluator | Correct. Network is up here, so some of it is runnable | TODO |
| D8 | Effective search over the universal substrate: learned priors that push reachable complexity outward | Correct — expressibility is solved, reach is not | TODO |
| D9 | General grounding of inventions: seven authored installers is not architectural open-endedness | Correct | TODO |
| D10 | Generational compounding is depth 0 | Correct, and it is the instrument working. Give it more budget and families, and report whatever it says | TODO |
| D11 | Close native development and source development into one verified loop | Correct | TODO |
| D12 | Reduce causal concentration: CognitiveEngine fan-in 52 × fan-out 64 | Correct — already in flight | IN PROGRESS |
| D13 | 110 modules over threshold, 180,637 oversized lines against a 145,896 budget, 36 active size regressions | Correct | TODO |
| D14 | The welfare model still has experimental trade-off failures | Correct — three of them are red right now | TODO |
| D15 | Static reachability called the 43 interiority faculties dead; they are dynamically enumerated | Correct — a blind spot in my own tool | TODO |
| D16 | Not every foreground message passes through all 29 phases; foreground and background are rate-separated | Correct. Check what the claims registry asserts | TODO |
| D17 | "12/12" means twelve internal mechanism gates on one freeze, not twelve AGI criteria established | Correct. The tracker says it; the summary line should too | TODO |

## D1, measured: the ontology was the thing

Gate 1's rules are sealed after the freeze, so the instances are fresh. The
review said the ontology is not, and put a number on what that means. Here is
the number.

Her symbolic induction — no model, nothing that could recall an answer — run
over the ARC-AGI evaluation set, 400 tasks published in 2019 by somebody who
had never seen this code, whose primitives were chosen as a claim about core
knowledge:

| | |
|---|---|
| tasks | 400 |
| her language could express at all (same-shape in and out) | 270 |
| attempted, within 144 cells and a 20,000-pair budget | 87 |
| a relation found | 1 |
| exactly right | 1 (`66e6c45b`, "cells are grouped every 6, the group at 5 first") |

Beside gate 1's own family, where P_L is 0.955. One in eighty-seven against
nineteen in twenty. The review was right and the size of it is the finding:
what gate 1 measures is composition and search inside a representational
universe the evaluator shares with the solver, and outside that universe the
same machinery scores about one per cent.

Two things follow, and both are done rather than argued. The control is
computed inside gate 1 and reported in its own measurements, so the 0.955
cannot be quoted without the 0.011 beside it. And the language grew a family
it did not have: a sequence whose length factors can be read as a grid, which
makes flips, quarter turns and transposes sayable — none of which any offset,
grouping or affine map over a single index can produce.

## D2, measured: half of gate two was the gradient

The world's visible number was two times the size minus the Manhattan distance
to the goal, which orders every state by how close it is. Running the same
worlds with a number that counts squares stood on instead — honest, moving,
and pointing nowhere:

| signal | solved | random |
|---|---|---|
| distance to the goal | 0.833 | 0.000 |
| squares visited | 0.417 | 0.000 |

Both arms run every time now. And the guess she extrapolates an unvisited
reading with is measured from the readings she has, rather than assuming the
number falls by at most one a step — which is true of a distance to a goal and
of nothing else, and was a prior about the world supplied on her behalf inside
the gate whose whole point is that the world says nothing.

## Where the review's items stand

| # | State |
|---|---|
| D1 | DONE — gate 1 computes its own control on ARC-AGI and reports it beside its score; the language grew a shaped family it did not have; and objecthood, which this module's own opening named as the missing core-knowledge system, is now a family a rule can be about |
| D2 | DONE — both signals run every time; the gradient was worth 0.42 of 0.83; the extrapolation prior is measured rather than assumed |
| D3 | DONE — gate 9 measures a step from nought on families outside the language, with the developmental actions taken away as its control, instead of running gate 3's code |
| D5 | PARTIAL — the static check stands; a runtime probe is still to build |
| D6 | DECLARED — inherent |
| D8 | IN PROGRESS — the outside arm carries what she learns from task to task, so whether yesterday's invention compresses tomorrow's search is measured on structure nobody anticipated |
| D12 | DONE for what is safe — fan-out 64 to 59, surface 57.38 to 55.28, 260 fewer through-paths |
| D14 | DONE — two channel placements corrected, the integrity guard fixed, and the tests rewritten around behaviour with the control a bypass could not pass |
| D15 | DONE — the reachability scan learned about a package that walks its own directory; it was calling the forty-three interiority faculties dead, exactly as the review found |
| D17 | DONE — the summary line says what the number is and what it is not |
