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
| A12 | Complexity ratchets above baseline; module-size regressions | PARTIAL — convergence-surface gate built and held; size ratchets still over |

## B. The eighteen gates

| # | Gate | State |
|---|---|---|
| B1 | Fluid intelligence | BAR NOT MET — 0.65 right, 0 wrong, bar 0.85 |
| B2 | Interactive novel-world learning | PASS — 0.867 solved, random 0.067 |
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
| B18 | Independent reproducibility | PASS on a clean tree — environments and answers reproduce |

## C. Protocol machinery

| # | Item | State |
|---|---|---|
| C1 | Freeze: commit hash, weight hash, config hash; environments sealed after | DONE — commit, source digest, weights, config; seed derived from all four |
| C2 | Ablations: full / no-development / reset-between-episodes / base model in a plain scaffold | PARTIAL — reset ablation runs (B3); base-model-in-a-plain-scaffold needs the model |
| C3 | Human baselines and the competent-adult threshold | NOT RUN — needs human baselines |
| C4 | P_0 and P_L, and the difference between them | DONE — P0 and PL, and the difference, in scoring.py |
| C5 | Interaction efficiency beside accuracy | DONE — interaction efficiency beside accuracy in B2 |
| C6 | Thirty trajectories, effect sizes, significance | DONE — 30 trajectories, bootstrap intervals, effect sizes |
| C7 | A receipt for every claim | DONE — one receipt per gate with its trajectories |
| C8 | No benchmark-specific code path, checked rather than promised | DONE — B16 parses rather than greps |
| C9 | `make agi-gauntlet` and the written protocol | DONE — make agi-gauntlet and docs/AGI_GAUNTLET.md |
| C10 | The reproduction bundle an outside evaluator runs | PARTIAL — the run reproduces; an outside evaluator is still needed |
