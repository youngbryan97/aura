# Self-repair on defects from her own history: first measurement, 18-19 October 2026

Item 5 of the evaluation of 17 October: her self-modification loop had never staged a fix. This is the first
measurement of the replacement (`core/self_modification/repair_by_selection.py`): a failing test is the defect,
candidates are proposed, and a candidate mends only when the failing tests pass, the tests that passed before
still pass, and ruff and the writing gate find nothing new in the changed files.

## The defects

`tools/defects_from_her_history.py` read every commit on main since 10 October whose subject starts "fix" and
which changed source and tests, ran the tests it changed on its parent with its test files put in, and kept the
commits where those tests fail there and pass with the commit's source files: 81 defects with known answers
(`~/.aura/rlc-evidence/repair-by-selection-20261018/defects.json`). Most are fixes of several lines across one or
two files, not one-line faults.

## Her own Python alone

Proposals from `core/self_modification/proposals_without_a_model.py`: a missing import copied from her own code,
the edits for shapes of code almost never meant, ruff's fixes, and fix templates (comparison, boolean and
arithmetic operators, shifts by one, negated conditions, swapped arguments, flipped literals, another name in
scope, a dropped statement) at the lines the failing tests run, ranked by Ochiai. Ten minutes a defect.

| Defects run | Mended | Nothing proposed | Ran out of time | Waiting (their tests run MLX) |
|---:|---:|---:|---:|---:|
| 67 | 0 | 33 | 34 | 14 |

On planted one-line faults the same proposers mend at once: a removed import on the first candidate
(2d626f651) and a flipped comparison on the 44th (bf8d580ce). On the faults she actually had, none.

## Her 27B on her code lane

Proposals from her persona 27B alone (`--only-model`), with each defect given an equal share of the night left
until 07:00 (two to five minutes each), from 22:59 to 06:59 (`~/.aura/rlc-evidence/aura-fixes/run27b-20261018-1656/repair/`).

| Defects run | Mended | Ran out of time |
|---:|---:|---:|
| 79 | 4 | 75 |

Each mend, read:

- **3ae986ac8** (her model answers who she is, what she can do and how she would plan). It removed the three
  written floors (planning, capability, identity) and the planning outline from her chat's quick reply, the
  change the commit made, without its explanatory comment; 4 of 4 tests fixed, the 4 guarding tests held.
- **9c27d78cc** (induction grades a BBH kind's answers with its own grader). A five-line change that looks the
  grader up by the kind an example belongs to; 2 guarding tests held. Sound.
- **89826b8fc** (BBH kinds posed with BBH's worked examples). Poses the problem with the worked examples and keys
  it by task, as the commit did; 29 guarding tests held. Sound.
- **b093aad7b** (a book written for her lands where her launched app reads it). The test passes, but no test
  imports the tool, so nothing guarded it, and the patch carries the model's thinking as comments ("Actually,
  `git rev-parse --git-common-dir` returns the path to the shared git directory."). The writing gate did not
  catch them. It counts as mended by the rule; it is not a change to keep.

## What it shows

Her own Python's exact changes mend planted faults and none of her real ones, which are wider than one line.
Her model mended 4 of 79 in a few minutes each; three of the four are sound and one shows two gaps in the gate:
a candidate with nothing guarding it, and comments that think aloud. Nothing here applies a change to her
running code.
