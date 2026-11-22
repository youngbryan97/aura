# The outside bar

<!-- reader-navigation:start -->
> A test states what will be measured, what it will be compared with, and what counts as a pass.
>
> Source code, a passing test, and a useful result in practice each answer a different question. To see whether memory helps, try the same tasks with memory enabled and disabled. Keep every success and failure.
>
> [Plain-language guide](../docs/PLAIN_ENGLISH_GUIDE.md#measurement) · [Terms and examples](../docs/VARIABLES_EXPLAINED.md#measurement) · [Evidence and related work](../docs/DEMOS_AND_RELATED_WORK.md#measurement)
> [All docs](../docs/DOCUMENTATION_INDEX.md) · [Glossary](../docs/GLOSSARY.md) · [Worked formulas](../docs/FORMULAS.md)
<!-- reader-navigation:end -->

This folder measures Aura with benchmarks that other people wrote. Her self-repair must not make these results worse.
Aura cannot change this folder or its results.

## What it measures

Each run asks the same frozen problems through `/api/chat`, the door a person uses.

| Domain | Source | Grader |
| --- | --- | --- |
| `blocksworld` | PlanBench, plan generation | STRIPS simulator of the domain's PDDL |
| `mystery_blocksworld` | PlanBench, plan generation | STRIPS simulator of the domain's PDDL |
| `math500` | MATH-500 test split | PRM800K `grade_answer` |
| `bbeh_mini` | BIG-Bench Extra Hard Mini | The authors' evaluator, restated |
| `cold_restart` | Facts made up for each run | The fact returns after a restart; a fact never told gets no code |

The planning problems come from the half that G09 kept for testing. No problem here drives a repair.

## How it is held apart

- `outside_bar/**` is tier 3 in `core/self_modification/mutation_tiers.py`.
- A draft that changes this folder waits for Bryan (`HER_OWN_GOVERNANCE`).
- The write gateway refuses every path in `outside_bar` and `~/.aura/outside-bar`.
- The code here imports nothing from her source.
- `SEAL.json` holds a SHA-256 for each file. A run stops if one file differs.
- Each ledger line holds the hash of the line before it.

Limit: Aura runs as the same operating-system user as this harness. A process outside her governed paths can still
write here. The seal and the ledger make such a change visible. They do not prevent it.

## Procedure

1. Do a check of the seal:

   ```bash
   python -m outside_bar.run check
   ```

2. Do a run. Use one name for each run:

   ```bash
   python -m outside_bar.run ask --run NAME --each 12
   ```

3. Do the cold restart for the same run. This command restarts Aura:

   ```bash
   python -m outside_bar.cold_restart --run NAME --facts 12
   ```

4. Compare the run with the baseline:

   ```bash
   python -m outside_bar.run compare --baseline BASELINE --candidate NAME
   ```

The exit code of `compare` is 0 for `HELD`, 1 for `REGRESSED` and 2 for `INCOMPLETE`.

## How a result is judged

The comparison counts the problems that changed between two runs. A problem that was right and is now wrong is a loss.
A regression is a set of losses with less than a 5% chance under a fair coin. A single domain must pass the same test
at 5% divided by the number of domains.
