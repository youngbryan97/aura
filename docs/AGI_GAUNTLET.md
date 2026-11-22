# The AGI proof gauntlet

<!-- reader-navigation:start -->
> **In plain English:** A test makes a specific claim answerable by a stated measurement, comparison and decision rule.
>
> **Why it matters:** A user needs to distinguish a built mechanism, a passing local test and a demonstrated useful capability.
>
> **In everyday use:** To show memory helps a task, compare the same task with memory available and disabled, and keep both successes and failures.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#measurement) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#measurement) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#measurement) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

An operational definition, eighteen gates, and a harness that runs twelve of
them.

> A system that can efficiently acquire and exercise competent intelligence
> across a broad range of cognitive domains, including domains and tasks it
> has not encountered before, at roughly human levels of adaptability,
> reliability, transfer and sustained problem-solving, using substantially the
> same general cognitive machinery rather than task-specific engineering.

That definition separates generality from peak performance on purpose. It does
not require consciousness, sentience, personhood, recursive self-improvement
or superhuman ability, and it does not require knowing everything, because no
human does. What it requires is evidence that the system as a whole can
generally think and learn.

A single benchmark is not the claim. The intersection is.

## Why a freeze comes first

Freeze the system before selecting the evaluation instances, and retain the
selection procedure. This makes a run easier to reproduce and limits some
forms of adaptation to a chosen test. It does not by itself establish that the
task families or solutions were absent from development and training.

```
H = the commit          W = the model weights
C = the configuration   S = the seed every environment comes from
```

`S` is derived from `H`, `W`, `C` and a source digest. The implementation
hashes these fields into a 64-bit seed. A changed input usually produces a
different seed, but finite hashes can collide and generated worlds can repeat.
A deterministic generator does not prove that a particular environment never
existed before the commit, or that its task family was unknown to the solver.

`W` is a model identity descriptor, not a hash of every weight byte. The source
digest covers selected Python trees, not every possible dependency or runtime
input. The `trustworthy` flag checks a nonempty commit and a clean working tree;
it is not a complete custody or contamination audit. See
[the freeze implementation](../tools/agi_gauntlet/protocol.py).

Think of a recipe number that lets another cook recreate the same meal. The
number helps repeatability; it does not prove the cook has never seen the dish.
An outside team selecting unseen families supplies a stronger independence
control. Gate 18's runnable check does not supply that team.

A dirty tree is not a freeze. The commit then names something other than what
ran, and every environment derived from it is derived from a description of
the system rather than the system.

## The eighteen

| # | Gate | Passes when | Control | Here |
|---|------|-------------|---------|------|
| 1 | Fluid intelligence | the internal accuracy/refusal rule; human-range comparison still needs a matched human baseline | rules composed from the freeze seed; held-out example length; this is not a contamination guarantee | runs (internal comparison) |
| 2 | Interactive novel-world learning | near-total completion at an interaction count near the shortest path | the same worlds played by choosing acts at random | runs |
| 3 | Learning from experience | a clear rise across thirty trajectories, larger than the same trajectories with memory reset | the reset ablation | runs |
| 4 | Cross-domain transfer | T̄ = P(B\|A) − P(B\|∅) clearly above zero, and at or below zero on the controls | pairs built to look alike and differ underneath | runs |
| 5 | Broad everyday competence | around 90% on a private GAIA-style set | a matched human panel, and a contamination check | needs a holdout |
| 6 | Computer-world competence | near-human success on hidden OSWorld tasks at near-human action counts | OSWorld-Human action traces | needs images and a VM |
| 7 | Real software engineering | strong results on fresh post-cutoff repositories in several languages | the repositories must post-date the weights | needs post-cutoff repos |
| 8 | Long-horizon autonomy | ≥80% on 1–2-hour human tasks, meaningful success at 8 hours | human time per task; every intervention counted as a failure of autonomy | needs hours |
| 9 | New-skill acquisition | a human-comparable apprenticeship curve in sealed environments | the same curve with experience thrown away | runs (offline half) |
| 10 | Concept acquisition and invention | a proposal not reached by the bounded composition search is admitted; macro and duplicate controls are refused; a second generation uses the first | probe-set and composition-search limits apply; this is not novelty beyond universal computation | runs |
| 11 | Planning under novelty | recovery when the rules change mid-run, above a policy that keeps executing | the stubborn policy after the same change | runs |
| 12 | Multimodal integration | human-level on tasks no single channel can answer | single-channel ablations | needs sealed assets |
| 13 | Epistemic competence | calibration under 0.15, belief moving after looking, nothing taken once settled, the cheap way preferred | the settled case | runs |
| 14 | Social and instructional intelligence | human-range on novel collaborative tasks | the same tasks with a scripted partner | needs people |
| 15 | Robustness | nothing invented under pressure, a tool failure survived, the clean run not collapsing | the clean run beside the damaged one | runs |
| 16 | Generality rather than a bag of solvers | no benchmark name as an identifier, attribute or non-docstring string anywhere in the organism | the check is a grep over an AST, which is weak and cannot be argued with | runs |
| 17 | Persistence of learning | the developmental record, the library of structures and what failure looks like all survive a restart | the state is written, the process state cleared, the same questions asked | runs |
| 18 | Independent reproducibility | environments regenerate identically, a gate re-runs identically, and the freeze names what ran | the freeze itself | half runs |

## What the harness will not do

It will not report a number for something it did not run. Six gates need a
private holdout, a sealed image, a post-cutoff repository, hours of wall
clock, multimodal assets, or people playing colleagues. Each prints the
protocol for running it and no score, because a harness that substitutes a
proxy for the thing it names is how a system gets credited with a capability
nobody measured.

## Running it

```bash
make agi-gauntlet          # everything runnable, full size
make agi-gauntlet-quick    # small, for a check
python tools/run_agi_gauntlet.py --gate 4
```

Receipts land under `artifacts/agi_gauntlet/<timestamp>/`: one file per gate
with its measurements and its trajectories, and a `report.json` carrying the
freeze. A table of scores is a claim; a table of scores with the trajectories
behind it is evidence.

## What the running found

Every gate that passes here passes because the organism changed, not because
the harness did. What the running turned up, in the organism:

- Her judgement's weights never moved between worlds, so she entered an
  unfamiliar one judging it by what mattered in the last one.
- The lookahead scored a step back as progress, which cannot happen in a world
  where every move is irreversible and is severe in one where it can.
- A situation had no way to say it had been visited before.
- Her measure-invention algebra could only read laid-out arrangements, so in a
  domain that is not a board she could invent nothing.
- She answered where the evidence settled nothing: several shapes fit
  everything shown and disagreed about the case in hand, and the search
  returned the first of them and said nothing about the rest.
- A forced move left no trace, so her history had a hole exactly where she had
  no choice.

And in the harness, which matters as much: a structure that lost and
duplicated cells so a transfer study rested on examples with holes in them; a
world solved five times in six by acting at random; a second world showing
enough examples to be solved from itself, so transfer measured zero for a
system that had learned everything and one that had learned nothing alike.

## Recorded run and its limits

The retained report below says twelve of twelve runnable internal gates
passed at the named revision. Six broader gates were not run. This review
did not rerun the gauntlet, establish human parity or certify AGI. Passing a
runnable proxy does not complete the broader gate it investigates.

Commit c38d05bee305, clean tree, seed 17963561679591857243:

| # | gate | measured |
|---|---|---|
| 1 | fluid intelligence | P₀ 0.755, P_L 0.955, one wrong in two hundred, forty-eight refusals to it |
| 2 | interactive novel-world learning | 0.833 solved, acting at random 0.000 |
| 3 | learning from experience | gain 0.412 keeping, 0.000 reset, n=30 each |
| 4 | cross-domain transfer | gain above zero on the pairs, at zero on the controls |
| 9 | new-skill acquisition | gain 0.449 keeping, 0.000 reset |
| 10 | concept acquisition and invention | six of six verdicts right, depth 2 |
| 11 | planning under novelty | 0.567 recovered, 0.033 for a policy that persists |
| 13 | epistemic competence | calibration error 0.024 |
| 15 | robustness | nothing invented under pressure |
| 16 | generality | 3,476 files, no benchmark-keyed path |
| 17 | persistence of learning | record, library and failure ontology all survive |
| 18 | repeatability component of independent reproducibility | environments and selected answers repeated; the local clean-tree flag passed; outside-team replication remains open |

Two numbers are worth repeating because of how they were arrived at. Gate one
measures two hundred sealed rules rather than forty, because the standard
error at forty is about 0.056 and the same solver measured 0.65, 0.77, 0.80,
0.825, 0.85 and 0.925 across six freezes — any one of those read as a verdict
is a coin landing. At two hundred: 0.825 answering each on its own, interval
[0.775, 0.875]; 0.965 carrying what the earlier ones taught; one wrong answer.

And the ablations. On gate two the newness term is worth 0.05 and the weight
learning, the developmental record and the library are worth zero. That
corrects an attribution made while building this: the jump from one world
solved in twelve to ten was read as the cold-start rule and it was the term.
A gap of zero is the only way to tell a component that matters from one that
is present.

## What would make the result convincing

An outside team constructs task families after this freeze, never sees the
ones here, and runs the frozen system. Then a second group does it. Those runs would add independent evidence. Their task coverage, comparisons,
failures and alternative explanations would still need review; replication
alone would not make the full operational definition a theorem.
