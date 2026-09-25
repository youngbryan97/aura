# G03 nested selection and research, 2026-09-25

This is source-only development evidence. It grants no serving authority, G03
closure, general transfer, or broad reasoning claim.

## Nested experiment

The outer construction fold contains 256 sources. Its proposal bank was
generated without cross-fit updates on those sources: 205 have a witnessed correct
program, the ordinary selector chooses 179, and the bank's joint score
chooses 192. These are different measurements: 205 is an observed reach
ceiling for this finite proposal bank, not a claim that the other 51 problems
are impossible.

To train a selector without reusing its own proposal model's fit examples,
`probe_semantic_proposer_crossfit.py --outer-fold 0` split the other 508
sources into three inner held folds. A source-only split seed was selected
before reading held labels so that each inner proposer's fit complement
retained at least two source geometries. Each inner proposal bank has a signed
plan and row receipts; the selector trainer replays the split from source
constructions and refuses overlap or mixed provenance.

The shared parent transducer was fitted before this experiment on the full
source cohort. The nested protocol excludes outer labels from the *new*
cross-fit proposer and selector updates, but it does not erase the parent's
earlier exposure. Consequently this is development selection evidence, not
a fresh no-exposure transfer result.

| Inner fold | Population | Correct reachable | Ordinary correct | Joint-score correct |
| --- | ---: | ---: | ---: | ---: |
| 0 | 170 | 91 | 88 | 86 |
| 1 | 170 | 144 | 130 | 133 |
| 2 | 168 | 153 | 107 | 115 |

The selector learned from 388 inner rows with a witnessed correct program,
for three epochs and 1,164 updates. Its training loss fell 0.480 to 0.174.
On the untouched outer fold it chose 108/256 correct versus the incumbent's
179/256, with 5 gains and 76 regressions. **Rejected.** Lower source-fold
training loss is not a transferable selection mechanism. The weight file is
development evidence only and must not be loaded by serving. Evidence:
`~/.aura/rlc-evidence/semantic-nested-ranker-outer0-20260925/report.json`
and the three `semantic-nested-proposer-seed1-outer0-inner*-20260925` banks.

## Research hypotheses, not imported authority

Wikipedia's [compositionality](https://en.wikipedia.org/wiki/Principle_of_compositionality),
[program synthesis](https://en.wikipedia.org/wiki/Program_synthesis),
[abduction](https://en.wikipedia.org/wiki/Abductive_reasoning), and
[symbol grounding](https://en.wikipedia.org/wiki/Symbol_grounding_problem) pages
were used as concept maps. Their links led to the primary work below.
Public discussions were read as sources of questions, not empirical verdicts.

| Lead | Aura hypothesis | Discriminating experiment | Decision |
| --- | --- | --- | --- |
| [Coverage principle](https://arxiv.org/abs/2505.20278) distinguishes structure-based, property-based, and shared-operator generalization. | The missing 51 outer programs may be structural/argument compositions absent from the finite bank, not ranking errors. | Group held misses by operation inventory, graph shape, and argument bindings; compare source construction coverage and exact witnessed reach. | Adopt as a failure taxonomy, not a promised gain. |
| [Learning compositional rules](https://arxiv.org/abs/2003.05562) and [compositional program-synthesis benchmark](https://arxiv.org/abs/2204.03758) separate learned rule systems from direct answer prediction, and test new combinations/depth. | Explicit rule induction may generate an otherwise absent program, while a selector cannot. | Freeze source-only rule learning; measure new held candidate reach separately from selected accuracy, with equal-compute and rule-lesion controls. | Candidate mechanism for the proposal deficit. |
| [DreamCoder](https://arxiv.org/abs/2006.08381) and [LAPS](https://arxiv.org/abs/2106.11053) learn reusable program libraries and search heuristics. | Reusable operators might reduce search cost on new constructions. | Include abstraction acquisition cost; evaluate fresh construction/family held tasks and lesion/rescue of the invented operator. | Reuse Aura's existing operator-invention path; no second library. |
| [Causal invariant transformations](https://arxiv.org/abs/2203.11528) studies transformations that alter nuisance features while preserving a causal mechanism. | A source paraphrase or variable permutation should preserve executable semantics and expose a selector relying on surface shortcuts. | Transform held and source tasks with independently verified semantic equivalence; test equivariance and failure correlations. | Adopt as a controlled robustness test, not a training shortcut. |
| [Compositionality](https://plato.stanford.edu/archives/win2025/entries/compositionality/) and [pragmatics](https://plato.stanford.edu/entries/pragmatics/) distinguish constituent meaning from contextual speaker meaning. | Some binding errors require source-grounded reference resolution, but ordinary arithmetic graphs should not gain a hidden social-pragmatic prior. | Compare typed graph cases with and without ambiguous discourse context; independently label referents. | Preserve the existing scoped pragmatic substrate; do not inject unverified context into G03 labels. |
| [Abduction and discovery](https://plato.stanford.edu/entries/scientific-discovery/) distinguishes hypothesis generation from selection. | Candidate absence and wrong selection require different repairs. | Report bank reach, selection conditional on reach, and public answer separately. | Already enacted in the source miss profile; retain. |
| [MachineLearning discussion of OOD definitions](https://www.reddit.com/r/MachineLearning/comments/1b1he3r) questions what distribution shift means. | A result called transfer may only recombine covered fragments. | Name held axes and source coverage, including unseen structures and new families; compare matched in-distribution controls. | Adopt as a question for G04, not as evidence. |
| [MachineLearning discussion of meta-learning](https://www.reddit.com/r/MachineLearning/comments/17ghrbz) questions whether training-example diversity, rather than a new model architecture, drives systematicity. | Construction imbalance across inner folds may explain the selector's failure. | Report per-construction source coverage and per-construction gains/regressions before changing model size or architecture. | Investigate, without fitting to outer labels. |
| [AskPhilosophy discussion of meaning as use](https://www.reddit.com/r/askphilosophy/comments/1c5ok1m) raises usage versus referential grounding. | Usage evidence can suggest a reading but cannot prove a task's executable truth. | Keep utterance/context evidence distinct from execution/observation evidence and measure each consumer. | Fits existing pragmatic-evidence separation; no further claim. |

The [no-free-lunch analysis](https://arxiv.org/abs/2304.05366) is a reminder
that finite-budget transfer needs assumptions about task structure. It does
not make this particular failure inevitable or license changing the bar.

Next: diagnose the 76 regressions by construction and candidate graph,
then test a source-only proposal mechanism against the 51 observed reach
misses. Do not tune against the outer fold after reading its labels; use a
new outer fold for any revised candidate and reserve fresh tasks for G04-G08.
