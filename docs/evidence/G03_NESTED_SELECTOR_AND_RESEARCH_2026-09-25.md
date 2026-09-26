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

## Where the remaining proposals disappear

The signed 4-chart bank was replayed against the source targets only **after**
generation. Independent candidate comparisons were revalidated. Of 256 outer
sources, 205 have an equivalent proposal, 43 never yielded the target operation
chart, and eight yielded that chart but no equivalent argument graph. These are
finite-search observations, not impossibility proofs: every one of the 51
missing proposals has incomplete search. Artifact:
`~/.aura/rlc-evidence/semantic-proposal-gaps-outer0-verified-20260925/report.json`.

The operation-node diagnostic expands the *same frozen candidate's* source
inventory offline, without using the target during proposal. Across the 51
misses, 47 retain all target operation nodes in the ranked inventory. Four
cataphoric examples prune one target operation label; none lose the source
span. The rest are chart or graph search failures downstream of a retained
node. Artifact:
`~/.aura/rlc-evidence/semantic-operation-inventory-outer0-20260925/report.json`.

Simply widening the chart allowance from four to eight on the 43 missing-chart
sources found an equivalent proposal in 12, but selected none correctly in
ordinary decode. This diagnostic subset excludes baseline successes and is
not a matched full-cohort gain. An existing complete-operation-search policy,
now exposed in the cross-fit diagnostic with a signed expansion allowance,
was tried on one source at 1,000 expansions and eight charts. It stopped
incomplete after four charts and found no equivalent proposal. This is a
bounded search-cost result, not evidence that the target is absent. Artifacts:
`~/.aura/rlc-evidence/semantic-chart-width-outer0-misses-20260925/report.json`
and `~/.aura/rlc-evidence/semantic-complete-op-pilot-outer0-20260925/report.json`.

The immediate mechanism problem is not an empty universal vocabulary. It is
how to allocate finite graph-search work over plausible operation charts and
then select among valid candidates. Any revised policy needs source-only fit,
matched cost, a new construction holdout, and controls preserving ordinary
successes. No new policy is qualified by these diagnostics.

## Signature-preserving search, still diagnostic

An exact bounded dynamic program now keeps the highest-scoring nonoverlapping
chart for each ordered operation sequence. Randomized small-graph tests compare
every result to exhaustive enumeration; the table has an explicit refusal on
work exhaustion. This is not exhaustive graph search: a lower-scored span
placement with the same operations may bind differently.

On the 43 outer-fold sources without a target operation chart, 39 target
operation sequences appear within the top five *signature representatives*.
The other four are precisely the label-pruned cataphoric cases identified
above. This is target-visible **offline diagnosis**, not an inference-time
oracle. Evidence:
`~/.aura/rlc-evidence/semantic-operation-signature-outer0-20260925/report.json`.

The existing transducer now has an opt-in `signature_diverse_v1` policy that
records its work bound in its signed receipt. On one exposed source it surfaced
the missing `mul, idiv` chart; eight graph alternatives then witnessed an
equivalent program. Ordinary selection remained wrong. A deliberately small
fold-1 pilot selected three former misses and three former ordinary successes
before running the new policy. All six had an equivalent proposal; the three
controls stayed ordinary-correct, while the three rescued misses stayed
ordinary-wrong. This is a 3/3 proposal-reach rescue and 0/3 answer rescue on
a label-selected pilot, not a transfer estimate or promotion evidence.
Artifacts: `~/.aura/rlc-evidence/semantic-signature-pilot-outer0-20260925/report.json`,
`~/.aura/rlc-evidence/semantic-signature-pilot-widegraph-outer0-20260925/report.json`,
and `~/.aura/rlc-evidence/semantic-signature-pilot-fold1-20260925/report.json`.

The six-source pilot took about eight minutes: its 20-second solve allowance
is reused across chart/graph attempts rather than a whole-source allowance.
The next mechanism must address selection and work accounting together,
preserve the incumbent's answer in a mixed candidate bank, and then earn a
full new-fold comparison with matched time and source-only training.

The offline candidate bank now also accepts a separate, opt-in whole-bank
allowance. It preserves the ordinary decode and marks diagnostic search
incomplete when that allowance is spent; it does not turn an unobserved
candidate into an impossibility verdict or shorten Aura's conversational work.
One fold-1 control replay with a 30-second bank allowance remained
ordinary-correct and candidate-reachable. The pilot's end-to-end process still
took about 52 seconds including source/model loading, so the allowance must
not be reported as whole-turn latency. Evidence:
`~/.aura/rlc-evidence/semantic-signature-bank-budget-pilot-fold1-20260925/report.json`.

## Argument-evidence selection result

The nested source-fold selector was retrained with operation-conditioned
argument and definition evidence. Three epochs used 388 reachable training
rows and 1,164 updates. On the unchanged outer bank it selected 151/256,
against 179/256 ordinary correct: 18 gains and 46 regressions. This improves
on the plain selector's 108/256 but still loses 28 answers net. The frozen
weights are rejected for serving. Evidence:
`~/.aura/rlc-evidence/semantic-argument-ranker-full-outer0-20260925/report.json`.

The frozen selector was also evaluated against a wider bank on the 51
previously missed outer sources. Thirty became candidate-reachable and it
selected 12 correct, versus zero ordinary correct on this selected subset.
That is diagnostic only: selecting misses after seeing labels cannot establish
a deployable population gain. On one signature-diverse exposed source, the
bank contained an equivalent program but neither selector found it. A signed
variant evaluator checks model weights, parent, source corpus, partitions,
bank receipts, and row identity before comparison. Evidence:
`~/.aura/rlc-evidence/semantic-argument-ranker-margin-outer0-20260925/report.json`.

The raw score advantage over the incumbent overlaps between gains and
regressions. The 18 gain margins range 2.09-34.57; the 46 regression margins
range 0.09-20.34. Choosing an override cutoff after reading these outer
labels would leak evaluation feedback into selection, so no cutoff is
qualified. Executing candidates establishes their outputs, not the source's
intended operation or binding. G03 remains open: selection needs stronger
independent source evidence, evaluated on a new construction holdout.

The source construction ledger localizes the harm. On 64
`fork_join:fork_begin_independently_combine` sources the ordinary selector was
64/64 and the ranker 44/64, with no ranker gains. On two 64-source sequential
arithmetic constructions, ordinary was 28/64 and 27/64; the ranker was 24/64
and 25/64. Those arithmetic constructions account for all 18 ranker gains,
but also 24 regressions. The family labels were read only for diagnosis after
outer evaluation. Routing by those labels now would be another leak.

Aura already has `core.evidence.calibrated_candidate_selector` and a semantic
path ensemble. Reuse them if an independent source-only calibration and a new
held construction admit a challenger. The present evidence cannot distinguish
a genuinely better reading from a high-scoring wrong graph on every source;
neither a second selector implementation nor a margin tuned on this fold
would close that gap.

The methods do have complementary correct choices. Replaying the 256 signed
rows gives 179 ordinary-correct, 192 joint-score-correct, and 151
argument-ranker-correct. Their label-aware union is 202; the proposal bank
contains 205 reachable correct programs. The overlap is 131 all-correct,
46 ordinary-plus-joint only, 10 joint-plus-ranker only, eight ranker only,
five joint only, two ordinary-plus-ranker only, and 54 all-wrong. This is an
oracle upper bound on choosing among these three outputs, not a result Aura
can obtain at runtime. The missing mechanism is target-free arbitration that
captures part of this complementarity on fresh constructions without
discarding the 46 ordinary-plus-joint successes. The overlap is replayable
from the signed bank and frozen weights at
`~/.aura/rlc-evidence/semantic-argument-ranker-overlap-outer0-20260925/report.json`.

An adapter to Aura's existing calibrated pairwise selector was attempted on
the nested construction banks. It exposes only candidate program geometry and
relative joint score as features; answer labels enter the independent fit and
calibration receipts, never runtime features. The three inner folds contain
20, five, and 12 sources where ordinary and top-joint choices differ. The
predeclared five-pair tuning fold supplies only ten binary observations;
`fit_calibrated_binary_scorer` requires 24, so it refused the scorer. No
selector was built and the outer fold was not evaluated through this policy.
Keeping agreement rows merely to inflate the calibration count would not add
switch evidence. Evidence:
`~/.aura/rlc-evidence/semantic-bank-pairwise-calibration-outer0-20260925/report.json`.
The signed construction ledger makes the limit sharper: all 20 fit pairs are
one `arithmetic:nominal_nested` construction, and each of the five tuning and
12 admission pairs is one distinct cataphoric construction. The 31 outer
disagreements span only two sequential arithmetic constructions. The adapter
now reports independent-construction support and refuses a one-construction
fit, calibration, or admission split even if paraphrases inflate its row
count. The revised receipt is
`~/.aura/rlc-evidence/semantic-bank-pairwise-construction-support-outer0-20260925/report.json`.

The existing counterfactual corpus could not safely supply independent groups
as originally encoded. Its v1 variant used a parent `example_id` as
`contrast_id`, while `construction_folds` joins by source-text SHA. All v1
variants also shared one construction ID, which would collapse unrelated
parent constructions if that lineage link were repaired in place. A new
`counterfactual_natural_source_v2` kind now joins on the parent's text SHA and
keeps the parent construction in the variant's construction ID. A regression
test verifies that each variant stays in its parent's fold without merging
distinct parent constructions. V1 rendering and its receipt stay unchanged;
the signed 36-example historical v1 feature bundle was reloaded successfully.
After that code checkpoint, the supported feature-acquisition sidecar loaded
the explicit 27B model, acquired all 36 v2 `lexical_mid_final_v1` records, and
closed its worker. The v2 manifest is
`f3cf4ea7734123834e479ac059bfdb53834de296cf1f5b74219d79dfd9d12679`
at `~/.aura/rlc-evidence/semantic-counterfactual-v2-20260925/features`.
Independent reload verified all 36 v2 examples and the historical 36-example
v1 bundle. The isolated acquisition state had no migration authority key, so
optional affective steering did not attach; that warning is not evidence of
an acquisition failure or of the live desktop's state. V2 is now available
as training input, but no selector has been fitted or qualified on it.
The attempted mixed-cohort training preflight refused it: the v2 worker's
`worker_source_sha256` is `8ef364036dc029b4998dffa06c61fde4852e2c91add76aa99660edb8da988012`,
whereas the older source cohorts bind
`53676c08c7bfd4348c48cfc56e369721ca3c75fb54f10c842ac2d8746d17a94a`.
The same 36 v1/v2 texts produced identical hidden arrays, token IDs, and
executable programs, but this comparison cannot establish compatibility for
the other cohorts. A source-matched reacquisition or an independently proved
representation equivalence is needed before a joint training receipt; the
identity gate was not relaxed.
The supported one-load reacquisition completed the other seven cohorts under
the v2 worker at
`~/.aura/rlc-evidence/semantic-source-reacquisition-v2-20260925/features`
(1,728 records, seven complete manifests, worker closed). The existing
`prepare_compositional_source_training` accepted those seven plus v2
counterfactual as one representation. Its 764 training and 500 validation
source-text ID digests exactly equal the original source-fit receipt; no test
rows enter the plan. A sequential, strict bundle reload compared every
reacquired source ID, token hash, and hidden-state hash with the seven old
cohorts: all 1,728 match. The 36 counterfactual v1/v2 texts, token IDs, and
hidden arrays also match. Thus this repairs provenance and fold lineage, but
does not add new learned signal. A new source fit was started and interrupted
before producing artifacts when its liblinear pointer fit reached about
19 GB resident memory under concurrent host work. No new candidate or quality
result is claimed. The next experiment needs genuinely independent source
constructions or a better search/selection mechanism, not a relabeled fit of
identical features.
The CPU-only eight-cohort comparison is reproducible with
`tools/compare_semantic_feature_cohorts.py`; its signed receipt is
`~/.aura/rlc-evidence/semantic-source-reacquisition-v2-20260925/feature-comparison.json`
(`4042a6776e459053592a6eafb0133f035a769f7cf955cf4e3e84357236e47698`).
It asserts measured feature equality only, not universal neural-function
equivalence or a serving entitlement.

## Source-order input identity

The original program-first corpus sometimes ordered `public_inputs` by
construction dependency rather than source occurrence. The live extractor
`semantic_public_character_inputs` orders top-level literals by source
position. When two inputs shared a value, the decoder could not recover the
corpus's private register permutation from the values: a correct selection
span became the wrong input index. A concrete role-binding example had
`(sequence, selector=3, initial=3)` in the corpus but presented the initial
literal before the sequence and selector in the source. The frozen decoder
found the selection and arithmetic spans, then attached them to swapped
equal-value registers.

`source_order_training_example` now rebinds the public inputs, input spans,
SSA references, and register-definition spans together. The old training
contract remains v1; source-order fitting is an explicit v2 opt-in. This is
label/representation correction, not new model weights or a phrase rule.
On every one of the 500 selected validation texts, the normalized values
matched the live character-literal extractor's source order.

The frozen 27B transducer was regraded twice over the same 500 development
identities with exact register-index scoring. Corpus-order gold gave 472/500;
live source-order gold gave 485/500, comprising 17 corrected rows and four
regressions. The signed, non-serving receipt is
`~/.aura/rlc-evidence/semantic-source-order-regrade-20260925/report.json`
(`f46dcf5bdc309eff9b909691d12a5543ff02a93ca49281cefacf6711e90ae784`).
No learning or fresh transfer occurred. Eleven residual misses and all four
regressions fall in cataphoric construction 5 or reserved-alias construction 5;
they include distinct operation-label and argument/alias-binding errors.
Source-order correction is necessary for live equivalence but does not close
G03 or qualify the unchanged model.

A new eight-cohort fit used the source-order labels with the original 764
training and 500 validation source identities. Its saved candidate is
`~/.aura/rlc-evidence/semantic-source-order-fit-20260925/candidate.json`
(`5475bb1bd125078796736c7a42ea4f846a5f48f747c03d2e77cf88422c90b0b5`).
The signed validation receipt is
`~/.aura/rlc-evidence/semantic-source-order-fit-20260925/validation.json`
(`3c444d7df154f49494ef68c74dd8e302dac356751ad1b18336eb6954cfc7e657`).
The new candidate scored 470/500 against old corpus-order labels and 487/500
against source-order labels. Against the frozen prior candidate's source-order
485/500 on those same validation identities, it gained three rows and lost
one. Thirteen rows remain incorrect. The candidate was not promoted or served.
The fit used about 19 GB of resident memory; the worker finished and closed.

The completed candidate predates a model-receipt repair: its report names the
source-order policy, but its model identity does not. Future v2 fits bind
`source_token_order_v1` into the model training receipt. The shared refit
loader now refuses any report/model policy mismatch and applies source-order
normalization before checking split identities. The original artifact keeps
its old identity. An explicit, non-serving migration verified its signed fit
plan, model coefficients, source cohort and validation receipt, then issued a
new model/report/candidate-report lineage under
`~/.aura/rlc-evidence/semantic-source-order-identity-rebind-20260925/`.
The new model receipt is
`a7976f7cdbe34395d50086a1a71e237edf646068e6d467bf8d0aff27d8f7cd08`;
the source report is `6d78190eae26c0ce316d010e56baad8d0daa06acfed1f8ff2e53fb829e6ca38c`
and the candidate report is
`09bb2a43515b4e58b131c4068b6783c5611c733dca5608fe456e74359ac41df0`.
Regrading all 500 validation identities after migration reproduced every
old candidate decision, 487/500 source-order and 470/500 corpus-order
(`validation.json`, receipt
`12da06180502e39a404e61ff8bd969688da68eb63785f0898b62382c71863348`).
This repairs artifact identity, not accuracy or serving qualification. The
bank and ranker consumers now verify signed candidate and source reports,
including the migrated parent lineage, before source-only acquisition.
The next experiment must address the remaining operation and alias-binding
failures and retain paired regression accounting.

The read-only replay in `tools/profile_semantic_source_order_misses.py` checked
the saved candidate, validation receipt and cohort identity, then decoded only
the 13 failed validation rows. Its signed profile is
`~/.aura/rlc-evidence/semantic-source-order-fit-20260925/miss-profile.json`
(`cd2b458357ab2328c21581185a1e2d0db1420843e2d7d40c20b7b7711c0502c4`).
Six rows select a different operation multiset, five keep the operations but
bind arguments differently, and two reverse operation order. The five
cataphoric misses are construction 5 and the eight reserved-alias misses are
also construction 5. This is a development-set diagnosis, not evidence that
any alternative selection policy will transfer. The common unresolved task is
binding a delayed or named value to the correct role in a dependency chain.

The same frozen candidate was replayed through the target-blind candidate
bank. At eight operation charts, four graphs per chart, three seconds per
solve and eight seconds per bank, the exact target was observed on 11 of the
13 failed rows. The two unobserved rows were replayed alone at 16 charts,
16 graphs per chart, five seconds per solve and 20 seconds per bank; both
targets were observed. Both banks reported incomplete search, so this is a
13/13 observed reach result under two declared bounds, not proof of complete
search. It isolates target-blind selection as the measured development failure
for these 13 rows. No selector was fitted or promoted. The signed receipts are
`~/.aura/rlc-evidence/semantic-source-order-fit-20260925/bank-profile.json`
(`eff5d71d2d169bd6781b7c8cff0d45a5ec6d43cf50223fab8c20fe8d20621a7a`)
and `bank-wide-two.json` in the same directory
(`c2860058bc8d98c4db7a99351fba3b9eba3baf3d115fceda6d308babbd0b3a58`).

## Source-order selector pilot

`tools/freeze_semantic_source_folds.py` replayed the signed, identity-rebound
source cohort into three construction/contrast-disjoint folds: 764 source
training rows in 53 independent groups, with no validation or test rows.
The receipt is
`~/.aura/rlc-evidence/semantic-source-order-identity-rebind-20260925/folds.json`
(`cd8c92f5aa8771aa1c5c56387a047d38d0a31c7e82eead873bce7e8a038adccc`).
The source-bank pilot sampled one row per independent construction group,
not by validation failure. All 53/53 had an observed equivalent candidate
and all 53 incumbents selected one. Execution and emission remained
unmeasured by this bank. Its signed report is
`~/.aura/rlc-evidence/semantic-source-order-bank-pilot-20260925/report.json`
(`7e2f7285ba921a472eb93a6b7bc847778abf875a6e14a7d417fa84fa34537ad7`).
A one-epoch, argument-evidence selector fitted on 38 of those rows selected
36/38 correctly on its training probe and 12/15 on the disjoint held fold,
versus 15/15 for the incumbent. The three regressions and zero gains reject
this pilot; it is not a validation or transfer result. The signed fold report
is `~/.aura/rlc-evidence/semantic-source-order-ranker-pilot-20260925/fold-0.json`.
Its receipt is `4efa3d3e8ff8e98f35d42e75ece105e6c3cfaa4ae3a049d8181adba067689a46`.
The pilot's absence of hard incumbent errors is evidence against spending a
full acquisition run on the same sampling rule. Source-only hard examples
must be obtained without using the 13 development labels as training data.
