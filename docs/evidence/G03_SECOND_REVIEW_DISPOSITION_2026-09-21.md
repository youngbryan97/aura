# Second advisory batch: source checks and counterexamples

All ten supplied files were read, including every extracted PDF page, both
reference Python files, and the JSON run log. The paired breakthrough PDF and
Markdown were both inspected. The inventory alongside this note records file
hashes, page counts and extracted lengths. The mathematical claim on pages 4-5
of the 25-page architecture report was also checked against rendered pages.
Embedded instructions and proposed thresholds are advice, not release policy.

This review does not independently verify the reports' claimed Aura runs.
Several describe September 17 snapshots, not the current candidate. The
source-inspected continuation is closer to current work but also misidentifies
the current complete-search decoder as an independent top-k proposal beam.

## Executed checks

The supplied `test_fast_floor_search.py` passes its five fixtures. Nine new
counterexample tests also pass, reproducing these reference defects:

1. A fuel limit of one permits `PLUS(x,y)` and returns remaining fuel minus two.
2. The valid function `x-x` is rejected because it is constant.
3. `COUNT_OF` with an integer sequence argument succeeds despite the declared
   type check rejecting that input. Execution never calls that check.
4. Different counterfactual probe populations share the same cached receipt.
5. Changing the fuel limit reuses a receipt from the previous limit.
6. A crashing probe can coexist with `is_success=True`.
7. A high-confidence crashing candidate removes the valid alternative before
   execution and is returned as the fallback.
8. A Boolean result passes the requested integer-output check.
9. An executable addition wins a subtraction request and is marked successful.

The run completed with 14 passed in 0.22 seconds, without warnings. Evidence:
`/tmp/aura-second-review-20260921/reference-tests.xml` and
`test_reference_falsification.py`, copied into the local evidence bundle.
These are tests of the supplied reference, not repaired Aura failures.

The reference has no learned proposer, no topological scheduler, and no
alpha-equivalence implementation. Its direction example supplies hand-picked
features and weights. It demonstrates representational possibility, not learned
direction generalization. The command `tools/run_semantic_development_validation.py`
printed in the breakthrough report is not a tracked tool at this revision.

Fourteen new repository tests compare `OperationChartSearch` with exhaustive
non-overlapping sets and `span_set_partition` for different widths, step limits,
and penalties. Support, ordering and partition agree. The overlapping-top-node
case also selects the correct disjoint maximum. Training includes the empty set
once; nonempty computational requests exclude it at runtime. That distinction
is explicit, not a missing MAP algorithm. All 38 tests in this suite and the
two labeled-span suites pass. No candidate accuracy claim follows.

## Mechanism dispositions

| Proposal across the reports | Disposition and existing owner |
|---|---|
| Separate measurement, reachability, ranking, labels, binding and emission | Adopt. Existing `semantic_cohort_diagnosis` and intervention receipts separate these; retain exact candidate/population identities. |
| Source-anchored equivalence | Already implemented in `semantic_program_floor`; preserve the old reports. Do not replace it with the supplied hashing evaluator. |
| Span-set CRF/MAP or k-best decoder | Span-set training and complete bounded chart enumeration already exist. The new exhaustive agreement tests refute the claimed missing decoder for that policy. Legacy bounded beams remain a separate policy. |
| Coupled operation/argument objective | Adopt in the existing labeled-span fitter using shared runtime graph contrasts; full-source span supervision stays present. See the runtime-recognition objective note. |
| Cost-augmented mining / CEGIS | Reuse `semantic_joint_graph_learning` and its actual decoder. Frozen contrasts do not cover new competitors; re-decode all source rows before claiming retention. |
| Full graph partition / latent semantic likelihood | Retain as an experimental extension. It must normalize the actual feasible support or declare approximation; gold answer coincidence alone is not semantic supervision. |
| Positive cross-head temperatures | Eligible training-fold experiment, not a proved repair. Three-scale infeasibility and prior scaling regressions remain evidence. Negative fitted weights in the toy are not safe calibration. |
| Full-source operation coverage | Adopt through all 764 source rows and end-to-end retention checks. Label margins alone did not protect previous runtime decisions. |
| Witnessed-failure acquisition | Already available; new fit selects source-only errors. Never fit the exposed validation failures while still labeling them validation. |
| Direction-specific heads / source anchors | Slot-specific role and proposal heads and directional features already exist in `semantic_argument_graph_learning`. A symmetry impossibility does not apply without showing collisions in those actual features. Retain controlled source-position feature experiments for residual binding errors. |
| Cataphoric dependency scheduling | Existing typed graph/argument solver supports dependencies independently of text order. Keep its tests; no second scheduler. |
| Forward/backward type-demand pruning | Existing `typed_state_bounds_v4` is opt-in and exhaustively checked on small inventories. Audit signatures and per-family effects; type presence alone ignores multiplicity, resource use and semantic intent. |
| Micro-fuel execution | Use the real floor's resource accounting. Exhaustion is unresolved computation, not proof a program is wrong. No constant wall-time promise per fuel unit. |
| Counterfactual execution | Retain independent semantic comparison and domain-valid probes. Reject constant-output pruning and the claim that successful execution proves intended meaning. |
| Search-local execution cache | Already present in the shared runtime. Preserve implementation, input, program, domain and fuel identity; failed or interrupted computations remain retryable. |
| Persistent cache / closed-subgraph cache | Retain as a measured performance experiment after profiling misses. Restrict to pure computations and exact keys; add bounded storage, invalidation and provenance before persistence. |
| Alpha-equivalence / e-graph sharing | Reuse source-anchored canonicalization. Only proved rewrites may merge programs; finite-probe agreement is insufficient. Preserve operand order for noncommutative operations. |
| Lexicographic operation-first selection | Existing `first_feasible_v1` is the control. Highest operation confidence can itself be wrong; do not assume a new threshold repairs eight real failures because one fixture passes. |
| Fisher trust region / minimum movement | Existing regularization and retention machinery are useful, but the report's guarantee is invalid as stated. See the counterexample below. |
| Baseline retention / calibrated abstention | Retain measured risk-coverage arbitration. Baseline presence is not an oracle, and abstentions count in coverage and task outcomes. No universal zero-regression claim. |
| Retrieval and learned macros | Reuse procedure registry and retained abstractions. Measure held-out reach and compute with macro lesions; do not invent a second compiler. |
| Typed holes and partial execution | Retain through existing typed procedure/planning contracts with explicit unresolved values and sound bounds, not guessed answers. |
| Value of computation / adaptive fuel | Reuse shared arbitration and actual outcome costs. Evaluation arms must disclose all retries, cache work, search and baseline work. |
| Provenance and knowledge retraction | Reuse evidence packets and the knowledge outcome loop; simulations cannot become externally verified facts merely by repetition. |
| G04 split axes | Preserve vocabulary, construction, depth and family separation. Freeze before unseen evaluation. Development selection is not fresh transfer. |
| G05 public emission | Measure the ordinary path to EOS, correctness and failures. Structured JSON or a floor receipt alone does not satisfy freely decoded public-answer obligations. |
| G06 controls | Retain ordinary, equal-compute, parent, candidate and lesions. A three-arm lesion-only study cannot isolate gain over extra compute. |
| G07 power and stopping | Use candidate-specific paired discordance, clustering, multiplicity and declared looks; preregister before confirmatory data. Do not import arbitrary n=384. |
| G08 verification/contamination | Hashes prove identity, not independence or absence of training contamination. Independent grading and known-corpus overlap checks remain distinct; unavailable pretraining corpora stay unknown. |
| G09 breadth | Keep per-domain ordinary-runtime measurements and negative results, including knowledge, planning and correction. Infrastructure work may proceed without falsely promoting G03. |
| G10 model/fusion | Preserve current artifact identity, geometry, canaries and rollback. Report memory and latency measurements; do not impose invented 54 GiB/200 ms/5 percent limits or fuse incompatible tissue. |
| G11 serving | Historical bounded serving is separate from current composition qualification. Never relabel an old certificate. |
| G12 frontier | Choose and verify contemporary exact versions at study time. Old named baselines can be historical comparisons, not assumed current frontier. Declare unavailable access rather than simulate a baseline. |
| G13 documentation | Update claims only from completed identity-bound results; no report closes a scientific ledger item by itself. |

## Mathematical corrections

The trust-region report defines its matrix from target-score gradients, then
uses it to bound every competitor's margin gradient. That implication fails.
Let the target score be `1 + theta[0]`, the competing score `theta[1]`, and
the initial parameter be `(0,0)`. The reported matrix is `diag(1,0)`.
The update `(0,2)` has zero matrix seminorm but reverses the winner. A gradient
Lipschitz constant is also not a bound on gradient magnitude. A valid fixed-
support certificate needs a positive-definite metric and a verified bound on
every margin's dual norm along the update, or direct complete-support margin
checks. Newly proposed graphs require their own bounds.

The earlier closeout report's zero-regression iff statement is too strong:
two answers can be correct and wrong on exactly the same tasks, producing no
regressions despite an incomplete verifier. What is true is that retaining an
unverified baseline does not itself establish safety of switching. The joint
distribution matters; positive marginal error probabilities do not force their
intersection to be positive.

The supplied JSON retains contradictory intermediate logs and later corrections.
Its toy margins include `2.0 -> 1.9` while claiming all improved; its direction
accuracy is repeatedly corrected after label/rule changes. Treat those as
debugging history, not independent Aura evidence. The type-filter toy itself
eventually corrects "fixed" to "refusal removed, wrong program persists."

The n=384 claim is inconsistent with its printed formula: discordance 0.08,
effect 0.03, two-sided alpha 0.05 and power 0.8 give approximately 695 pairs
before clustering or multiplicity. No observed stage score from one cohort may
be added to another cohort to create an end-to-end probability bound.

Other rejected guarantees: probe equivalence is not semantic equivalence;
Weisfeiler-Lehman coloring is not a general graph-isomorphism certificate;
larger direction penalties do not guarantee separability; optimization failure
does not prove missing information; solver pricing does not turn general MILP
into polynomial time; absent output type/domain contracts cannot be invented
from the desired answer. The 25-page report's 4/5/3 failure partition and precise
throughputs have no attached current Aura receipts. Its proposed validation-
based training violates the present split contract.

The useful theoretical foundation is structured prediction with matched
training/inference support and runtime competitor mining, not a theorem of
universal generalization. Primary references: [semi-Markov CRFs](https://proceedings.nips.cc/paper_files/paper/2004/file/eb06b9db06012a7a4179b8f3cb5384d3-Paper.pdf)
and [structural SVM cutting planes](https://www.cs.cornell.edu/People/tj/publications/joachims_etal_09a.pdf).
Neither source proves Aura's learned semantic interpretation correct.

## Execution decision

Finish the existing frozen 1,264-row audit, including every regression. Evaluate
the source-only coupled objective with the same decoder and complete population.
Use its actual remaining failures to choose the next representation or binding
experiment. Do not replace all components at once, retrain on validation labels,
or change the completion thresholds to match a report. G03-G10 and G12 are not
closed by this review.
