# Joint semantic refinement workplan

<!-- reader-navigation:start -->
> **In plain English:** The reasoning programmes investigate whether extra computation or learned internal representations improve answers on tasks not used to choose the method.
>
> **Why it matters:** A slower or more elaborate process is useful only if it produces a measurable benefit that survives relevant checks.
>
> **In everyday use:** On a difficult constraint problem, a better internal method should increase correct answers or expose uncertainty rather than only generate longer prose.
>
> [Example and metaphor](PLAIN_ENGLISH_GUIDE.md#reasoning) · [Variables and practical meanings](VARIABLES_EXPLAINED.md#reasoning) · [Demos and related work](DEMOS_AND_RELATED_WORK.md#reasoning) · [Document map](DOCUMENTATION_INDEX.md)
> [Glossary](GLOSSARY.md) · [Formulas with worked examples](FORMULAS.md)
<!-- reader-navigation:end -->

This work implements the six mechanisms requested on 2026-09-15. It extends
the existing semantic program substrate, procedure registry and proof kernel.
It does not replace or close the master G03-G12 obligations.

The [retained-constraint design](G_SEMANTIC_CONSTRAINT_DESIGN.md) derives the
bounded correctness conditions and separates numerical fitting, capacity
expansion and fresh transfer. It adds no new master-ledger identifiers.

## Current promotion path, 2026-09-28

One v7 candidate follows the stages below in order. The model-active fit
finished and its source-only residual was independently replayed. The first
three-request fitted reference arm generated three exact procedures and
answers; its baseline arm stopped without a row under the original time bound.
A source-matched one-request grouped replay matched the individual scorer.
The [recovery record](evidence/G03_V7_CAUSAL_GROUP_RECOVERY_2026-09-29.md)
separates those observations from the fresh grouped micro campaign now running.

| Stage | Current state | Required result before advancing |
| --- | --- | --- |
| Fit and independent replay | Complete for source-only selection | Complete source-only schedule, exact model/source identities, checked weights, and replayed calibration selection. No held answers used to select a checkpoint. |
| Three unseen-construction requests | Fitted arm 3/3; controls incomplete | The [frozen target-blind protocol](evidence/G04_V7_TARGET_BLIND_MICRO_PROTOCOL_2026-09-28.md), with fitted, base, and fitted source-erasure arms. Report all three interpretations and controls individually. |
| Nine relation controls | Not run | The [mechanism micro-probe](evidence/G04_V7_RELATION_MECHANISM_MICRO_PROTOCOL_2026-09-28.md) reuses the three reference results and tests paraphrase invariance, role reversal, and dependency changes. All nine must recover the intended graph and public value, with no forced completion. |
| Six retained development requests | Not run | The [frozen source-held protocol](evidence/G03_V7_MICRO_PROBE_PROTOCOL_2026-09-28.md), with the same three arms. Report baseline successes and losses, not only the fitted total. |
| Full development bank | Not run | All three micro cohorts must be entirely exact with correct public values, zero lost baseline-exact procedures, no forced completions, complete independent verification, and at least one new exact procedure showing source dependence. Then measure all 500 development requests under frozen matched budgets. No micro outcome closes G03. |
| Fresh transfer and controls | Not run | Freeze the development-selected candidate; satisfy the original G04-G08 construction, vocabulary, depth, family, causal-control, power, contamination, and verification requirements. |
| Public and live qualification | Not run for this candidate | Satisfy G05 and current-model G10-G11. Prove ordinary public answers, materialization or fusion, rollback, and eligible live use. Historical activation cannot qualify v7. |
| Broad and frontier measurement | Not run for this candidate | Satisfy G09 and G12 on independent broad tasks and named current baselines with fair tool/resource accounting. Bounded integer-procedure success is not this claim. |

The micro launch bar is stricter than the protocols' criterion for reporting
one bounded gain. A single exact gain remains useful evidence even when the
cohort is not ready for a full run. It must not be called full cohort success.
The [budget frontier](evidence/G03_NATIVE_BUDGET_FRONTIER_2026-09-29.md)
shows the fitted three-case reference result survives a retrospective 16-node
replay, with 255 rather than 1,100 scored alternatives. Future candidates may
use a separately frozen 16-node matched screen before the full stage. A screen
failure saves a long run; a screen pass still owes every full-budget and held
control requirement above.
The [prospective screen runner](evidence/G03_PROSPECTIVE_REFERENCE_SCREEN_2026-09-29.md)
now freezes those three arms and verifies their receipts. It has not been run
on a new candidate. A failed screen is a development stop decision, not an
inference that the candidate cannot succeed at 256 nodes.
The [cached grouped scorer](evidence/G03_CACHED_GROUPED_SEARCH_2026-09-29.md)
is a separate opt-in compute path. Small-model checks establish bounded score
agreement; it still needs real-27B ranking and timing checks before a new
candidate can use it.
The [six-scale oracle audit](evidence/G03_RESIDUAL_SCALE_ORACLE_CEILING_2026-09-29.md)
bounds the existing source-calibration evidence: even a target-aware scale
choice per decision reaches only 126/185 exact teacher paths. The next fit
must improve reference-conditioned ranking, not only scale selection.
The verified baseline arm completed at one exact procedure and answer among
three. Paired retrospective replay has fitted 3/3 versus base 0/3 at 16 nodes
and fitted 3/3 versus base 1/3 at 256. The source-erasure arm is still needed
before the reference stage can pass.

Stage receipts are immutable and bound to the candidate, source population,
code, model, and budget. For the same identities, a passed stage is reused,
not decoded again. A later failure stays at the later stage. A repair that
changes an identity creates a new candidate and requires only the affected
acceptance and regression checks; the earlier evidence remains valid for the
earlier candidate and must not be relabeled as proof of the changed one.

`tools/adjudicate_semantic_native_micro_stages.py` recomputes the three-arm
comparisons from their durable files, checks the unchanged fit verification,
and records the first incomplete or failed stage. Supply a `--reference-root`
containing `fitted`, `base`, and `erasure` directories, then add
`--controls-directory` and `--retained-root` as those stages finish. Each
snapshot uses a new immutable `--output` path. Passed references are replayed
on CPU, not decoded again. The full-development readiness field requires all
three exact cohorts and source-dependent gain; it grants no serving or broad
reasoning authority.

`tools/run_semantic_native_micro_stages.py` runs these same checks in one
sequence through the existing detached supervisor's exact-command broker.
Its `--policy-output` mode freezes all seven arms' commands and their
budgets without loading a model. After independent fit verification, the
supervised controller freezes every evaluator plan, runs one arm at a time,
verifies its files, and advances at each passed acceptance. A failed later
stage preserves the earlier receipts. Completed arms are reverified on CPU
on continuation, never decoded again. Partial decodes require a new declared
attempt. The controller does not start the 500-request bank or grant G-ledger
closure. Every native grammar arm now requires exclusive, non-evicting lane
ownership, independent of its report schema version.
With `--wait-fit-supervisor`, `--fit-bank`, and `--fit-parent`, the controller
waits for that existing trainer's verified terminal receipt, runs independent
fit verification in a brokered CPU process, then starts the same micro stages.
It never restarts the trainer or loads a model while its process remains alive.

The [complete development window contract](evidence/G03_DEVELOPMENT_WINDOW_EXECUTION_2026-09-28.md)
adds `--source-offset` for the later 500-request stage. Each window binds the
whole ordered population and retains the same per-request scorer and search.
`tools/adjudicate_semantic_native_development.py` replays all matched windows
and refuses missing requests or changed budgets. Completed windows can be
retained while the remaining fixed windows run. A complete micro result is
still required before that broader stage; the window implementation supplies
no generated v7 outcome or G03 closure.

For each target-blind miss, retain one of these causal diagnoses:

1. The intended graph was absent from the generated proposals: interpretation
   or bounded search reach failed. A ranking change cannot repair an absent graph.
2. The intended graph was generated but lost whole-graph selection: ranking
   failed. Preserve every proposal, score and source-token receipt.
3. The selected graph was exact but its answer or public emission was wrong:
   execution or output integration failed. Do not retrain interpretation to
   hide this boundary.
4. A gain survived source erasure unchanged: exactness was measured, but that
   observation does not establish source-dependent interpretation.

Correct primitive semantics and semantics-preserving composition support an
inductive executor proof. Three correctly interpreted requests prove those
three interpretations; they do not establish a universal learned compiler.
The larger transfer test therefore remains a substantive test of coverage and
generalization, not merely a longer repetition of the same computation.

Search receipts now retain scores for discarded branches and source-token
receipts for whole-graph selection. The independent verifier reconstructs
search and selection from those saved scores. It checks numerical
self-consistency and source binding; it does not independently recompute the
model's logits. Use `tools/compare_semantic_native_grammar_fit.py` with
`--erasure-directory` for one matched three-arm adjudication. Its bounded
micro-gain verdict grants no serving, broad-gain or general-transfer authority.

## Implementation and acceptance

- [x] S01 Capacity certificates. Convert frozen score comparisons into linear
  constraints. Reuse the exact arithmetic kernel to check witnesses. Separate
  proved infeasibility, verified feasible weights and unresolved numerical
  search. Retain the comparison identities and assumptions with the result.
  [Checked evidence](evidence/G03_CAPACITY_AND_SEARCH_2026-09-15.md).
- [x] S02 Complete bounded operation search. Retain every interpretation
  inside the declared candidate grammar, with lazy search and sound bounds.
  Compare to exhaustive enumeration on small cases. Report an interrupted
  search as incomplete, never as proof that no interpretation exists.
  [Implementation and exhaustive checks](evidence/G03_CAPACITY_AND_SEARCH_2026-09-15.md).
  Complete only for the declared operation grammar; S03/S07 remain open.
- [ ] S03 Joint semantic learning. Train operation, reference, scope and
  dependency decisions against complete incorrect interpretations. Preserve
  equivalent correct programs and use the same candidate construction at
  training and inference. A three-scalar calibration is not this task.
  [Witnessed graph negatives and algebraic positives](evidence/G03_SEMANTIC_COUNTEREXAMPLES_2026-09-15.md)
  are implemented; joint neural-head learning remains open.
  [Graph-supervised relation tissue](evidence/G03_GRAPH_RELATION_CANARY_2026-09-15.md)
  repairs the measured relation conflict and selects equivalent programs on
  four training canaries. Full-cohort measurement and operation learning remain.
  [Completed full-cohort comparison](evidence/G03_GRAPH_RELATION_RESULT_2026-09-15.md)
  improves exact programs from 436 to 465 and equivalent programs from 456 to
  471 out of 500, but one regression prevents promotion.
  [Joint operation/relation trainer](evidence/G03_JOINT_GRAPH_TRAINER_2026-09-15.md)
  now consumes witnessed runtime-selected errors and differentiates both
  shipped heads under one graph objective. Its full-cohort result is pending.
  [Source-operation retention](evidence/G03_SOURCE_OPERATION_RETENTION_2026-09-15.md)
  now preserves all source operation labels during joint fitting. The
  contrast-only run increased source errors across rounds; paired evaluation
  and measurement of the retention candidate remain separate obligations.
  [Decode-local feature reuse](evidence/G03_CHART_FEATURE_REUSE_2026-09-15.md)
  removes repeated span pooling and definition-pointer scoring across charts.
  The contrast-only trial was stopped with only 216/500 candidate rows; it
  supplies no complete paired verdict.
  [Completed retention comparison](evidence/G03_SOURCE_RETENTION_RESULT_2026-09-15.md)
  measured all 500 validation rows: 451 exact with 31 regressions and 454
  equivalent with 30 regressions. The incumbent remains selected.
  Retained-constraint fitting is now implemented as an opt-in path, including
  wrong binding competitors from already-correct source cases and every source
  operation-label competitor. Unit checks cover retention under conflicting
  gradients and exported-dtype margin replay. Full-source fitting and the
  unchanged development comparison remain required before any promotion.
- [ ] S04 Counterexample refinement. Repeatedly find new wrong complete
  programs, retain their distinguishing evidence and refit against the whole
  retained set. Detect unsupported or inseparable distinctions and request
  representation expansion rather than declaring optimizer convergence done.
  The relation refit retains witnessed pairs across rounds; full-cohort
  iterative qualification is still required.
- [ ] S05 Counterfactual training. Generate meaning-preserving renamings,
  reorderings and recompositions plus minimal meaning-changing contrasts.
  Validate their IR independently. Freeze fresh evaluation families outside
  the generation/training inventory.
  [Counterfactual corpus implementation](evidence/G03_COUNTERFACTUAL_CORPUS_2026-09-15.md)
  passes independent execution and the existing feature-bundle round trip.
  Model feature acquisition, training and fresh transfer remain unmeasured.
  Update: 36/36 counterfactual examples were subsequently materialized on the
  resident 27B into `~/.aura/rlc-evidence/semantic-counterfactual-features-20260915/features`.
  The standard bundle reader accepts them. Mixing with the older source banks
  is still refused because worker-source identity and parameter-count basis
  changed; no compatibility exception or training promotion was granted.
  [One-load reacquisition](evidence/G03_SINGLE_LOAD_REACQUISITION_2026-09-15.md)
  now rebuilds the exact cohorts from their manifests and uses one existing
  worker lifecycle. CPU checks pass; the new feature bank is not yet acquired.
  Update: reacquisition completed all 1,764 examples in eight cohorts under
  `~/.aura/rlc-evidence/semantic-source-reacquisition-20260915/features`.
  Standard bundle reconstruction and representation admission pass: 764 train,
  500 validation, 500 test. Validation identity is unchanged
  (`0b7112a312440fbfc5dcad54be3fecb3a13f89058fc0a3d78907eee889caf259`).
  The new source-fit path excludes test rows and allows training-only cohorts.
  Acquisition and admission are not training or transfer success.
  [Fresh source fit](evidence/G03_FRESH_SOURCE_FIT_2026-09-15.md) subsequently
  completed training and all 500 validation cases: 472 exact and 472 equivalent.
  No held-out test examples were used and no serving promotion was granted.
- [ ] S06 Reusable verified abstractions. Store parameterized procedures with
  scope, dependencies and evidence in the existing registry. Demonstrate
  reuse under new names and values without answer lookup or task-label access.
  [Learned-procedure measurement](evidence/G03_LEARNED_PROCEDURE_REUSE_2026-09-15.md)
  now connects decoded programs to the existing registry and measures new-value
  execution separately from interpretation correctness. Full-cohort and
  new-wording transfer remain unmeasured.
  [Full-cohort domain diagnosis](evidence/G03_PROCEDURE_DOMAIN_CONTRACT_2026-09-15.md)
  completes 500 interpretations and exposes an evaluator domain-accounting
  defect plus three empty-sequence lowering mismatches. Those are repaired;
  the corrected full measurement and new-wording transfer remain required.
  [Corrected full measurement](evidence/G03_REUSABLE_PROCEDURE_RESULT_2026-09-15.md)
  preserves selected-program semantics on 15,860 fresh-value probes with no
  lowering mismatch or execution error. Selection remains 471/500 proved
  equivalent; new-wording transfer and runtime qualification are still open.
- [ ] S07 Integrated measurement. Run the unchanged 500-row development
  comparison, freeze a selected candidate, publish prospective fresh transfer
  and matched-arm plans, then perform G04-G08 measurement and verification.
- [ ] S08 Runtime and broad transfer. Qualify current-model materialization,
  validate live use and evaluate broad tasks and named frontier comparisons
  under the original G09-G12 requirements.

## Correctness boundary

For a bounded, unambiguous task, correct selection follows if the correct
semantic class is reachable, its best score exceeds every incorrect class,
search is complete, and execution preserves the selected meaning. A proof
about this finite score model does not prove language understanding, future
generalization, or broad frontier performance.

Training annotations and diagnostic targets remain outside runtime inputs.
The independent verifier must distinguish interpreting the requested task
correctly from merely executing a well-typed but incorrect program.
