# Verified reasoning reference review

## Inputs actually inspected

Read all 26 PDF pages, including the equations, proposed interfaces, ledger
criteria and references. Rendered pages 7 and 10 to inspect the architecture
table and proof statements. Read all four archive members, then extracted and
ran the reference in an isolated scratch directory. No supplied code was
installed into Aura or given runtime authority.

- `Aura_G_Ledger_Reasoning_Gain_General_Transfer_Report.pdf`, SHA-256
  `0d09c8a4d3d06d59598eff30c1ad5fd8d4b3fce195d415c443b8d80a7856f100`
- `aura_verified_reasoning_reference_clean.zip`, SHA-256
  `06c981b3e93fd16950aab14238f8ab877306fb5f02ec882b28b6cd7f8dd9d974`

The reference's 22 tests passed locally. Six additional executable falsifiers
reproduced gaps that its tests did not cover:

1. Missing execution and emission observations can produce `SUCCESS`.
2. A certificate for an unrelated scope can replace a correct baseline.
3. A declared child type can bypass its parent's value predicate.
4. A boolean register reference is accepted as an integer index.
5. A string such as `incorrect` is counted as a successful paired outcome.
6. Nonfinite current utility can return a nonfinite computation value.

The additional tests are retained at
`/tmp/aura-reviewed-reference-20260920/aura_design_test/test_additional_falsifiers.py`.
They reproduce reference defects; their passing is not evidence of a repaired
Aura capability. The report's sound-verifier and finite-grammar theorems are
conditional statements. Its standalone implementation does not establish
those premises for Aura or arbitrary language.

## Disposition of the recommendations

| Report item | Aura owner and action | Status and acceptance |
| --- | --- | --- |
| A: task-level roles and provenance | Extend the existing language work contract and procedure/evidence currency; preserve the exact leaf IR. A role label alone is not grounding. | Open: ordinary-language role inference, source binding, lowerer and goal-verifier agreement need causal tests. No second language substrate. |
| B: exact semantic leaf IR | Keep `semantic_program_ir.py` and the universal-floor compiler exact. | Retained. Do not make all numbers operands or broaden types to admit unsupported meanings. |
| C-D: candidate bank and decoding API | `semantic_candidate_bank.py`, `CompositionalSemanticProgramTransducer.decode_candidates`; reuse deployed operation and argument builders. | Implemented; bounded, target-free generation retains the incumbent and reports separate operation/argument coverage. Tests verify unchanged ordinary selection. |
| E: failure attribution | `semantic_failure_diagnosis.py`; compare the fixed bank with source evidence afterward. | Implemented. Unknown equivalence prevents an absence claim; unmeasured execution/emission cannot become success. Real archived-feature replay remains a separate result. |
| F: closed proof typing | Extend the shared procedure structural contracts, with a closed evidence mode. Preserve nominal legacy labels. | Open. Require subtype/value consistency and reject unknown proof types; do not import the reference registry unchanged. |
| G: retained counterexamples | Existing `semantic_runtime_graph_retention`, `semantic_graph_constraints`, joint graph training and checkpoint stores. | Adopted through iterative re-mining and broad source retention. Training reached 32/32; validation did not improve. No CEGIS convergence claim for continuous fitting. |
| H: external-utility computation value | Reuse latent-cortex action calibration and `OutcomeLedger`; architecture-level ranking motion remains heuristic. | Open: ordinary-route task outcomes must reach the shared compute decision. Self-consistency is not external utility. |
| I: public answer binding | Existing G05/G06 public-channel receipts and G08 paired measurement. | Required; free decode, parsed result, refusal/truncation and fallback provenance remain separate. No deterministic serializer counted as free generation. |
| J: ordinary-route broad runner | Existing G09 ordinary evaluation path, G07 frozen campaign and G08 replay. | Open: execute a candidate-bound cross-domain campaign with identical scoring and explicit resource access. |
| K: promotion evidence classes | Existing source/model/geometry qualification, independent verification and serving manifests. | Retained. Unit correctness, scoped proof, empirical gain and live service cannot substitute for each other. |
| Baseline-preserving replacement | Existing arbitration/evidence owners; bind verifier scope, task, candidate, specification and implementation. | Open where independent verification exists. Reject importing an unbound boolean certificate as a safety theorem. Unknown candidates do not receive correctness authority. |
| Library growth and reusable abstractions | Existing `ProcedureRegistry`, alternative planning, outcome credit and universal floor. | Open: retained semantic checks, fresh benefit, measured match/run cost and causal lesion. No second procedure store. |
| G04 four-axis transfer | Existing bound source inventory and fresh construction/vocabulary/depth/family exclusions. | Required before transfer claims. Exposed diagnostic rows cannot become fresh tasks by relabeling. |
| G06 fair controls and lesions | Ordinary model, equal-compute alternatives, shape-matched content lesions and retry accounting. | Required. Coalition attribution supplements paired outcomes; it does not replace them. |
| G07-G09 power and broad criterion | Existing paired prospective power/preregistration and independent verifier. | Candidate-specific plan remains open. Do not weaken the criterion after outcomes or pool away a regressing domain. |
| G10-G11 materialization | Bind the exact model, compiler, steering, registry and runtime identities that earned evidence. | Requalify changed identity; earlier live evidence does not transfer automatically. |
| G12 named frontier comparisons | Freeze current external baseline identities and their tool/compute access. | Open. Model names and rankings in an advisory report are not verified market evidence. |
| MDL, active contrasts, adaptation and multi-step VOC | Reuse existing abstraction and outcome learners; add only missing causal feedback edges. | Research candidates, not results. Each requires a small falsifiable intervention before a full campaign. |

The completed policy control is recorded in
[G03 policy and coefficient controls](G03_POLICY_COEFFICIENT_CONTROLS_2026-09-20.md).
The [candidate-bank replay](G03_CANDIDATE_BANK_DIAGNOSIS_2026-09-20.md)
finds correct alternatives in all eight additional joint-selection failures;
an expanded replay also reaches the correct program for the shared error.
It shows why a mathematically optimal search under a miscalibrated score need
not improve correctness: changing the policy alone reduced 99/100 to 91/100.

This review does not close G03-G12, assert a universal world model, or promise
perfect scores. It turns the supplied direction into implementation and
measurement obligations while preserving every existing ledger requirement.
