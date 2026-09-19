# Operation boundary learning, 2026-09-18

The complete-graph trainer could update operation labels, reference relations
and argument heads, but not the operation pointer. An incorrectly selected
span could retain a large fixed score while those other heads changed.

The opt-in `learn_operation_pointer` path now differentiates the pointer score
inside the existing retained-constraint objective. Its feature is the shipped
concatenation of start state, end state and their diagonal interaction. The
exported weights and bias return to the same runtime pointer; there is no
second decoder or source-text lookup. Source-label retention still applies.
Background-classifier candidates are refused because their graph score needs
a different derivative contract.

## Checks

The focused suites pass 74 tests in the earlier pointer-only checkpoint. The
current combined boundary/binding subset passes 56 tests. They cover score
replay, finite differences, batched gradients, parameter export, source-only
fitting and serialization. An induced boundary deficit is solved through the
new pointer parameters without changing input grounding. Smoke passes 164
tests with one skipped; lint, compile, governance, layering, writing and
doc-drift pass.

## Small diagnostic

The first trial scans 16 source-training examples, selects four, retains all
764 source examples' operation labels, and measures eight validation examples.
It examines four runtime operation charts per selected training example.

- Training: 4/4 equivalent before and after.
- Validation: 6/8 equivalent before and after.
- All 8,616 retained comparisons already satisfy the required margin.
- The optimizer accepts zero steps. This is an unchanged model, not a gain.
- Operation-search coverage is incomplete. No model is exported or promoted.

Receipt: `~/.aura/rlc-evidence/semantic-operation-pointer-trial-20260918.json`,
`2431f4c357338510a6d311409f95ace4dddd6e3d9af281ac891984d3cc6ff68f`.

G03 remains open. A wider source-only scan is needed to test the new learning
path on actual errors before a full development comparison.

## Wider scan and selected-error repair

A 64-example training scan found one incorrect interpretation. The bounded
four-chart miner did not retain that interpretation: it lies outside the
first four operation charts but wins after argument scoring. All sampled
comparisons remained positive, so the diagnostic again made no updates.
Receipt: `semantic-operation-pointer-trial64-20260918.json`,
`a03852b3f0b3cd623ebc4a87ff6b76b76f415c1b495f9d8dd9104befb4db29d2`.

The small-trial path now always retains the actual selected error through
the same contrast miner used by the full trainer. It also reports a missing
or inconsistent selected-graph replay. Twenty-seven focused tests pass,
including selected errors outside the chart allowance. Repeated smoke
passes 164 tests with one skipped.

The corrected trial retains 8,617 comparisons and accepts one update:

- Training: 3/4 to 4/4 equivalent.
- Validation: 6/8 to 6/8, one gain and one regression.
- One already-incorrect validation case becomes a decode refusal.
- No retained positive comparison regresses, but fresh decoding does.

Receipt: `semantic-operation-pointer-selected-20260918.json`,
`d8c0681d6085baefd44ed4fdf5b97058a57a141abe227a5c712906249f2a2393`.
The candidate is rejected. Protecting recorded comparisons does not establish
preservation of every correct interpretation, particularly with only three
correct complete-graph training controls. The next diagnostic expands those
controls within the source-training pool; it does not fit validation targets.

## Full-retention and source-binding controls

The trial implementation now mirrors the full retained-constraint trainer in
two ways. Operation-boundary constraints use the complete source-retention
cohort rather than only the small witnessed-error cohort, and each selected
training item contributes a source binding constraint when the argument heads
are trainable. These controls protect the source operation boundaries and the
source argument/relation assignment while runtime counterexamples add new
evidence. They do not authorize validation fitting or serving.

The 8-row targeted trial used a 64-row source-only acquisition pool, selected
the two witnessed semantic failures in that pool plus six source controls,
retained 64 source rows, and replayed eight independent validation rows. The
candidate produced one validation gain without a regression: training 7/8 to
7/8 equivalent and validation 6/8 to 7/8. The known selected failure remained
incorrect, so this is a development signal, not closure. Retained inequalities
were still unsatisfied and the bounded operation search was incomplete.

Receipt: `semantic-operation-pointer-targeted-20260918.json`,
`d4201664a60f74ed6614fe6d16a19e0dcbf76d2e97eb5721ca5c3657c5ec114`.

The next step is a complete candidate comparison with the same source-only
retention and an independently replayed 500-row validation set. G03 remains
open until the full comparison, fresh transfer and controls are measured.

## Completed full comparison

The full comparison completed all 764 source-training rows and all 500
validation rows. The incumbent remains selected:

| Arm | Exact / equivalent | Gains vs incumbent | Regressions vs incumbent |
| --- | --- | --- | --- |
| Incumbent | 488/500 | 0 | 0 |
| Joint-scoring fit start | 474/500 | 7 | 21 |
| Boundary/binding refit | 474/500 | 7 | 21 |

The fit retained 28,143 comparisons, including 14 witnessed runtime errors.
It accepted zero updates. All 1,363 initially nonpositive margins remained
nonpositive, and the loss stayed at 0.23122249410128787. The optimizer stopped
with `no_retention_preserving_step_found`. None of the 764 bounded operation
searches exhausted its grammar. This candidate is rejected.

The pre-fit control attributes the measured regression to the joint-scoring
decoder change; this training run did not alter those decisions.

Candidate: `semantic-operation-pointer-full-20260918.json`, receipt
`2ff90b36652a22f65263ef3112073582dedaccf234e37f1d1e4f267dcd957b68`.
Selection: `semantic-operation-pointer-full-20260918.validation.json`, report
`1ee19d2716de215200ac8f44a1d2fbb93664a77a8f34bae72b839dae6ea8b26f`.

The optimizer initially projected against only 32 protected comparisons;
83 source comparisons already lay on their retention floor. A focused
counterexample reproduces an avoidable stall when an omitted protected
comparison blocks the initial direction. The next repair lets failed
proposals add constraints to direction selection while retaining the full
float32 acceptance check. The numerical defect is reproduced; its effect on
this full corpus still needs measurement.

## Protected-direction numerical repair

Two optimizer defects now have reproducing tests. Omitted protected
comparisons can block a step even when a feasible direction exists; rejected
proposals now add those comparisons to the projection. Separately, the old
L-BFGS-B dual solve accepted small violating directions under its absolute
stopping tolerance. The equivalent nonnegative least-squares solve normalizes
the direction before projection. Tests cover gradient scales from 1e-12 to
1e6, dependent constraint normals, and a protected face beyond the initial
32-comparison batch. Float32 export and all-comparison acceptance stay intact.

The 64-row source-acquisition/retention trial now fixes the selected training
error: 7/8 to 8/8 equivalent with no training regression. Validation remains
6/8, with no correct-to-incorrect regression. One already-incorrect validation
case changes to `typed_argument_chart_empty`; this is still a failure.
The fit takes all 2,504 retained margins positive (148 were initially
nonpositive), with zero retained-positive regressions. It reaches a minimum
margin of 0.09999999485883215 and reports its 64-step budget exhausted,
not exact satisfaction of the requested 0.1 margin.

Receipt: `semantic-operation-pointer-nnls-20260918.json`,
`6f8dbc677ce14eb9f1956f4236b6ff520e3defc6756eb9d2637f7b71f9c35dbd`.
The preceding constraint-discovery-only trial accepted five updates but left
69 nonpositive margins; its receipt is
`d577494f8513e5b931b626372d0d2ba76102d6dc3992f7769c2af75e41f499ec`.

Eighty-two focused tests pass. The small trial establishes that the numerical
repair enables learning of a witnessed error; it does not establish transfer.
The next full fit must re-mine runtime competitors after coefficient updates
and compare fresh decoding against the unchanged incumbent. No promotion,
serving change, or G-ledger closure follows from the retained-margin result.
