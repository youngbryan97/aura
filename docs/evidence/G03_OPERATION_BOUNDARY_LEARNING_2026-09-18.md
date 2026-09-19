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
