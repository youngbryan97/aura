# G03 Architecture Reset

## Decision

Stop the sequence of local coefficient variants followed by repeated checks
on the same exposed validation population. Keep every failed candidate and
its measurements. Do not promote the paired-boundary candidate, widen a
passing threshold, or treat completion of its optimizer as progress toward
transfer.

The next candidate class is a learned full-request structured recognizer,
compared with the frozen local-head architecture. Reuse the resident hidden
sequence, language substrate, typed semantic IR, graph construction, and
universal-floor executor. Do not build another language store or interpreter.
The recognizer must make operation boundaries, labels, and references jointly
available to selection; exact execution cannot rescue a wrong interpretation.

This is an architectural experiment, not a claim that an attention layer
guarantees generalization. No implementation or successful result is implied
by this decision record.

## Evidence For Changing Course

The converged paired fit makes the target boundary set optimal on 764/764
source-training examples. On development validation it does so on 356/500,
against the parent's 300/500. Despite that boundary-only improvement, its
end-to-end selected pilot regresses from 39/54 to 37/54. Boundary ranking,
label evidence, and final graph selection must be measured together.

The exact boundary-only diagnostic is
`semantic-boundary-separability-development-20260921/report.json`, receipt
`a316ce71551b6ce3cee68c42f56c2e0fe73fde7caadb86d2991da4c5352dec4b`.
Its first attempt encountered the archive's test split and stopped before
scoring that row. The completed attempt explicitly filters train/validation
before publishing its plan and measures exactly 1,264 examples. It performs
no training and is not semantic-accuracy evidence.

The resident features declare one causal forward pass. Local pointer scores
use start/end hidden states, optionally their diagonal interaction. They do
not directly reconsider those states in light of the entire request.
This motivates testing full-request contextualization; it does not prove
that current features cannot solve the task. An exact-prefix collision audit
found zero contradictory endpoint annotations among these 1,264 examples.
Receipt: `c3a7938ffbdb54374d4761907df076a40eec9d25d0ac3514ad96a2e687a72d10`.

## Experiment Order

1. Freeze source-construction folds and acquisition provenance. Choose model
   capacity and training settings only inside source-training folds. Repeatedly
   inspected validation rows remain development data, never fresh evidence.
2. Compare local heads with a trainable full-context encoder under the same
   data, decoder, and measured work allowances. Include random-encoder and
   removed-context controls. Measure complete graphs, not token F1 alone.
3. Use existing program-first counterfactual generation to vary names,
   values, clause order, and dependencies independently. Hold entire
   transformation families out. Audit unsupported generation explicitly.
4. Retain source successes and all failure categories. Distinguish training
   underfit, source-construction transfer failure, absent candidates,
   incorrect selection, and public-answer failure. Each calls for a different
   intervention; another optimizer run is not the default response.
5. Freeze the selected candidate before complete development and fresh
   replication. On failure, decide whether to expand acquisition, change
   representation, or reject the architecture using the source-fold evidence.

## Broader Programme

### Source Fold Implementation

The source-only fold builder joins construction families and contrast lineage
transitively before assignment. It rejects validation/test inputs, repeated
source identities, and populations with too few independent groups. On the
764 source examples, it yields 42 independent groups and fold sizes
256/254/254. Frozen plan receipt:
`9c6413bfdf573d03b4239ac82577001ee6f7499561ede194b24cc07022801b94`
at `semantic-architecture-source-folds-20260921/folds.json`.
Six focused tests pass. This implements the split contract only; the
full-context architecture comparison has not run.

The subsequent smoke run passed 164 tests but failed the live serving alarm:
`semantic_neural_activation_invalid:source_drift:core/brain/llm/qualified_recurrent_ingress.py`.
The inspected change in commit `15d45a58d` adds explanatory comments only.
Its byte-bound activation still needs an authorized provenance disposition;
the evidence was not resealed or the alarm suppressed.

G03 is a bounded language-to-program obligation. It must not indefinitely
serialize the independent G09 ordinary-runtime reasoning experiment behind
a perfect synthetic parser. Develop the ordinary runtime comparison and
cross-backend procedure path separately, with the unqualified compiler kept
out of serving authority. Neither those activities nor this architecture
reset close G03, G09, G10, or G12. Frontier standing still requires actual
matched measurements against named external references.
